"""Score the extractor field by field against evaluation/ground_truth.json.

    python -m tools.evaluate --split test                  # clean renders, patients 8-10
    python -m tools.evaluate --split all --degrade medium  # simulated phone photos
    python -m tools.evaluate --pages 3,11 --show-errors

Splits: dev = patients 1-7 (used while building), test = patients 8-10 (held out).

Metrics
  field accuracy     status right AND (when something is written) value right
  value accuracy     on fields with a written value: the pre-filled value is right
  status accuracy    KNOWN / NOT_PROVIDED / NOT_APPLICABLE... right
  review rate        written fields sent to the midwife (NEEDS_REVIEW / ILLEGIBLE)
  silent errors      fields marked KNOWN that are wrong -- the number that matters for
                     trust: the agent claimed certainty and was wrong
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2

from extraction import extract
from extraction.catalog import get_field
from extraction.normalize import parse
from extraction.pdf_layout import fold
from tools.degrade import degrade
from tools.paths import EVAL_DIR, GROUND_TRUTH, REGISTRY, REPO

SPLITS = {"dev": range(1, 8), "test": range(8, 11), "all": range(1, 11)}


def _squash(s) -> str:
    return re.sub(r"[\s.,;:'’\-_/]", "", fold(str(s)))


def value_correct(f, g: dict, p) -> bool:
    if f.kind in ("check", "radio"):
        return p.value == g.get("value")
    raw = g.get("raw")
    if not raw or g["status"] != "KNOWN":
        return True  # nothing to read: only the status matters
    if g.get("glyph_missing"):  # the image shows a gap where an accent should be
        pat = re.escape(_squash(raw)).replace(re.escape(_squash("�")), ".?")
        return any(re.fullmatch(pat, _squash(x)) for x in (p.raw, p.display, p.value) if x is not None)
    gp = parse(raw, f)
    if gp.ok and p.value is not None and f.dtype not in ("str", "code"):
        if isinstance(gp.value, float) or isinstance(p.value, float):
            try:
                return abs(float(gp.value) - float(p.value)) < 1e-6
            except (TypeError, ValueError):
                return False
        return gp.value == p.value
    return any(_squash(raw) == _squash(x) for x in (p.value, p.display, p.raw) if x is not None)


def status_correct(g: dict, p) -> bool:
    if p.status.value == g["status"]:
        return True
    return bool(g.get("glyph_missing")) and g["status"] == "NOT_APPLICABLE" and p.status.value == "NOT_PROVIDED"


def evaluate(pages: list[str], degrade_level: str | None, show_errors: bool, save_images: Path | None):
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["pages"]
    rows, page_rows = [], []
    for pg in pages:
        info = gt[pg]
        path = REGISTRY / info["images"][0]
        img = cv2.imread(str(path))
        if degrade_level:
            img = degrade(img, degrade_level, seed=int(pg))
            if save_images:
                save_images.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_images / f"page{int(pg):02d}_{degrade_level}.jpg"), img)
        t0 = time.time()
        r = extract(img)
        dt = time.time() - t0
        page_ok = r.page_type == info["page_type"]
        page_rows.append({"page": pg, "expected": info["page_type"], "got": r.page_type, "ok": page_ok,
                          "seconds": round(dt, 2), "align_conf": r.page_type_confidence})
        for key, g in info["fields"].items():
            f = get_field(info["page_type"], key)
            p = r.fields.get(key) if page_ok else None
            if p is None:
                rows.append({"page": pg, "page_type": info["page_type"], "key": key, "kind": f.kind,
                             "gt_status": g["status"], "pred_status": "MISSING", "status_ok": False,
                             "value_ok": False, "conf": 0.0, "gt": g.get("raw", g.get("value")), "pred": None})
                continue
            rows.append({"page": pg, "page_type": info["page_type"], "key": key, "kind": f.kind,
                         "gt_status": g["status"], "pred_status": p.status.value, "status_ok": status_correct(g, p),
                         "value_ok": value_correct(f, g, p), "conf": p.confidence,
                         "gt": g.get("raw", g.get("value")), "pred": p.display if p.display is not None else p.raw})
        n_bad = sum(1 for x in rows if x["page"] == pg and not (x["status_ok"] and x["value_ok"]))
        print(f"  page {pg:>2} {info['page_type']:26s} -> {r.page_type:26s} {dt:4.1f}s  {n_bad:3d} wrong / "
              f"{len(info['fields'])}", flush=True)
    return rows, page_rows


def summarise(rows, page_rows) -> dict:
    def block(rs):
        n = len(rs)
        written = [x for x in rs if x["gt_status"] == "KNOWN"]
        known_pred = [x for x in rs if x["pred_status"] == "KNOWN"]
        silent = [x for x in known_pred if not (x["status_ok"] and x["value_ok"])]
        review = [x for x in written if x["pred_status"] in ("NEEDS_REVIEW", "ILLEGIBLE")]
        return {
            "fields": n,
            "field_accuracy": round(sum(x["status_ok"] and x["value_ok"] for x in rs) / n, 4) if n else None,
            "value_accuracy_on_written": round(sum(x["value_ok"] for x in written) / len(written), 4) if written else None,
            "status_accuracy": round(sum(x["status_ok"] for x in rs) / n, 4) if n else None,
            "review_rate_on_written": round(len(review) / len(written), 4) if written else None,
            "silent_error_rate": round(len(silent) / len(known_pred), 4) if known_pred else None,
            "silent_errors": len(silent),
        }

    out = {"overall": block(rows), "by_kind": {}, "by_page_type": {},
           "page_type_accuracy": round(sum(p["ok"] for p in page_rows) / len(page_rows), 4),
           "seconds_per_page": round(sum(p["seconds"] for p in page_rows) / len(page_rows), 2)}
    for kind in ("text", "cell", "check", "radio"):
        rs = [x for x in rows if x["kind"] == kind]
        if rs:
            out["by_kind"][kind] = block(rs)
    for pt in sorted({x["page_type"] for x in rows}):
        out["by_page_type"][pt] = block([x for x in rows if x["page_type"] == pt])
    # calibration of the confidence on written text fields
    bins = defaultdict(lambda: [0, 0])
    for x in rows:
        if x["gt_status"] == "KNOWN" and x["kind"] in ("text", "cell"):
            b = min(9, int(x["conf"] * 10)) / 10
            bins[b][0] += 1
            bins[b][1] += x["value_ok"]
    out["calibration"] = {f"{b:.1f}-{b + 0.1:.1f}": {"n": n, "accuracy": round(k / n, 3)}
                          for b, (n, k) in sorted(bins.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=list(SPLITS))
    ap.add_argument("--pages", help="comma-separated PDF page numbers (overrides --split)")
    ap.add_argument("--degrade", choices=["mild", "medium", "hard"])
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--save-images", action="store_true", help="keep the degraded images in eval_out/")
    ap.add_argument("--from-rows", help="re-summarise a saved eval_out/*_fields.json without re-running OCR")
    a = ap.parse_args()
    out_dir = REPO / "eval_out"
    out_dir.mkdir(exist_ok=True)

    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["pages"]
    if a.pages:
        pages = a.pages.split(",")
    else:
        pages = [pg for pg, v in gt.items() if v["patient"] in SPLITS[a.split]]
    name = f"{a.pages and 'pages' or a.split}_{a.degrade or 'clean'}"
    if a.from_rows:
        saved = json.loads(Path(a.from_rows).read_text())
        rows, page_rows, name = saved["rows"], saved["pages"], saved["name"]
    else:
        print(f"evaluating {len(pages)} pages ({name})")
        rows, page_rows = evaluate(pages, a.degrade, a.show_errors, out_dir / "images" if a.save_images else None)
        (out_dir / f"{name}_fields.json").write_text(
            json.dumps({"name": name, "rows": rows, "pages": page_rows}, ensure_ascii=False, default=str))
    summary = summarise(rows, page_rows)
    summary["run"] = {"name": name, "pages": len(page_rows)}
    EVAL_DIR.mkdir(exist_ok=True)
    if not a.pages and not name.startswith("pages"):
        (EVAL_DIR / f"results_{name}.json").write_text(json.dumps(summary, indent=2))

    o = summary["overall"]
    print(f"\n{name}: {o['fields']} fields | field acc {o['field_accuracy']:.1%} | value acc (written) "
          f"{o['value_accuracy_on_written']:.1%} | status acc {o['status_accuracy']:.1%} | review rate "
          f"{o['review_rate_on_written']:.1%} | silent errors {o['silent_errors']} ({o['silent_error_rate']:.2%}) | "
          f"page type acc {summary['page_type_accuracy']:.0%} | {summary['seconds_per_page']}s/page")
    for k, v in summary["by_kind"].items():
        print(f"  {k:6s} field acc {v['field_accuracy']:.1%}  silent {v['silent_errors']}")
    if a.show_errors:
        print("\nerrors:")
        for x in rows:
            if not (x["status_ok"] and x["value_ok"]):
                flag = "SILENT" if x["pred_status"] == "KNOWN" else "      "
                print(f"  {flag} p{x['page']:>2} {x['key']:42s} gt={x['gt']!r} [{x['gt_status']}]  "
                      f"pred={x['pred']!r} [{x['pred_status']} {x['conf']:.2f}]")


if __name__ == "__main__":
    main()
