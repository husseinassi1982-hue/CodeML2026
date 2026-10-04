"""Score the extractor on the 5 real booklet photos against the hand transcription in
evaluation/real_photos_truth.json.

    python -m tools.evaluate_real [--show]

Scoring (per transcribed field):
  value / accept  the extracted value matches (also accepted in an `alt` neighbouring row)
  status          the extracted status matches (e.g. NOT_PROVIDED for a blank field)
  span            a word written across rows: correct if read as that word or sent to review
Reported: correct, silent errors (KNOWN but wrong), sent to review (NEEDS_REVIEW/ILLEGIBLE).
"""

import argparse
import json
import re
import time

from extraction import extract
from extraction.catalog import get_field
from extraction.normalize import parse
from extraction.pdf_layout import fold
from tools.paths import EVAL_DIR, REGISTRY

TRUTH = EVAL_DIR / "real_photos_truth.json"


def _sq(x):
    return re.sub(r"[\s.,;:'’\-_/()]", "", fold(str(x)))


def matches(f, exp, got) -> bool:
    if got is None:
        return False
    if "accept" in exp:
        return any(_sq(a) == _sq(v) for a in exp["accept"] for v in (got.value, got.display, got.raw) if v is not None)
    if "value" in exp:
        want = exp["value"]
        if isinstance(want, bool):
            return got.value is want
        gp = parse(str(want), f)
        if gp.ok and gp.value is not None:
            gv = got.value
            if isinstance(gp.value, float) or isinstance(gv, float):
                try:
                    return abs(float(gp.value) - float(gv)) < 1e-6
                except (TypeError, ValueError):
                    return False
            if isinstance(gp.value, str) and isinstance(gv, str):
                return _sq(gp.value) == _sq(gv)
            return gp.value == gv
        return any(_sq(want) == _sq(v) for v in (got.value, got.display, got.raw) if v is not None)
    return False


def score_field(page_type, key, exp, fields):
    f = get_field(page_type, key)
    got = fields.get(key)
    if "span" in exp:
        if got is None:
            return False, "missing"
        ok = got.status.value in ("NEEDS_REVIEW", "ILLEGIBLE") or (
            got.status.value == "KNOWN" and _sq(got.value or got.display or "") == _sq(exp["span"]))
        return ok, got.status.value
    if "status" in exp and not ("value" in exp or "accept" in exp):
        return (got is not None and got.status.value == exp["status"]), (got.status.value if got else "missing")
    cands = [key] + exp.get("alt", [])
    for k in cands:
        g = fields.get(k)
        if g is not None and matches(f, exp, g):
            if "status" in exp and g.status.value != exp["status"]:
                continue
            return True, g.status.value
    if "status" in exp and got is not None and got.status.value == exp["status"]:
        return True, got.status.value
    return False, (got.status.value if got else "missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))["photos"]
    tot = ok = silent = review = 0
    rows = []
    for photo, info in truth.items():
        t0 = time.time()
        r = extract(REGISTRY / photo)
        dt = time.time() - t0
        n_ok = 0
        for key, exp in info["fields"].items():
            good, st = score_field(info["page_type"], key, exp, r.fields)
            got = r.fields.get(key)
            tot += 1
            ok += good
            n_ok += good
            if st == "KNOWN" and not good:
                silent += 1
            if st in ("NEEDS_REVIEW", "ILLEGIBLE"):
                review += 1
            rows.append((photo, key, exp, good, st, got))
        print(f"{photo}: {r.page_type} [{r.layout}] {n_ok}/{len(info['fields'])} correct, {dt:.1f}s  {r.summary}")
    print(f"\nreal photos: {ok}/{tot} correct ({ok / tot:.0%}) | silent errors {silent} ({silent / tot:.0%}) | "
          f"sent to review {review} ({review / tot:.0%})")
    (EVAL_DIR / "results_real_photos.json").write_text(json.dumps(
        {"photos": len(truth), "fields": tot, "correct": ok, "silent_errors": silent, "sent_to_review": review},
        indent=2))
    if a.show:
        for photo, key, exp, good, st, got in rows:
            flag = "ok " if good else ("SILENT" if st == "KNOWN" else "  -  ")
            g = f"{got.display!r} raw={got.raw!r} {got.confidence:.2f}" if got else "-"
            print(f"  {flag:6s} {photo} {key:42s} expected={exp}  got [{st}] {g}")


if __name__ == "__main__":
    main()
