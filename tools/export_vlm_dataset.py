"""Export field crops + answers to fine-tune a vision-language model (notebooks/finetune_vlm_ocr.ipynb).

    python -m tools.export_vlm_dataset                 # synthetic specimen pages -> eval_out/vlm_dataset.zip
    python -m tools.export_vlm_dataset --real          # ALSO the real booklet photos, local folder only

Synthetic (safe to upload to Colab: the organisers' synthetic data, no real patient):
  every text/cell field of the specimen pages, cut as the field's area plus a margin (writing spills
  over the rows), clean and as simulated phone photos (mild/medium/hard). Answers = the specimen's
  ground truth as written. Patients 1-7 -> train, 8-10 -> val (held out, as in tools/evaluate.py).
  Empty fields are kept (answer EMPTY) at a lower rate so the model learns to say so.

Real (--real, eval_out/vlm_dataset_real/, NEVER zipped for upload): the 5 booklet photos with the
hand transcription (evaluation/real_photos_truth.json) and the crop editor's true texts. For training
or testing on the team's own GPU only: the brief forbids sending real patient data to a third party.

Identifier fields are never exported.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile

import cv2
import numpy as np

from extraction import booklet, ocr
from extraction.align import align
from extraction.catalog import get_field
from extraction.imageproc import load_image, warp_region
from extraction.template import load_templates
from tools.degrade import degrade
from tools.paths import REGISTRY, REPO
from tools.vlm_common import EMPTY, MARGIN_X, MARGIN_Y, prompt

OUT = REPO / "eval_out" / "vlm_dataset"
OUT_REAL = REPO / "eval_out" / "vlm_dataset_real"
TRAIN, VAL = range(1, 8), range(8, 11)
LINE_PT = 10.0  # printed line height on the specimen, in PDF points
SCALE = 3.0  # px per point in the crops (~215 dpi)
EMPTY_RATE = 0.25


def answer_of(g: dict) -> str | None:
    raw = g.get("raw")
    if g.get("status") == "NOT_PROVIDED" or (raw is None and g.get("value") in (None, "")):
        return EMPTY
    if not isinstance(raw, str) or not raw.strip() or "�" in raw:  # NanumPen font gaps: unknowable
        return None
    return raw.strip()


def specimen(rng: random.Random) -> list[dict]:
    gt = json.loads((REPO / "evaluation" / "ground_truth.json").read_text(encoding="utf-8"))["pages"]
    tpls = load_templates()
    rows = []
    for page_name, page in sorted(gt.items()):
        split = "train" if page["patient"] in TRAIN else "val" if page["patient"] in VAL else None
        if split is None:
            continue
        variants = [None, "mild", "medium", "hard"] if split == "train" else [None, "medium"]
        for img_name in page["images"]:
            clean = load_image(REGISTRY / img_name)
            for vi, level in enumerate(variants):
                img = clean if level is None else degrade(clean, level, seed=hash((img_name, vi)) % 10_000)
                lines = ocr.read_page(img)
                al = align(lines, img)
                if al is None or al.page_type != page["page_type"]:
                    print(f"  skip {img_name} [{level}]: page not aligned")
                    continue
                tpl = tpls[al.page_type]
                for key, g in page["fields"].items():
                    geo = tpl.fields.get(key)
                    try:
                        f = get_field(al.page_type, key)
                    except KeyError:
                        continue
                    if geo is None or f.pii or f.kind not in ("text", "cell"):
                        continue
                    ans = answer_of(g)
                    if ans is None or (ans == EMPTY and rng.random() > EMPTY_RATE):
                        continue
                    x0, y0, x1, y1 = geo["region"]
                    rect = (x0 - MARGIN_X * LINE_PT, y0 - MARGIN_Y * LINE_PT, x1 + MARGIN_X * LINE_PT, y1 + MARGIN_Y * LINE_PT)
                    crop = warp_region(img, al.H, rect, scale=SCALE)
                    name = f"{split}/{img_name.rsplit('.', 1)[0]}__{level or 'clean'}__{key}.jpg"
                    save(OUT, name, crop)
                    rows.append(dict(image=name, split=split, key=key, dtype=f.dtype, page=img_name,
                                     variant=level or "clean", prompt=prompt(f.label_fr, f.label_en, f.dtype),
                                     answer=ans))
                ocr.trim_memory()
            print(f"  {img_name}: {sum(1 for r in rows if r['page'] == img_name)} crops")
    return rows


def real() -> list[dict]:
    """The real booklet photos (local use only)."""
    truth = json.loads((REPO / "evaluation" / "real_photos_truth.json").read_text(encoding="utf-8"))["photos"]
    labels_path = REPO / "evaluation" / "crop_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["images"] if labels_path.exists() else {}
    rows = []
    for photo, spec in truth.items():
        img = load_image(REGISTRY / photo)
        lines = ocr.read_page(img)
        lay, _ = booklet.classify(lines)
        page = booklet.analyse(img, lines)
        loc = booklet.locate(page, lay)
        regions = {r.key: r for r in loc.regions}
        # identifiers (name, ID, phone, address...) are painted out before cutting: the crops' margins
        # reach into the neighbouring lines
        masked = page.img.copy()
        paper = np.median(masked.reshape(-1, 3), axis=0)
        for x0, y0, x1, y1 in loc.pii:
            masked[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = paper
        answers = {}
        for k, e in spec["fields"].items():
            if e.get("span"):
                continue
            if e.get("status") in ("NOT_PROVIDED",) and "value" not in e and "accept" not in e:
                answers[k] = EMPTY
            else:
                v = e.get("value", (e.get("accept") or [None])[0])
                if v is not None and not isinstance(v, bool):
                    answers[k] = str(v)
        for k, e in labels.get(photo, {}).get("fields", {}).items():
            if e.get("text"):
                answers[k] = e["text"]
            elif e.get("empty"):
                answers[k] = EMPTY
        for k, ans in answers.items():
            r = regions.get(k)
            try:
                f = get_field(lay.page_type, k)
            except KeyError:
                continue
            if r is None or f.pii or r.kind not in ("text", "cell"):
                continue
            x0, y0, x1, y1 = r.rect
            mx, my = int(MARGIN_X * page.th), int(MARGIN_Y * page.th)
            h, w = page.img.shape[:2]
            crop = masked[max(0, y0 - my):min(h, y1 + my), max(0, x0 - mx):min(w, x1 + mx)]
            if crop.size == 0:
                continue
            name = f"real/{photo.rsplit('.', 1)[0]}__{k}.jpg"
            save(OUT_REAL, name, crop)
            rows.append(dict(image=name, split="real", key=k,
                             dtype=f.dtype, page=photo, variant="real", prompt=prompt(f.label_fr, f.label_en, f.dtype),
                             answer=ans))
        ocr.trim_memory()
        print(f"  {photo}: {sum(1 for r in rows if r['page'] == photo)} crops")
    return rows


def save(out, name: str, crop):
    p = out / name
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])


def write(rows: list[dict], out, zip_name: str | None):
    for split in sorted({r["split"] for r in rows}):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                if r["split"] == split:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "README.txt").write_text(__doc__, encoding="utf-8")
    if zip_name:
        z = out.parent / zip_name
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out.parent))
        print(f"-> {z.relative_to(REPO)} ({z.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also export the real booklet photos (local folder, never zipped)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for out in (OUT, OUT_REAL):
        if out.exists():
            shutil.rmtree(out)
    rows = specimen(rng)
    for split in ("train", "val"):
        rs = [r for r in rows if r["split"] == split]
        print(f"{split}: {len(rs)} crops ({sum(r['answer'] == EMPTY for r in rs)} empty)")
    write(rows, OUT, "vlm_dataset.zip")
    if a.real:
        rr = real()
        print(f"real: {len(rr)} crops -> {OUT_REAL.relative_to(REPO)} (local only: do not upload)")
        write(rr, OUT_REAL, None)


if __name__ == "__main__":
    main()
