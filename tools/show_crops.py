"""Draw what the OCR actually reads on each photo (debugging aid for word cropping).

    python -m tools.show_crops "data/Paper Registry/" eval_out/crops [--only 1-]

For every image, two files in the output folder:
  <name>_crops.png   the page: green = where the field is expected, red = the crop sent to the
                     OCR, label = field key and what the OCR read in it. Identifiers blacked out.
  <name>_sheet.png   every crop exactly as the recogniser receives it, with its reading.
Real booklet pages are drawn on the deskewed page; specimen pages on the original photo.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from extraction import booklet, ocr
from extraction.align import align
from extraction.backends.local import MIN_PAGE_CONFIDENCE
from extraction.catalog import fields_for
from extraction.imageproc import CROP_SCALE, analyze_text_crop, load_image, text_crop_for_ocr, warp_region
from extraction.template import load_templates, printed_crop

GREEN, RED, BLUE = (0, 170, 0), (0, 0, 230), (200, 80, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(vis, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, FONT, 0.4, 1)
    y = max(h + 2, y)
    cv2.rectangle(vis, (x, y - h - 2), (x + w + 2, y + 2), (255, 255, 255), -1)
    cv2.putText(vis, text, (x + 1, y), FONT, 0.4, color, 1, cv2.LINE_AA)


def _sheet(items, width=1400):
    """items: (key, crop, reading) -> one image, crops left, key + reading on the right."""
    rows = []
    for key, crop, reading in items:
        h, w = crop.shape[:2]
        s = min(1.0, 600 / w, 90 / h) if w and h else 1.0
        c = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        row = np.full((max(c.shape[0], 34) + 6, width, 3), 255, np.uint8)
        row[3:3 + c.shape[0], 4:4 + c.shape[1]] = c
        cv2.putText(row, key, (620, 16), FONT, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(row, reading, (620, 32), FONT, 0.5, RED, 1, cv2.LINE_AA)
        cv2.line(row, (0, row.shape[0] - 1), (width, row.shape[0] - 1), (220, 220, 220), 1)
        rows.append(row)
    return np.vstack(rows) if rows else np.full((40, width, 3), 255, np.uint8)


def _reading(r):
    return f'"{r[0]}" ({r[1]:.2f})' if r else "-"


def show_booklet(img, lines, lay):
    page = booklet.analyse(img, lines)
    loc = booklet.locate(page, lay)
    inks = booklet.word_crops(page, loc)
    keys = [k for k, i in inks.items() if i.present and not i.is_dash]
    small = dict(zip(keys, ocr.read_crops([inks[k].clean for k in keys], model="small")))
    medium = (dict(zip(keys, ocr.read_crops([inks[k].clean for k in keys], model="medium")))
              if ocr.medium_available() else {})
    vis = page.img.copy()
    for r in loc.regions:
        if r.key not in inks:
            continue
        cv2.rectangle(vis, r.rect[:2], r.rect[2:], GREEN, 1)
        ink = inks[r.key]
        if ink.box and ink.present:
            cv2.rectangle(vis, ink.box[:2], ink.box[2:], BLUE if ink.is_dash else RED, 2)
            text = "dash" if ink.is_dash else (small.get(r.key) or ("", 0))[0]
            _label(vis, f"{r.key.split('.')[-1][:12]}: {text}", ink.box[0], ink.box[1] - 2, RED)
    for _, b in loc.spans:
        cv2.rectangle(vis, b[:2], b[2:], BLUE, 2)
    for b in loc.pii:
        cv2.rectangle(vis, b[:2], b[2:], (0, 0, 0), -1)
    items = [(k, inks[k].clean, f"small {_reading(small.get(k))}   medium {_reading(medium.get(k))}") for k in keys]
    return vis, items


def show_specimen(img, al):
    tpl = load_templates()[al.page_type]
    H = al.H

    def to_img(rect):
        x0, y0, x1, y1 = rect
        q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(q, H).reshape(-1, 2).astype(np.int32)

    vis = img.copy()
    todo = []
    for f in fields_for(al.page_type):
        geo = tpl.fields.get(f.key)
        if geo is None or f.kind not in ("text", "cell"):
            continue
        cv2.polylines(vis, [to_img(geo["region"])], True, GREEN, 1)
        ink = analyze_text_crop(warp_region(img, H, geo["region"]), printed=printed_crop(al.page_type, geo["region"]))
        if ink.present:
            todo.append((f, geo, ink))
    need = [(f, geo, ink) for f, geo, ink in todo if not ink.is_dash]
    crops = [text_crop_for_ocr(ink) for _, _, ink in need]
    reads = ocr.read_crops(crops)
    read_of = {f.key: r for (f, _, _), r in zip(need, reads)}
    for f, geo, ink in todo:
        x0, y0 = geo["region"][:2]
        pad = 0 if ink.is_dash else 2  # text_crop_for_ocr adds 2 pt around the ink
        bx0, by0, bx1, by1 = ink.bbox
        rect = (x0 + bx0 / CROP_SCALE - pad, y0 + by0 / CROP_SCALE - pad,
                x0 + bx1 / CROP_SCALE + pad, y0 + by1 / CROP_SCALE + pad)
        q = to_img(rect)
        cv2.polylines(vis, [q], True, BLUE if ink.is_dash else RED, 2)
        text = "dash" if ink.is_dash else read_of[f.key][0]
        _label(vis, f"{f.key.split('.')[-1][:12]}: {text}", int(q[:, 0].min()), int(q[:, 1].min()) - 2, RED)
    for key, geo in tpl.fields.items():
        if geo.get("pii"):
            cv2.fillPoly(vis, [to_img(geo["region"])], (0, 0, 0))
    items = [(f.key, c, _reading(r)) for (f, _, _), c, r in zip(need, crops, reads)]
    return vis, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="an image or a folder of images")
    ap.add_argument("out", nargs="?", default="eval_out/crops")
    ap.add_argument("--only", default="", help="only files whose name starts with this")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir()
                                               if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for path in files:
        if not path.name.startswith(a.only):
            continue
        img = load_image(path)
        lines = ocr.read_page(img)
        al = align(lines, img)
        if al is not None and al.confidence >= MIN_PAGE_CONFIDENCE:
            kind = al.page_type
            vis, items = show_specimen(img, al)
        else:
            lay, _ = booklet.classify(lines)
            if lay is None:
                print(f"{path.name}: page not recognised, skipped")
                continue
            kind = lay.name
            vis, items = show_booklet(img, lines, lay)
        cv2.imwrite(str(out / f"{path.stem}_crops.png"), vis)
        cv2.imwrite(str(out / f"{path.stem}_sheet.png"), _sheet(items))
        print(f"{path.name}: {kind}, {len(items)} crops read")
        ocr.trim_memory()


if __name__ == "__main__":
    main()
