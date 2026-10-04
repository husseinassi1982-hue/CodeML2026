"""Live demo helper (notebooks/finetune_vlm_ocr.ipynb, section 8): one page photo -> every field with
writing, read by the current OCR pipeline and by the fine-tuned vision-language model, side by side.

    rows, page_type, vis = field_crops("photo.jpg")   # pipeline: page type, field crops, picture
    for r in rows: r["vlm"] = ask(r["image"], r["prompt"])

Identifier fields are never cut, and identifier areas are painted out of every crop.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from extraction import booklet, extract, ocr
from extraction.align import align
from extraction.backends.local import MIN_PAGE_CONFIDENCE
from extraction.catalog import fields_for, get_field
from extraction.imageproc import analyze_text_crop, load_image, warp_region
from extraction.template import load_templates, printed_crop
from tools.vlm_common import MARGIN_X, MARGIN_Y, prompt

LINE_PT, SCALE = 10.0, 3.0


def _pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def field_crops(path: str, only_written: bool = True):
    """-> (rows, page_type, picture): rows = {key, label, dtype, image, prompt, ocr, ocr_status}."""
    img = load_image(path)
    result = extract(path)  # the current pipeline, for comparison
    lines = ocr.read_page(img)
    al = align(lines, img)
    rows, vis = [], None
    if al is not None and al.confidence >= MIN_PAGE_CONFIDENCE:  # specimen layout
        tpl = load_templates()[al.page_type]
        page_type = al.page_type
        H = al.H
        pii = [g["region"] for g in tpl.fields.values() if g.get("pii")]
        masked = img.copy()
        for x0, y0, x1, y1 in pii:  # paint identifiers out of the photo itself
            q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64).reshape(-1, 1, 2)
            cv2.fillPoly(masked, [cv2.perspectiveTransform(q, H).reshape(-1, 2).astype(np.int32)], (0, 0, 0))
        vis = masked.copy()
        for f in fields_for(page_type):
            geo = tpl.fields.get(f.key)
            if geo is None or f.pii or f.kind not in ("text", "cell"):
                continue
            if only_written and not analyze_text_crop(warp_region(img, H, geo["region"]),
                                                      printed=printed_crop(page_type, geo["region"])).present:
                continue
            x0, y0, x1, y1 = geo["region"]
            rect = (x0 - MARGIN_X * LINE_PT, y0 - MARGIN_Y * LINE_PT, x1 + MARGIN_X * LINE_PT, y1 + MARGIN_Y * LINE_PT)
            crop = warp_region(masked, H, rect, scale=SCALE)
            q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64).reshape(-1, 1, 2)
            cv2.polylines(vis, [cv2.perspectiveTransform(q, H).reshape(-1, 2).astype(np.int32)], True, (0, 160, 0), 2)
            rows.append(_row(f, crop, result))
    else:  # real pink booklet
        lay, _ = booklet.classify(lines)
        if lay is None:
            return [], "unknown", img
        page_type = lay.page_type
        page = booklet.analyse(img, lines)
        loc = booklet.locate(page, lay)
        inks = booklet.word_crops(page, loc) if only_written else {}
        masked = page.img.copy()
        for x0, y0, x1, y1 in loc.pii:
            masked[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = (0, 0, 0)
        vis = masked.copy()
        h, w = masked.shape[:2]
        for r in loc.regions:
            try:
                f = get_field(page_type, r.key)
            except KeyError:
                continue
            if f.pii or r.kind not in ("text", "cell"):
                continue
            if only_written and not (r.key in inks and inks[r.key].present):
                continue
            x0, y0, x1, y1 = r.rect
            mx, my = int(MARGIN_X * page.th), int(MARGIN_Y * page.th)
            crop = masked[max(0, y0 - my):min(h, y1 + my), max(0, x0 - mx):min(w, x1 + mx)]
            if crop.size == 0:
                continue
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 160, 0), 2)
            rows.append(_row(f, crop, result))
    return rows, page_type, _pil(vis)


def _row(f, crop, result) -> dict:
    got = result.fields.get(f.key)
    return dict(key=f.key, label=f.label_fr, dtype=f.dtype, image=_pil(crop),
                prompt=prompt(f.label_fr, f.label_en, f.dtype),
                ocr=(got.display or got.raw or "") if got is not None else "",
                ocr_status=got.status.value if got is not None else "")
