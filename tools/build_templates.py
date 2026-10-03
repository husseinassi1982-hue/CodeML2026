"""Build extraction/data/templates.json from the specimen PDF (patient 1's 8 pages).

    python -m tools.build_templates [--pdf PATH]
"""

import argparse
import json

import numpy as np
import pymupdf

from extraction.catalog import PAGE_ORDER
from extraction.imageproc import ink_bbox
from extraction.pdf_layout import read_page
from extraction.template import TEMPLATES_PATH, build_template
from tools.paths import SPECIMEN_PDF


SCALE = 4.0


def add_label_ink(page, tpl):
    """Measure where the ink of each printed label really is (font bboxes include side
    bearings and line height); the aligner measures photos the same way."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, ::-1].copy()
    for lab in tpl["labels"]:
        x0, y0, x1, y1 = lab["bbox"]
        m = 1.0
        X0, Y0 = int((x0 - m) * SCALE), int((y0 - m) * SCALE)
        crop = img[max(0, Y0):int((y1 + m) * SCALE), max(0, X0):int((x1 + m) * SCALE)]
        bb = ink_bbox(crop) if crop.size else None
        if bb:
            lab["ink"] = [round(max(0, X0) / SCALE + bb[0] / SCALE, 2), round(max(0, Y0) / SCALE + bb[1] / SCALE, 2),
                          round(max(0, X0) / SCALE + bb[2] / SCALE, 2), round(max(0, Y0) / SCALE + bb[3] / SCALE, 2)]


def save_printed_mask(doc, page_type: str, tpl: dict):
    """Ink mask of the blank form, in template coordinates at SCALE px/pt.

    For each patient's copy of this page: render it, erase the handwriting and ticks, align
    it to the template. A pixel is printed if it is inked in >= 2 copies where it was not
    erased (each patient wrote in different places, so every printed line is seen)."""
    import cv2

    from extraction.imageproc import _ink
    from extraction.template import Template, printed_mask_path
    from tools.build_ground_truth import label_affine

    t = Template.from_dict(tpl)
    W, H = int(round(t.width * SCALE)), int(round(t.height * SCALE))
    votes = np.zeros((H, W), np.uint16)
    k = PAGE_ORDER.index(page_type)
    for i in range(k, len(doc), len(PAGE_ORDER)):
        page = doc[i]
        layout = read_page(page)
        M, _ = label_affine(t, layout)  # template pt -> page pt
        pix = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        ink = _ink(cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY))
        erased = np.zeros_like(ink)
        for g in layout.hw:
            for _, bb in g.chars:
                cv2.rectangle(erased, (int(bb[0] * SCALE) - 8, int(bb[1] * SCALE) - 8),
                              (int(bb[2] * SCALE) + 8, int(bb[3] * SCALE) + 8), 1, -1)
        for m in layout.marks:
            cv2.rectangle(erased, (int(m[0] * SCALE) - 3, int(m[1] * SCALE) - 3),
                          (int(m[2] * SCALE) + 3, int(m[3] * SCALE) + 3), 1, -1)
        A = M.copy()
        A[:, 2] *= SCALE  # same affine in pixel units
        ink_t = cv2.warpAffine(ink & (1 - erased), A, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        votes += ink_t
    mask = (votes >= 3).astype(np.uint8)
    # drop specks: overhanging pen strokes that a few patients share are not printed matter
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < 40)[0]
    mask[np.isin(lab, small[small > 0])] = 0
    cv2.imwrite(str(printed_mask_path(page_type)), mask * 255, [cv2.IMWRITE_PNG_BILEVEL, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(SPECIMEN_PDF))
    args = ap.parse_args()
    doc = pymupdf.open(args.pdf)
    out, n_problems = {}, 0
    for i, page_type in enumerate(PAGE_ORDER):
        tpl, problems = build_template(read_page(doc[i]), page_type)
        add_label_ink(doc[i], tpl)
        save_printed_mask(doc, page_type, tpl)
        out[page_type] = tpl
        print(f"{page_type:28s} {len(tpl['fields']):4d} fields  {len(tpl['labels']):3d} labels")
        for p in problems:
            print("   !", p)
        n_problems += len(problems)
    TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATES_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {TEMPLATES_PATH} ({n_problems} problems)")


if __name__ == "__main__":
    main()
