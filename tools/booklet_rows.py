"""Measure the visit-table row positions on the left pregnancy page of the real booklet
(rows are labelled there) relative to the shaded section bands, for use on the right page.

    python -m tools.booklet_rows "data/Paper Registry/1-4.jpg"
"""

import sys

import numpy as np

from extraction import ocr
from extraction.booklet import LAYOUT_BY_NAME, _row_band, analyse, bands, find_label, find_labels
from extraction.imageproc import load_image


def main(path):
    img = load_image(path)
    page = analyse(img, ocr.read_page(img))
    lay = LAYOUT_BY_NAME["booklet_pregnancy_left"]
    table = [f for f in lay.fields if getattr(f, "prefix", None) == "visits"][0]
    cols = [c for c in (find_label(page, lab, min_score=85) for _, lab in table.cols) if c]
    xc = float(np.median([c.xc for c in cols]))
    hdr_y = max(c.yc for c in cols)
    b = [y for y in bands(page) if y > hdr_y]
    assert len(b) == 4, b
    x_lim = min(c.x0 for c in cols)
    anchors = {}
    for k, lab in table.rows:
        fs = [f for f in find_labels(page, lab, 80 if len(lab) > 3 else 95, x_max=x_lim) if f.yc > hdr_y + 5]
        if not fs:
            print("missing row label", lab)
            continue
        f = fs[0]
        a, _ = _row_band(page, f.yc, f.xc)
        y = f.yc + (a.a if a else 0.0) * (xc - f.xc)
        i = 0 if y < b[1] else (1 if y < b[2] else (2 if y < b[3] else 3))
        frac = 0.0 if i == 3 else (y - b[i]) / (b[i + 1] - b[i])
        anchors[k] = (i, round(float(frac), 4))
    print("bands", [round(v, 1) for v in b])
    print("ROW_ANCHORS =", anchors)


if __name__ == "__main__":
    main(sys.argv[1])
