"""Draw the fields located on a real-booklet photo (debugging aid).

    python -m tools.show_booklet "data/Paper Registry/1-4.jpg" out.png
"""

import sys

import cv2

from extraction import ocr
from extraction.booklet import analyse, classify, locate
from extraction.imageproc import load_image


def main(src, dst):
    img = load_image(src)
    lines = ocr.read_page(img)
    lay, score = classify(lines)
    print("layout", lay.name if lay else None, round(score, 2))
    if not lay:
        return
    page = analyse(img, lines)
    loc = locate(page, lay)
    vis = page.img.copy()
    for r in loc.regions:
        if r.kind in ("text", "cell") and r.rect[2] > r.rect[0]:
            cv2.rectangle(vis, r.rect[:2], r.rect[2:], (0, 160, 0), 1)
            cv2.putText(vis, r.key.split(".")[-1][:10], (r.rect[0] + 2, r.rect[3] - 2), cv2.FONT_HERSHEY_PLAIN, 0.7,
                        (0, 120, 0), 1)
        elif r.kind == "check" and r.extra.get("found"):
            cv2.rectangle(vis, r.rect[:2], r.rect[2:], (255, 0, 0), 2)
        elif r.kind == "radio":
            for v, b in r.extra["options"].items():
                if b:
                    cv2.rectangle(vis, b[:2], b[2:], (255, 0, 255), 2)
    for keys, b in loc.spans:
        cv2.rectangle(vis, b[:2], b[2:], (0, 0, 255), 2)
    for b in loc.pii:
        cv2.rectangle(vis, b[:2], b[2:], (0, 0, 0), -1)
    print("regions", len(loc.regions), "spans", len(loc.spans), "warnings", loc.warnings)
    cv2.imwrite(dst, vis)


if __name__ == "__main__":
    main(*sys.argv[1:3])
