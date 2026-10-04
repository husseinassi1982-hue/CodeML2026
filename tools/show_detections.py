"""Draw everything the extractor detects on a real booklet photo: field areas, the handwriting crop
read in each, diagonal writing, checkboxes / circled options, and each field's value and status.
Identifiers are blacked out.

    python -m tools.show_detections "data/Paper Registry/1-1.jpg" [out.png]
"""

import sys

import cv2
import numpy as np

from extraction import booklet, extract, ocr
from extraction.imageproc import load_image

COL = {"KNOWN": (0, 150, 0), "NEEDS_REVIEW": (0, 140, 255), "NOT_PROVIDED": (170, 170, 170),
       "NOT_APPLICABLE": (150, 150, 150), "ILLEGIBLE": (0, 0, 220), "UNKNOWN": (180, 0, 180)}


def draw(path: str, out: str, S: int = 2) -> dict:
    img = load_image(path)
    lines = ocr.read_page(img)
    lay, _ = booklet.classify(lines)
    if lay is None:
        raise SystemExit("not a real-booklet page")
    page = booklet.analyse(img, lines)
    loc = booklet.locate(page, lay)
    inks = booklet.word_crops(page, loc)
    res = extract(path).fields
    vis = cv2.resize(page.img, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    for b in loc.pii:
        cv2.rectangle(vis, (int(b[0] * S), int(b[1] * S)), (int(b[2] * S), int(b[3] * S)), (0, 0, 0), -1)

    def label(x, y, txt, col):
        (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x, y - h - 3), (x + w + 4, y + 3), (255, 255, 255), -1)
        cv2.putText(vis, txt, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    counts: dict[str, int] = {}
    for r in loc.regions:
        g = res.get(r.key)
        if g is None:
            continue
        st = g.status.value
        counts[st] = counts.get(st, 0) + 1
        col = COL.get(st, (0, 0, 0))
        tag = st.replace("NEEDS_REVIEW", "REVIEW").replace("NOT_APPLICABLE", "N/A")
        x0, y0, x1, y1 = (int(v * S) for v in r.rect)
        if r.kind in ("text", "cell") and x1 > x0:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (60, 180, 60), 1)
            ink = inks.get(r.key)
            if ink is not None and ink.present and ink.box:
                bx0, by0, bx1, by1 = (int(v * S) for v in ink.box)
                cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 0, 230), 2)
            if st != "NOT_PROVIDED":
                label(x0 + 2, y1 - 4, f"{r.key.split('.')[-1][:16]}: {(g.display or g.raw or '')[:16]} [{tag}]", col)
        elif r.kind == "check" and r.extra.get("found"):
            cv2.rectangle(vis, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 0, 230) if g.value else col, 3)
        elif r.kind in ("radio", "circled"):
            a = r.extra.get("anchor")
            if a is not None:
                ax, ay = int(a.x0 * S), int(a.y0 * S)
                cv2.rectangle(vis, (ax - 4, ay - 4), (int(a.line_x1 * S) + 4, int(a.y1 * S) + 4), (60, 180, 60), 2)
                label(ax, ay - 6, f"{r.key}: {g.display or g.value} [{tag}]", col)
    for keys, b in loc.spans:
        x0, y0, x1, y1 = (int(v * S) for v in b)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 0, 200), 2)
        label(x0, y0 - 4, f"diagonal writing over {len(keys)} rows", (200, 0, 200))
    items = [("field area", (60, 180, 60)), ("handwriting read", (0, 0, 230)), ("diagonal writing", (200, 0, 200)),
             ("KNOWN", COL["KNOWN"]), ("NEEDS REVIEW", COL["NEEDS_REVIEW"]), ("identifier, blacked out", (0, 0, 0))]
    leg = np.full((40, vis.shape[1], 3), 255, np.uint8)
    x = 10
    for t, c in items:
        cv2.rectangle(leg, (x, 12), (x + 22, 28), c, 3)
        cv2.putText(leg, t, (x + 28, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        x += 40 + 11 * len(t)
    cv2.imwrite(out, np.vstack([leg, vis]))
    return dict(layout=lay.name, statuses=counts, spans=len(loc.spans), identifiers=len(loc.pii))


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else f"eval_out/{src.rsplit('/', 1)[-1].rsplit('.', 1)[0]}_detections.png"
    print(draw(src, dst), "->", dst)
