"""Page templates: where every printed label, value region and checkbox is on each page type.

A template is built once from the specimen PDF (``tools/build_templates.py``) and saved
to ``extraction/data/templates.json``; at runtime nothing else is needed. All coordinates
are PDF points (A4 = 595 x 842) on the template page.

Checked against the 10 specimen patients: the printed layout of a given page type only
moves by a global affine transform from one patient to the next (residual 0.0 pt), so a
single template plus an alignment step covers every page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .catalog import CATALOG, Field
from .pdf_layout import Label, PageLayout, VOLATILE_LABEL, find_label

TEMPLATES_PATH = Path(__file__).parent / "data" / "templates.json"

RIGHT_MARGIN = 565.0
TEXT_HALF_HEIGHT = 8.5


@dataclass
class Template:
    page_type: str
    width: float
    height: float
    labels: list[dict]  # {"text", "bbox"}
    fields: dict[str, dict]  # key -> geometry

    @classmethod
    def from_dict(cls, d):
        return cls(d["page_type"], d["width"], d["height"], d["labels"], d["fields"])


PRINTED_SCALE = 4.0  # px per point of the printed-ink masks


def printed_mask_path(page_type: str) -> Path:
    return TEMPLATES_PATH.parent / f"printed_{page_type}.png"


@lru_cache(maxsize=2)
def printed_mask(page_type: str):
    """0/1 mask of everything printed on the blank form (template coords, 4 px/pt)."""
    import cv2

    m = cv2.imread(str(printed_mask_path(page_type)), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 127).astype("uint8")


def printed_crop(page_type: str, rect, scale: float = PRINTED_SCALE):
    m = printed_mask(page_type)
    if m is None:
        return None
    x0, y0, x1, y1 = rect
    w, h = max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))
    X0, Y0 = int(round(x0 * scale)), int(round(y0 * scale))
    out = m[max(0, Y0):Y0 + h, max(0, X0):X0 + w]
    if out.shape != (h, w):  # region touches the page edge
        import numpy as np

        pad = np.zeros((h, w), "uint8")
        pad[:out.shape[0], :out.shape[1]] = out
        out = pad
    return out


@lru_cache(maxsize=1)
def load_templates() -> dict[str, Template]:
    raw = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    return {k: Template.from_dict(v) for k, v in raw.items()}


# --- building -------------------------------------------------------------------------


def _same_line(a_yc, b_yc, tol=6.0):
    return abs(a_yc - b_yc) < tol


def _text_region(layout: PageLayout, anchor: Label) -> list[float]:
    x0 = anchor.bbox[2] + 2.0  # values start >= 4 pt after the label; keeps its ":" out
    stops = [l.bbox[0] for l in layout.labels
             if _same_line(l.yc, anchor.yc) and l.bbox[0] > anchor.bbox[2] + 1]
    stops += [b[0] for b in layout.boxes
              if _same_line((b[1] + b[3]) / 2, anchor.yc) and b[0] > anchor.bbox[2] + 1]
    x1 = (min(stops) - 2.0) if stops else RIGHT_MARGIN
    return [x0, anchor.yc - TEXT_HALF_HEIGHT, x1, anchor.yc + TEXT_HALF_HEIGHT]


def _find_box(layout: PageLayout, anchor: Label, side: str):
    best, best_d = None, None
    for b in layout.boxes:
        byc = (b[1] + b[3]) / 2
        if not _same_line(byc, anchor.yc, 7.0):
            continue
        if side == "left":
            d = anchor.bbox[0] - b[2]
        else:
            d = b[0] - anchor.bbox[2]
        if -2 <= d <= (30 if side == "left" else 80) and (best_d is None or d < best_d):
            best, best_d = b, d
    return list(best) if best else None


def _table_id(key: str) -> str:
    return key.split(".", 1)[0]


def build_template(layout: PageLayout, page_type: str) -> tuple[dict, list[str]]:
    problems: list[str] = []
    fields: dict[str, dict] = {}
    cat = CATALOG[page_type]

    # row spacing per table, from the row anchors actually present on the page
    row_y: dict[str, list[float]] = {}
    for f in cat:
        if f.kind == "cell":
            r = find_label(layout.labels, *f.row)
            if r:
                row_y.setdefault(_table_id(f.key), []).append(r.yc)

    for f in cat:
        geo = _field_geometry(layout, f, row_y, problems)
        if geo is not None:
            geo.update(kind=f.kind, pii=f.pii)
            fields[f.key] = geo

    # the mother's name is printed in the header of the postpartum pages: an identifier region
    for l in layout.labels:
        if l.text.startswith("MÈRE —"):
            fields["header_name"] = {"region": [round(l.bbox[0] - 2, 2), round(l.bbox[1] - 2, 2), 400.0,
                                                round(l.bbox[3] + 2, 2)], "kind": "text", "pii": True}

    labels = [{"text": l.text, "bbox": [round(v, 2) for v in l.bbox]}
              for l in layout.labels if not VOLATILE_LABEL.search(l.text)]
    tpl = {"page_type": page_type, "width": layout.width, "height": layout.height,
           "labels": labels, "fields": fields}
    return tpl, problems


def _field_geometry(layout: PageLayout, f: Field, row_y, problems) -> dict | None:
    if f.kind == "text":
        a = find_label(layout.labels, f.anchor, f.nth)
        if not a:
            problems.append(f"{f.key}: anchor {f.anchor!r}#{f.nth} not found")
            return None
        if f.placement == "below":
            region = [a.bbox[0] - 2, a.bbox[3] + 1, RIGHT_MARGIN, a.bbox[3] + 24]
        elif f.placement == "after_box":
            box = _find_box(layout, a, "right")
            if not box:
                problems.append(f"{f.key}: no checkbox after {f.anchor!r}")
                return None
            region = [box[2] + 2, a.yc - TEXT_HALF_HEIGHT, RIGHT_MARGIN, a.yc + TEXT_HALF_HEIGHT]
        else:
            region = _text_region(layout, a)
        return {"region": [round(v, 2) for v in region]}

    if f.kind == "cell":
        r = find_label(layout.labels, *f.row)
        c = find_label(layout.labels, *f.col)
        if not r or not c:
            problems.append(f"{f.key}: row {f.row} or col {f.col} not found")
            return None
        if f.col_next:
            n = find_label(layout.labels, *f.col_next)
            x1 = n.bbox[0] - 3.5
        else:
            stops = [l.bbox[0] for l in layout.labels
                     if _same_line(l.yc, c.yc) and l.bbox[0] > c.bbox[2] + 1]
            x1 = (min(stops) - 3.5) if stops else RIGHT_MARGIN
        ys = sorted(row_y.get(_table_id(f.key), []))
        gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 5]
        half = 0.48 * min(gaps) if gaps else 9.0
        return {"region": [round(c.bbox[0] - 1.5, 2), round(r.yc - half, 2), round(x1, 2), round(r.yc + half, 2)]}

    if f.kind == "check":
        a = find_label(layout.labels, f.anchor, f.nth)
        box = _find_box(layout, a, f.side) if a else None
        if not box:
            problems.append(f"{f.key}: checkbox for {f.anchor!r}#{f.nth} ({f.side}) not found")
            return None
        return {"box": [round(v, 2) for v in box]}

    if f.kind == "radio":
        opts = {}
        for o in f.options:
            a = find_label(layout.labels, o.anchor, o.nth)
            box = _find_box(layout, a, o.side) if a else None
            if not box:
                problems.append(f"{f.key}={o.value}: checkbox for {o.anchor!r}#{o.nth} not found")
                continue
            opts[o.value] = [round(v, 2) for v in box]
        return {"options": opts} if opts else None

    raise ValueError(f.kind)
