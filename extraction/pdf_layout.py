"""Read the specimen registry PDF: printed labels, handwritten values, checkboxes, ticks.

The organisers' PDF was generated with ReportLab. Printed text uses Helvetica; every
handwritten value is drawn character by character in a handwriting font (Caveat, Gaegu,
NanumPen, ShadowsIntoLight...). Checkboxes are 8 pt squares and ticks are coloured
strokes. That lets us recover exact ground truth for every page, which we use to build
the page templates and to score the extractor.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pymupdf

# printed text that changes from one patient to the next: never use it as an anchor
VOLATILE_LABEL = re.compile(r"Patiente fictive|^MÈRE —")

BOX_COLOR = (0.12, 0.08, 0.1)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def fold(s: str) -> str:
    """Case/accent-insensitive form used for fuzzy comparisons."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("œ", "oe").replace("Œ", "OE")
    return re.sub(r"\s+", " ", s).strip().casefold()


@dataclass
class Label:
    text: str
    bbox: tuple[float, float, float, float]

    @property
    def yc(self):
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def xc(self):
        return (self.bbox[0] + self.bbox[2]) / 2


def chars_to_text(chars) -> str:
    """Join handwritten characters (in writing order) into a string.

    Explicit space characters are kept; a gap wider than ~a third of the glyph height is
    also treated as a space (values written in separate strokes)."""
    out, prev_x1 = "", None
    for uni, bb in chars:
        h = bb[3] - bb[1]
        if prev_x1 is not None and bb[0] - prev_x1 > 0.45 * h and not out.endswith(" ") and uni != " ":
            out += " "
        out += uni
        prev_x1 = bb[2]
    return re.sub(r"\s+", " ", out).strip()


@dataclass
class HwGroup:
    """A run of handwritten characters drawn together (one written value or word)."""

    text: str
    bbox: list[float]
    chars: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)
    order: int = 0  # position in the content stream = writing order

    @property
    def yc(self):
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class PageLayout:
    width: float
    height: float
    labels: list[Label]
    hw: list[HwGroup]
    boxes: list[tuple[float, float, float, float]]
    marks: list[tuple[float, float, float, float]]


def _is_printed_font(font: str) -> bool:
    return font.startswith("Helvetica")


def read_page(page: pymupdf.Page) -> PageLayout:
    labels: list[Label] = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                t = norm_text(s["text"])
                if t and _is_printed_font(s["font"]):
                    labels.append(Label(t, tuple(s["bbox"])))
    labels.sort(key=lambda l: (round(l.bbox[1]), l.bbox[0]))

    # handwriting: walk the content stream in drawing order and cut it into groups
    groups: list[HwGroup] = []
    cur: HwGroup | None = None
    prev = None
    for span in page.get_texttrace():
        if _is_printed_font(span["font"]):
            continue
        style = (span["font"], span["color"])  # size jitters per character
        for ch in span["chars"]:
            uni = chr(ch[0])
            bb = tuple(ch[3])
            yc = (bb[1] + bb[3]) / 2
            new = (
                cur is None
                or style != prev[0]
                or abs(yc - prev[1]) > 3.5
                or bb[0] - prev[2] > span["size"] * 0.9
                or bb[0] < prev[2] - 3
            )
            if new:
                cur = HwGroup("", list(bb), order=len(groups))
                groups.append(cur)
            cur.text += uni
            cur.chars.append((uni, bb))
            cur.bbox = [min(cur.bbox[0], bb[0]), min(cur.bbox[1], bb[1]),
                        max(cur.bbox[2], bb[2]), max(cur.bbox[3], bb[3])]
            prev = (style, yc, bb[2])
    groups = [g for g in groups if g.text.strip()]

    boxes, marks = [], []
    for d in page.get_drawings():
        r = d["rect"]
        kinds = "".join(i[0] for i in d["items"])
        col = d.get("color")
        if kinds == "llll" and r.width < 12 and r.height < 12 and col and _close(col, BOX_COLOR):
            boxes.append((r.x0, r.y0, r.x1, r.y1))
        elif col and r.width < 25 and r.height < 25 and not (kinds == "llll" and _close(col, BOX_COLOR)):
            marks.append((r.x0, r.y0, r.x1, r.y1))  # ticks/crosses, coloured or black
    return PageLayout(page.rect.width, page.rect.height, labels, groups, boxes, marks)


def _close(a, b, tol=0.05):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def find_label(labels: list[Label], anchor: str, nth: int = 0) -> Label | None:
    """Exact match on normalised text; falls back to prefix match ("Autres : ......")."""
    a = norm_text(anchor)
    exact = [l for l in labels if l.text == a and not VOLATILE_LABEL.search(l.text)]
    if exact:
        return exact[nth] if nth < len(exact) else None
    pref = [l for l in labels if l.text.startswith(a) and not VOLATILE_LABEL.search(l.text)]
    return pref[nth] if nth < len(pref) else None
