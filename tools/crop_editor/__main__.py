"""Crop editor: trace field areas and handwriting crops on every photo, export a learning set.

    python -m tools.crop_editor ["data/Paper Registry"] [--port 8765]
    python -m tools.crop_editor --export          (write the dataset without starting the server)

Opens a local web page (http://localhost:8765) with two modes:

* Crops: for each field, the crop(s) the OCR should read. Squares, or lines used like a
  highlighter (the band around the line, any angle, adjustable thickness). The OCR re-reads
  every change at once. Saved to evaluation/crop_labels.json.
* Field areas: where each field is on the page. Squares, or the printed line the value is
  written on. Saved to evaluation/booklet_areas.json.

Every field can hold any number of shapes; fields the layout does not know can be added by
name. "Export dataset" (or --export) writes evaluation/crop_dataset.json: one example per
labelled field (area, program's area and crop, ideal crop, true text, the crop relative to
the area in line heights, the page's printed lines) for learning how to crop, and the crop
images as the OCR sees them to eval_out/crop_dataset/ (git-ignored: handwriting stays local).

Everything stays on this machine: the server only listens on localhost and identifiers are
blacked out in the images it serves.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from extraction import booklet, ocr
from extraction.align import align
from extraction.backends.local import MIN_PAGE_CONFIDENCE
from extraction.catalog import fields_for, get_field
from extraction.imageproc import CROP_SCALE, _ink, analyze_text_crop, load_image, text_crop_for_ocr, warp_region
from extraction.template import load_templates, printed_crop

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OCR_LOCK = threading.Lock()  # the OCR sessions are shared: one request at a time
CUSTOM = "custom:"  # prefix of fields added by hand (not in the layout)


def _b64png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def layout_keys(lay) -> list[tuple[str, str]]:
    """(key, kind) of every field a real-booklet layout defines, in reading order."""
    out = []
    for spec in lay.fields:
        if isinstance(spec, booklet.Table):
            for rk, _ in spec.rows:
                for ck, _ in spec.cols:
                    out.append((spec.key_fmt.format(prefix=spec.prefix, col=ck, row=rk), "cell"))
        elif isinstance(spec, booklet.CheckRow):
            out += [(k, "check") for k in spec.keys]
        else:
            kind = {"Inline": "text", "Check": "check", "Radio": "radio", "Circled": "circled",
                    "Column": "text"}.get(type(spec).__name__, "text")
            out.append((spec.key, kind))
    return out


def _label(page_type: str, key: str) -> str:
    try:
        return get_field(page_type, key).label_fr
    except KeyError:
        return key


def _is_pii(page_type: str, key: str) -> bool:
    try:
        return get_field(page_type, key).pii
    except KeyError:
        return False


# --- shapes ---------------------------------------------------------------------------------
# rect: {"shape": "rect", "box": [x0, y0, x1, y1]}
# line: {"shape": "line", "line": [x0, y0, x1, y1], "height": h}   (crops: band of height h
#        centred on the line; areas: the printed line itself, no height)


def band_corners(line, height) -> np.ndarray:
    x0, y0, x1, y1 = line
    L = max(1e-6, math.hypot(x1 - x0, y1 - y0))
    nx, ny = -(y1 - y0) / L, (x1 - x0) / L  # unit normal
    h = height / 2
    return np.array([[x0 + nx * h, y0 + ny * h], [x1 + nx * h, y1 + ny * h],
                     [x1 - nx * h, y1 - ny * h], [x0 - nx * h, y0 - ny * h]])


def shape_bbox(sh: dict) -> list[float]:
    if sh["shape"] == "rect":
        return list(sh["box"])
    pts = band_corners(sh["line"], sh["height"]) if sh.get("height") else np.array(sh["line"], float).reshape(2, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def union_bbox(shapes: list[dict]) -> list[float] | None:
    if not shapes:
        return None
    bs = [shape_bbox(s) for s in shapes]
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class Doc:
    """One photo, analysed once: its working frame, field areas and automatic crops."""

    def __init__(self, path: Path):
        self.name = path.name
        img = load_image(path)
        with OCR_LOCK:
            lines = ocr.read_page(img)
            al = align(lines, img)
            if al is not None and al.confidence >= MIN_PAGE_CONFIDENCE:
                self._specimen(img, al, lines)
            else:
                lay, _ = booklet.classify(lines)
                if lay is None:
                    raise ValueError("page not recognised")
                self._booklet(img, lines, lay)
            self._read_auto()
            ocr.trim_memory()

    # -- the two kinds of page -------------------------------------------------------------

    def _booklet(self, img, lines, lay):
        page = booklet.analyse(img, lines)
        loc = booklet.locate(page, lay)
        inks = booklet.word_crops(page, loc)
        self.kind, self.layout, self.page_type = "booklet", lay.name, lay.page_type
        self.frame, self.page, self.th = page.img, page, float(page.th)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), page.angle, 1.0)
        self.to_original = cv2.invertAffineTransform(M)  # frame -> original photo
        self.deskew_deg = page.angle
        self.pii = [list(map(int, b)) for b in loc.pii]
        self.hlines = [[round(s.lo), round(s.at(s.lo)), round(s.hi), round(s.at(s.hi))] for s in page.hl]
        self.vlines = [[round(s.at(s.lo)), round(s.lo), round(s.at(s.hi)), round(s.hi)] for s in page.vl]
        self.fields = []
        for r in loc.regions:
            if r.key not in inks or _is_pii(lay.page_type, r.key):
                continue
            ink = inks[r.key]
            self.fields.append(dict(key=r.key, label=_label(lay.page_type, r.key), kind=r.kind,
                                    region=list(map(int, r.rect)),
                                    auto=list(map(int, ink.box)) if ink.present and ink.box else None,
                                    auto_dash=bool(ink.is_dash), _crop=ink.clean if ink.present else None))
        found = {r.key: r for r in loc.regions}
        self.area_fields = []
        for key, kind in layout_keys(lay):
            if _is_pii(lay.page_type, key):
                continue
            r = found.get(key)
            self.area_fields.append(dict(key=key, label=_label(lay.page_type, key), kind=r.kind if r else kind,
                                         region=list(map(int, r.rect)) if r and r.rect[2] > r.rect[0] else None))

    def _specimen(self, img, al, lines):
        tpl = load_templates()[al.page_type]
        H = al.H
        self.kind, self.layout, self.page_type = "specimen", al.page_type, al.page_type
        self.frame, self.page = img, None
        self.to_original = np.array([[1, 0, 0], [0, 1, 0]], float)
        self.deskew_deg = 0.0
        hs = [l.quad[:, 1].max() - l.quad[:, 1].min() for l in lines if l.score >= 0.9 and len(l.text) >= 4]
        self.th = float(np.median(hs)) if hs else 24.0
        self.hlines, self.vlines = [], []

        def to_img(rect):
            x0, y0, x1, y1 = rect
            q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64).reshape(-1, 1, 2)
            q = cv2.perspectiveTransform(q, H).reshape(-1, 2)
            return [int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max())]

        self.pii = [to_img(g["region"]) for g in tpl.fields.values() if g.get("pii")]
        self.fields, self.area_fields = [], []
        for f in fields_for(al.page_type):
            geo = tpl.fields.get(f.key)
            if geo is None or f.pii:
                continue
            if f.kind == "check" and geo.get("box"):
                self.area_fields.append(dict(key=f.key, label=f.label_fr, kind="check", region=to_img(geo["box"])))
            if f.kind not in ("text", "cell"):
                continue
            region = to_img(geo["region"])
            self.area_fields.append(dict(key=f.key, label=f.label_fr, kind=f.kind, region=region))
            ink = analyze_text_crop(warp_region(img, H, geo["region"]), printed=printed_crop(al.page_type, geo["region"]))
            auto, crop = None, None
            if ink.present:
                x0, y0 = geo["region"][:2]
                bx0, by0, bx1, by1 = ink.bbox
                pad = 0 if ink.is_dash else 2
                auto = to_img((x0 + bx0 / CROP_SCALE - pad, y0 + by0 / CROP_SCALE - pad,
                               x0 + bx1 / CROP_SCALE + pad, y0 + by1 / CROP_SCALE + pad))
                crop = text_crop_for_ocr(ink)
            self.fields.append(dict(key=f.key, label=f.label_fr, kind=f.kind, region=region,
                                    auto=auto, auto_dash=bool(ink.is_dash), _crop=crop))

    def _read_auto(self):
        need = [f for f in self.fields if f["_crop"] is not None and not f["auto_dash"]]
        reads = ocr.read_crops([f["_crop"] for f in need], model="small")
        for f, (t, s) in zip(need, reads):
            f["auto_read"] = [t, round(s, 2)]
        for f in self.fields:
            f.setdefault("auto_read", ["—", 1.0] if f["auto_dash"] else None)

    # -- serving ---------------------------------------------------------------------------

    def image_jpeg(self) -> bytes:
        vis = self.frame.copy()
        for x0, y0, x1, y1 in self.pii:  # identifiers are never shown
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), -1)
        return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()

    def summary(self) -> dict:
        h, w = self.frame.shape[:2]
        return dict(name=self.name, kind=self.kind, layout=self.layout, page_type=self.page_type,
                    width=w, height=h, deskew_deg=round(self.deskew_deg, 3), line_height=round(self.th, 1),
                    fields=[{k: v for k, v in f.items() if not k.startswith("_")} for f in self.fields],
                    area_fields=self.area_fields)

    def info(self, key: str) -> dict:
        """label, kind, program's area, automatic crop of a field (custom fields: none)."""
        f = next((f for f in self.fields if f["key"] == key), None)
        a = next((f for f in self.area_fields if f["key"] == key), None)
        return dict(label=(f or a or {}).get("label"), kind=(f or a or {}).get("kind"),
                    program_area=(a or f or {}).get("region"), auto=f and f.get("auto"),
                    auto_read=f and f.get("auto_read"), auto_dash=bool(f and f.get("auto_dash")),
                    auto_crop=f and f.get("_crop"))

    def printed(self, traced_lines=()) -> tuple[np.ndarray, np.ndarray]:
        """(lines, text): masks of the printed form on the page frame, to remove from crops.

        lines = printed lines found by the program + any long straight ink run it missed + the field
        lines you traced (Field areas mode); text = printed labels/slashes: on a page written in blue
        pen, every ink piece that is not blue."""
        if getattr(self, "_printed", None) is None:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            ink = _ink(gray)
            pg = self.page if self.page is not None else SimpleNamespace(img=self.frame, ink=ink)
            lines = booklet._line_mask(self.page).copy() if self.page is not None else np.zeros_like(ink)
            L = max(12, int(1.5 * self.th))
            lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))  # missed underlines
            lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((int(2.5 * L), 1), np.uint8))  # column borders
            lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
            rel, blue_pen = booklet._pen_colour(pg)
            text = np.zeros_like(ink)
            if blue_pen:
                n, lab = cv2.connectedComponents(ink & (1 - lines), connectivity=8)
                cnt = np.maximum(np.bincount(lab.ravel(), minlength=n), 1)
                means = np.bincount(lab.ravel(), weights=rel.ravel(), minlength=n) / cnt
                # pale pieces of a pen stroke (a thin slash) are barely blue: keep neutral pieces that
                # touch clearly blue handwriting; print is warm, or neutral and standing apart
                blue_ids = np.nonzero(means < -6)[0]
                blue_ids = blue_ids[blue_ids > 0]  # label 0 is the paper
                pen = cv2.dilate(np.isin(lab, blue_ids).astype(np.uint8), np.ones((7, 7), np.uint8))
                near_pen = np.bincount(lab.ravel(), weights=pen.ravel().astype(float), minlength=n) > 0
                printed = np.nonzero((means > 0) | ((means > -3) & ~near_pen))[0]
                printed = printed[printed > 0]
                if len(printed):
                    text = cv2.dilate(np.isin(lab, printed).astype(np.uint8), np.ones((5, 5), np.uint8))
            self._printed = (lines, text, bool(blue_pen))
        lines, text, _ = self._printed
        if traced_lines:
            lines = lines.copy()
            for l in traced_lines:
                cv2.line(lines, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), 1, 5)
        return lines, text

    def _clean(self, crop: np.ndarray, lines: np.ndarray | None, text: np.ndarray | None) -> np.ndarray:
        """Erase the printed form from a crop: printed text painted with the paper colour, printed
        lines inpainted (so handwriting crossing them is rebuilt), then the paper normalised."""
        lines = lines > 0 if lines is not None else np.zeros(crop.shape[:2], bool)
        text = (text > 0) & ~lines if text is not None else np.zeros(crop.shape[:2], bool)
        if text.any():
            paper = ~lines & ~text & (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 0)
            ink = _ink(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) > 0
            bg_px = crop[paper & ~ink]
            crop = crop.copy()
            crop[text] = np.median(bg_px, axis=0) if len(bg_px) else 255
        if lines.any():
            crop = cv2.inpaint(crop, lines.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        if self.page is not None:
            return booklet._for_recogniser(cv2.copyMakeBorder(crop, 4, 4, 6, 6, cv2.BORDER_REPLICATE))
        return cv2.copyMakeBorder(crop, 6, 6, 10, 10, cv2.BORDER_REPLICATE)

    def crop_shape(self, sh: dict, clean: bool = True, traced_lines=()) -> np.ndarray:
        """What the recogniser sees for a hand-drawn shape (clean: the printed form erased first).
        A line is a highlighter stroke: the band of `height` px around it, turned horizontal."""
        h, w = self.frame.shape[:2]
        if clean:
            lm, tm = self.printed(traced_lines)
        else:
            lm = booklet._line_mask(self.page) if self.page is not None else None
            tm = None
        if sh["shape"] == "rect":
            x0, y0, x1, y1 = (int(round(v)) for v in sh["box"])
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                raise ValueError("box too small")
            cut = lambda m: m[y0:y1, x0:x1] if m is not None else None
            return self._clean(self.frame[y0:y1, x0:x1].copy(), cut(lm), cut(tm))
        x0, y0, x1, y1 = sh["line"]
        L = math.hypot(x1 - x0, y1 - y0)
        hb = float(sh.get("height") or 1.4 * self.th)
        if L < 4 or hb < 3:
            raise ValueError("line too short")
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        nx, ny = -uy, ux
        M = np.array([[ux, nx, x0 - hb / 2 * nx], [uy, ny, y0 - hb / 2 * ny]], np.float64)  # strip -> frame
        size = (int(round(L)), int(round(hb)))
        crop = cv2.warpAffine(self.frame, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        warp = lambda m: (cv2.warpAffine(m, M, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                          if m is not None else None)
        return self._clean(crop, warp(lm), warp(tm))

    def read(self, key: str, shapes: list[dict], clean: bool = True, traced_lines=()) -> dict:
        crops = []
        for sh in shapes:
            try:
                crops.append(self.crop_shape(sh, clean, traced_lines))
            except ValueError:
                crops.append(None)
        good = [c for c in crops if c is not None]
        with OCR_LOCK:
            small = ocr.read_crops(good, model="small") if good else []
            medium = ocr.read_crops(good, model="medium") if good and ocr.medium_available() else []
        out, i = [], 0
        for c in crops:
            if c is None:
                out.append(None)
                continue
            out.append(dict(crop=_b64png(c), small=[small[i][0], round(small[i][1], 2)],
                            medium=[medium[i][0], round(medium[i][1], 2)] if medium else None))
            i += 1
        auto = self.info(key)["auto_crop"]
        return dict(crops=out, auto_crop=_b64png(auto) if auto is not None else None)

    def to_orig(self, pts) -> list:
        q = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(pts))] @ self.to_original.T
        return [[round(float(x), 1), round(float(y), 1)] for x, y in q]

    def normalise(self, sh: dict, with_height: bool) -> dict:
        """Validate a shape from the page and add its coordinates on the original photo."""
        h, w = self.frame.shape[:2]
        cx = lambda v: max(0, min(w, int(round(float(v)))))
        cy = lambda v: max(0, min(h, int(round(float(v)))))
        if sh.get("shape") == "rect":
            x0, y0, x1, y1 = sh["box"]
            b = [cx(min(x0, x1)), cy(min(y0, y1)), cx(max(x0, x1)), cy(max(y0, y1))]
            if b[2] - b[0] < 3 or b[3] - b[1] < 3:
                raise ValueError("square too small")
            return dict(shape="rect", box=b, box_original=self.to_orig([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]))
        if sh.get("shape") == "line":
            x0, y0, x1, y1 = sh["line"]
            l = [cx(x0), cy(y0), cx(x1), cy(y1)]
            if l[0] > l[2]:
                l = [l[2], l[3], l[0], l[1]]  # left end first
            if math.hypot(l[2] - l[0], l[3] - l[1]) < 6:
                raise ValueError("line too short")
            out = dict(shape="line", line=l, line_original=self.to_orig([[l[0], l[1]], [l[2], l[3]]]))
            if with_height:
                out["height"] = max(4, min(600, int(round(float(sh.get("height") or 1.4 * self.th)))))
                out["band_original"] = self.to_orig(band_corners(l, out["height"]))
            return out
        raise ValueError("shape must be rect or line")


class Store:
    """{"images": {photo: {"fields": {key: {"shapes": [...], ...}}}}}, one file per mode."""

    def __init__(self, path: Path, mode: str, note: str):
        self.path, self.mode = path, mode
        self.lock = threading.Lock()
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"images": {}}
        self.data["note"] = note
        for img in self.data["images"].values():  # older files: one box/line per field
            for e in img.get("fields", {}).values():
                if "shapes" not in e:
                    shapes = []
                    if e.get("shape") == "line" and e.get("line"):
                        shapes.append(dict(shape="line", line=e["line"]))
                    elif e.get("box"):
                        shapes.append(dict(shape="rect", box=e["box"]))
                    for k in ("shape", "box", "line", "box_original", "line_original", "auto_box", "auto_region"):
                        e.pop(k, None)
                    e["shapes"] = shapes

    def get(self, name: str) -> dict:
        return self.data["images"].get(name, {}).get("fields", {})

    def put(self, doc: Doc, key: str, entry: dict | None):
        if not key or len(key) > 200:
            raise ValueError("bad field key")
        with self.lock:
            img = self.data["images"].setdefault(doc.name, {"fields": {}})
            img.update(kind=doc.kind, frame="deskewed" if doc.kind == "booklet" else "original",
                       deskew_deg=round(doc.deskew_deg, 3), layout=doc.layout, page_type=doc.page_type,
                       size=[int(doc.frame.shape[1]), int(doc.frame.shape[0])], line_height=round(doc.th, 1))
            if entry is None:
                img["fields"].pop(key, None)
            else:
                clean = dict(shapes=[doc.normalise(s, with_height=self.mode == "crops") for s in entry.get("shapes", [])])
                if self.mode == "areas" and entry.get("none") and not clean["shapes"]:
                    clean["none"] = True  # the field has no area on this page (deleted by hand)
                if self.mode == "crops":
                    clean["empty"] = bool(entry.get("empty"))
                    if entry.get("text"):
                        clean["text"] = str(entry["text"])[:200]
                if key.startswith(CUSTOM):
                    clean["label"] = str(entry.get("label") or key[len(CUSTOM):])[:120]
                clean["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                img["fields"][key] = clean
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)


CROPS_NOTE = ("Hand-made handwriting crops (python -m tools.crop_editor, mode 'Crops'). Per field: shapes = the crops "
              "the OCR should read: rect box [x0, y0, x1, y1], or line [x0, y0, x1, y1] with height = the band of that "
              "height centred on the line (any angle). Page-frame pixels (real booklet: the photo deskewed by deskew_deg "
              "around its centre; specimen: the photo itself); *_original = the same points on the original photo. "
              "empty = no handwriting; text = what is written.")
AREAS_NOTE = ("Hand-traced field areas (python -m tools.crop_editor, mode 'Field areas'). Per field: shapes = where the "
              "field is: rect box [x0, y0, x1, y1], or line [x0, y0, x1, y1] = the printed line the value is written on; "
              "none = the field has no area on this page (deleted by hand). "
              "Page-frame pixels as in crop_labels.json; *_original = the same points on the original photo.")


# --- export -----------------------------------------------------------------------------------


def traced_lines(areas: "Store", name: str) -> list:
    """Every field line traced in Field areas mode on this photo (printed lines to erase from crops)."""
    return [sh["line"] for e in areas.get(name).values() for sh in e.get("shapes", []) if sh.get("shape") == "line"]


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120]


def export_dataset(crops: Store, areas: Store, get_doc, out_json: Path, img_dir: Path) -> dict:
    """One example per field that has any hand label: what a learner needs to predict the ideal crop."""
    photos, n_ex, rel_rows, ious = [], 0, [], []
    names = sorted(set(crops.data["images"]) | set(areas.data["images"]))
    for name in names:
        c_fields, a_fields = crops.get(name), areas.get(name)
        keys = [k for k in dict.fromkeys(list(a_fields) + list(c_fields))]
        if not keys:
            continue
        doc = get_doc(name)
        th = doc.th
        pdir = img_dir / _safe(Path(name).stem)
        pdir.mkdir(parents=True, exist_ok=True)
        examples = []
        for key in keys:
            info = doc.info(key)
            ce, ae = c_fields.get(key, {}), a_fields.get(key, {})
            traced = ae.get("shapes") or []
            removed = bool(ae.get("none")) and not traced
            area_shapes = traced or ([dict(shape="rect", box=info["program_area"])] if info["program_area"] and not removed else [])
            area_bbox = union_bbox(area_shapes)
            ideal = ce.get("shapes") or []
            ideal_bbox = union_bbox(ideal)
            ex = dict(
                key=key, label=ce.get("label") or ae.get("label") or info["label"] or key,
                kind=info["kind"] or "custom", custom=key.startswith(CUSTOM),
                field_area=dict(source="traced" if traced else "deleted" if removed else ("program" if area_shapes else None),
                                shapes=area_shapes, bbox=area_bbox),
                program_area=info["program_area"],
                auto_crop=dict(box=info["auto"], read=info["auto_read"], dash=info["auto_dash"]) if info["auto"] else None,
                ideal_crop=dict(shapes=ideal, bbox=ideal_bbox, empty=bool(ce.get("empty"))) if ce else None,
                text=ce.get("text"),
            )
            # the ideal crop relative to the field area, in printed-line heights (scale-free targets)
            if area_bbox and ideal_bbox:
                rel = dict(left=(ideal_bbox[0] - area_bbox[0]) / th, top=(ideal_bbox[1] - area_bbox[1]) / th,
                           right=(ideal_bbox[2] - area_bbox[2]) / th, bottom=(ideal_bbox[3] - area_bbox[3]) / th,
                           width=(ideal_bbox[2] - ideal_bbox[0]) / th, height=(ideal_bbox[3] - ideal_bbox[1]) / th)
                ex["ideal_vs_area_in_line_heights"] = {k: round(v, 3) for k, v in rel.items()}
                rel_rows.append(rel)
            if info["auto"] and ideal_bbox:
                ex["iou_auto_vs_ideal"] = round(iou(info["auto"], ideal_bbox), 3)
                ious.append(ex["iou_auto_vs_ideal"])
            lines = [s for s in ideal if s["shape"] == "line"]
            if lines:
                ex["ideal_lines"] = [dict(angle_deg=round(math.degrees(math.atan2(s["line"][3] - s["line"][1],
                                                                                   s["line"][2] - s["line"][0])), 2),
                                          height_in_line_heights=round(s["height"] / th, 3)) for s in lines]
            # crop images as the recogniser sees them
            files = []
            for i, sh in enumerate(ideal):
                try:
                    p = pdir / f"{_safe(key)}__ideal{i}.png"
                    cv2.imwrite(str(p), doc.crop_shape(sh, True, traced_lines(areas, name)))
                    files.append(str(p.relative_to(REPO)))
                except ValueError:
                    pass
            if files:
                ex["ideal_crop_images"] = files
            if info["auto_crop"] is not None:
                p = pdir / f"{_safe(key)}__auto.png"
                cv2.imwrite(str(p), info["auto_crop"])
                ex["auto_crop_image"] = str(p.relative_to(REPO))
            examples.append(ex)
        n_ex += len(examples)
        photos.append(dict(
            photo=name, kind=doc.kind, layout=doc.layout, page_type=doc.page_type,
            frame="deskewed" if doc.kind == "booklet" else "original", deskew_deg=round(doc.deskew_deg, 3),
            size=[int(doc.frame.shape[1]), int(doc.frame.shape[0])], line_height=round(th, 1),
            frame_to_original=[[round(float(v), 6) for v in row] for row in doc.to_original],
            printed_lines=dict(horizontal=doc.hlines, vertical=doc.vlines),
            examples=examples))
    summary = dict(photos=len(photos), examples=n_ex,
                   with_ideal_crop=sum(1 for p in photos for e in p["examples"] if e["ideal_crop"] and e["ideal_crop"]["shapes"]),
                   marked_empty=sum(1 for p in photos for e in p["examples"] if e["ideal_crop"] and e["ideal_crop"]["empty"]),
                   with_traced_area=sum(1 for p in photos for e in p["examples"] if e["field_area"]["source"] == "traced"),
                   with_text=sum(1 for p in photos for e in p["examples"] if e["text"]))
    if rel_rows:
        summary["mean_ideal_vs_area_in_line_heights"] = {
            k: round(float(np.mean([r[k] for r in rel_rows])), 3) for k in rel_rows[0]}
    if ious:
        summary["mean_iou_auto_vs_ideal"] = round(float(np.mean(ious)), 3)
    data = dict(
        version=1,
        note=("Learning set for cropping handwriting (python -m tools.crop_editor --export). One example per field "
              "with a hand label. Inputs: the photo (frame = deskewed page for real booklets; frame_to_original maps "
              "frame pixels to the original photo), line_height (printed text height in px, the natural unit), the "
              "field area (traced by hand, else found by the program), the printed lines of the page, and the "
              "program's own crop. Targets: ideal_crop.shapes (rect box, or line + height = a band around the line), "
              "ideal_crop.empty, text. ideal_vs_area_in_line_heights = edges of the ideal crop minus edges of the "
              "area, divided by line_height (scale-free). Crop images (as the OCR sees them) are in eval_out/, kept "
              "out of git; the printed form (lines, traced field lines, printed labels on blue-pen pages) is erased "
              "from them, as in the editor. Identifier fields are never included."),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"), summary=summary, photos=photos)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return dict(summary, file=str(out_json.relative_to(REPO)) if out_json.is_relative_to(REPO) else str(out_json),
                images=str(img_dir.relative_to(REPO)) if img_dir.is_relative_to(REPO) else str(img_dir))


# --- server -----------------------------------------------------------------------------------


class App:
    def __init__(self, folder: Path, crops: Store, areas: Store, out_json: Path, img_dir: Path):
        self.folder, self.crops, self.areas, self.out_json, self.img_dir = folder, crops, areas, out_json, img_dir
        self.docs: OrderedDict[str, Doc] = OrderedDict()
        self.lock = threading.Lock()
        self.files = sorted((p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                            key=lambda p: (not p.name.startswith("1-"), p.name))

    def doc(self, name: str) -> Doc:
        with self.lock:
            if name in self.docs:
                self.docs.move_to_end(name)
                return self.docs[name]
        path = self.folder / name
        if path.parent != self.folder or not path.exists():
            raise FileNotFoundError(name)
        d = Doc(path)
        with self.lock:
            self.docs[name] = d
            while len(self.docs) > 6:
                self.docs.popitem(last=False)
        return d

    def export(self) -> dict:
        return export_dataset(self.crops, self.areas, self.doc, self.out_json, self.img_dir)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _q(self):
            return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

        def do_GET(self):
            route = urlparse(self.path).path
            try:
                if route == "/":
                    return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
                if route == "/api/images":
                    return self._send(200, [dict(name=p.name, edited=len(app.crops.get(p.name)),
                                                 traced=len(app.areas.get(p.name))) for p in app.files])
                if route == "/api/page":
                    name = self._q()["name"]
                    s = app.doc(name).summary()
                    s["labels"], s["areas"] = app.crops.get(name), app.areas.get(name)
                    return self._send(200, s)
                if route == "/api/dataset":  # the exported learning set, as a download
                    if not app.out_json.exists():
                        app.export()
                    body = app.out_json.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="{app.out_json.name}"')
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if route == "/api/image":
                    return self._send(200, app.doc(self._q()["name"]).image_jpeg(), "image/jpeg")
                return self._send(404, {"error": "not found"})
            except FileNotFoundError as e:
                return self._send(404, {"error": f"no such photo: {e}"})
            except ValueError as e:
                return self._send(422, {"error": str(e)})

        def do_POST(self):
            route = urlparse(self.path).path
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if route == "/api/export":
                    return self._send(200, app.export())
                d = app.doc(body["name"])
                if route == "/api/read":
                    return self._send(200, d.read(body["key"], body.get("shapes") or [], bool(body.get("clean", True)),
                                                  traced_lines(app.areas, d.name)))
                if route == "/api/save":
                    app.crops.put(d, body["key"], body.get("entry"))
                    return self._send(200, {"ok": True, "labels": app.crops.get(d.name)})
                if route == "/api/save_area":
                    app.areas.put(d, body["key"], body.get("entry"))
                    return self._send(200, {"ok": True, "areas": app.areas.get(d.name)})
                return self._send(404, {"error": "not found"})
            except FileNotFoundError as e:
                return self._send(404, {"error": f"no such photo: {e}"})
            except (ValueError, KeyError, TypeError) as e:
                return self._send(422, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Trace field areas and crops; export a learning set.")
    ap.add_argument("folder", nargs="?", default=str(REPO / "data" / "Paper Registry"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--labels", default=str(REPO / "evaluation" / "crop_labels.json"))
    ap.add_argument("--areas", default=str(REPO / "evaluation" / "booklet_areas.json"))
    ap.add_argument("--out", default=str(REPO / "evaluation" / "crop_dataset.json"))
    ap.add_argument("--images", default=str(REPO / "eval_out" / "crop_dataset"))
    ap.add_argument("--export", action="store_true", help="write the dataset and exit (no server)")
    a = ap.parse_args()
    app = App(Path(a.folder).resolve(), Store(Path(a.labels), "crops", CROPS_NOTE),
              Store(Path(a.areas), "areas", AREAS_NOTE), Path(a.out).resolve(), Path(a.images).resolve())
    if a.export:
        print(json.dumps(app.export(), indent=1))
        return
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(app))
    print(f"crop editor: http://localhost:{a.port}  ({app.folder})\n  crops -> {a.labels}\n  areas -> {a.areas}"
          f"\n  export -> {a.out} (+ images in {a.images})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
