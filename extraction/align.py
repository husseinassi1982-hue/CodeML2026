"""Recognise which registry page a photo shows and align it with that page's template.

Printed labels are the anchors: OCR lines that match template labels give point
correspondences (template points <-> image pixels) for a RANSAC homography, which absorbs
shift, rotation, scale and the perspective of a hand-held phone photo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from rapidfuzz import fuzz

from .catalog import PAGE_TYPES
from .geometry import apply
from .imageproc import CROP_SCALE, ink_bbox, warp_region
from .ocr import OcrLine
from .pdf_layout import fold
from .template import Template, load_templates

MIN_LABEL_LEN = 4  # shorter labels ("TA", "DR", "1") are too ambiguous to match on their own
MATCH = 80.0  # rapidfuzz score needed to call an OCR line a label


@dataclass
class Alignment:
    page_type: str
    confidence: float  # how sure we are about page type and geometry together
    H: np.ndarray  # 3x3, template points -> image pixels
    n_inliers: int
    error_pt: float  # median reprojection error, in points
    readability: float = 1.0  # how well the page's printed labels were read (0..1)
    scores: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _label_index():
    """Per template: usable labels with their idf weight across page types."""
    tpls = load_templates()
    df: dict[str, int] = {}
    for t in tpls.values():
        for txt in {fold(l["text"]) for l in t.labels}:
            df[txt] = df.get(txt, 0) + 1
    idx = {}
    for pt, t in tpls.items():
        counts: dict[str, int] = {}
        for l in t.labels:
            counts[fold(l["text"])] = counts.get(fold(l["text"]), 0) + 1
        labs = []
        for l in t.labels:
            f = fold(l["text"])
            if len(f) >= MIN_LABEL_LEN:
                labs.append({"fold": f, "bbox": l["bbox"], "idf": math.log(1 + len(tpls) / df[f]),
                             "unique": counts[f] == 1})
        idx[pt] = labs
    return idx


def _sim(line_fold: str, label_fold: str) -> float:
    """Similarity allowing the OCR line to continue after the label ("age : 31")."""
    full = fuzz.ratio(line_fold, label_fold)
    if len(line_fold) > len(label_fold) + 2:
        pref = fuzz.ratio(line_fold[:len(label_fold)], label_fold) * 0.98
        return max(full, pref)
    return full


def _classify(lines: list[OcrLine]) -> tuple[dict[str, float], dict[str, list]]:
    idx = _label_index()
    folded = [fold(l.text) for l in lines]
    scores, matches, reads = {}, {}, {}
    for pt, labs in idx.items():
        tot, got, m, rd = 0.0, 0.0, [], []
        for lab in labs:
            tot += lab["idf"]
            best, bi = 0.0, -1
            for i, lf in enumerate(folded):
                s = _sim(lf, lab["fold"])
                if s > best:
                    best, bi = s, i
            rd.append(best)
            if best >= MATCH:
                got += lab["idf"] * best / 100
                m.append((lab, bi, best))
        scores[pt] = got / tot if tot else 0.0
        matches[pt] = m
        reads[pt] = float(np.mean(rd)) / 100 if rd else 0.0
    # early vs late postpartum pages differ only by a word in the title
    words = [w for w in " ".join(folded).replace("-", " ").split() if len(w) >= 5]
    early = max((fuzz.ratio("precoce", w) for w in words), default=0)
    late = max((fuzz.ratio("tardif", w) for w in words), default=0)
    for kind in ("mother", "newborn"):
        e, l_ = f"postpartum_early_{kind}", f"postpartum_late_{kind}"
        if early >= 85 > late:
            scores[l_] *= 0.5
        elif late >= 85 > early:
            scores[e] *= 0.5
    return scores, matches, reads


def align(lines: list[OcrLine], img: np.ndarray) -> Alignment | None:
    if not lines:
        return None
    scores, matches, reads = _classify(lines)
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    if scores[best] < 0.15:
        return None
    margin_score = scores[best] - scores[ranked[1]]
    tpl: Template = load_templates()[best]

    px_per_pt = max(img.shape[:2]) / max(tpl.width, tpl.height)

    # 1) coarse affine fit from OCR lines that match labels printed once on the page
    src, dst = [], []
    for lab, li, s in matches[best]:
        if not lab["unique"] or s < 85:
            continue
        line = lines[li]
        b = lab["bbox"]
        src.append((b[0], (b[1] + b[3]) / 2))
        dst.append(line.left_mid)
        if abs(len(fold(line.text)) - len(lab["fold"])) <= max(2, 0.15 * len(lab["fold"])):
            src.append((b[2], (b[1] + b[3]) / 2))
            dst.append(line.right_mid)
    if len(src) < 3:
        return None
    src, dst = np.float32(src), np.float32(dst)
    est = cv2.estimateAffine2D if len(src) >= 6 else cv2.estimateAffinePartial2D
    A, _ = est(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5 * px_per_pt)
    if A is None:
        return None
    H = np.vstack([A, [0, 0, 1]])

    # 2) refine on the measured ink of every printed label (sub-point accuracy)
    n_inl, err_pt = 0, 99.0
    for margin in (6.0, 3.0, 2.0):
        refined = _refine(img, tpl, H, px_per_pt, margin)
        if refined is None:
            break
        H, n_inl, err_pt = refined
    if n_inl == 0:
        return None

    warnings = []
    if margin_score < 0.08:
        warnings.append(f"page type uncertain: {best} vs {ranked[1]}")
    page_conf = min(1.0, scores[best] / 0.6) * min(1.0, 0.5 + margin_score * 5)
    geo_conf = min(1.0, n_inl / 20) * math.exp(-max(0.0, err_pt - 0.5) / 2)
    return Alignment(best, round(page_conf * geo_conf, 3), H, n_inl, round(err_pt, 2), round(reads[best], 3),
                     {k: round(v, 3) for k, v in scores.items()}, warnings)


def _refine(img, tpl: Template, H: np.ndarray, px_per_pt: float, margin: float):
    """Re-measure every label's ink under the current transform and refit."""
    src, dst = [], []
    for lab in tpl.labels:
        ink = lab.get("ink")
        if not ink:
            continue
        w, h = ink[2] - ink[0], ink[3] - ink[1]
        if w < 2.5 or h < 2.5:
            continue
        rect = (ink[0] - margin, ink[1] - margin, ink[2] + margin, ink[3] + margin)
        bb = ink_bbox(warp_region(img, H, rect, CROP_SCALE))
        if bb is None:
            continue
        mx0, my0 = rect[0] + bb[0] / CROP_SCALE, rect[1] + bb[1] / CROP_SCALE
        mx1, my1 = rect[0] + bb[2] / CROP_SCALE, rect[1] + bb[3] / CROP_SCALE
        # it must look like the same label (size), not neighbouring handwriting or noise
        if abs((mx1 - mx0) - w) > max(1.5, 0.12 * w) or abs((my1 - my0) - h) > max(1.2, 0.25 * h):
            continue
        ty, my = (ink[1] + ink[3]) / 2, (my0 + my1) / 2
        src += [(ink[0], ty), (ink[2], ty)]
        dst += list(apply(H, [(mx0, my), (mx1, my)]))
    if len(src) < 6:
        return None
    src, dst = np.float32(src), np.float32(dst)
    thr = 1.5 * px_per_pt
    candidates = []  # (inliers, -error, H, error)

    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr)
    if A is not None:
        inl = inl.ravel().astype(bool)
        Ha = np.vstack([A, [0, 0, 1]])
        e = float(np.median(np.linalg.norm(apply(Ha, src[inl]) - dst[inl], axis=1)))
        candidates.append((int(inl.sum()), -e, Ha, e))

    # perspective (hand-held photo): kept if it explains more labels, the labels cover enough
    # of the page to constrain it, and the implied tilt is physically plausible
    if len(src) >= 12:
        Hp, inl_p = cv2.findHomography(src, dst, cv2.RANSAC, thr)
        if Hp is not None and abs(Hp[2, 2]) > 1e-9:
            inl_p = inl_p.ravel().astype(bool)
            Hn = Hp / Hp[2, 2]
            spread = np.ptp(src[inl_p], axis=0) if inl_p.any() else np.zeros(2)
            tilt = abs(Hn[2, 0]) * tpl.width + abs(Hn[2, 1]) * tpl.height
            if tilt < 0.3 and spread[0] > 0.3 * tpl.width and spread[1] > 0.3 * tpl.height:
                e = float(np.median(np.linalg.norm(apply(Hn, src[inl_p]) - dst[inl_p], axis=1)))
                candidates.append((int(inl_p.sum()), -e, Hn, e))

    if not candidates:
        return None
    n, _, best_H, best_err = max(candidates, key=lambda c: (c[0], c[1]))
    return best_H, n // 2, best_err / px_per_pt


def page_label(page_type: str) -> str:
    return PAGE_TYPES.get(page_type, "Page non reconnue")
