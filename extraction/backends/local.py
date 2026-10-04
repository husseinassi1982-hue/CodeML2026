"""Local backend: template alignment + CPU OCR. No network, no API key, nothing leaves the machine."""

from __future__ import annotations

import time

import numpy as np

from .. import booklet, ocr, validate
from ..align import align
from ..catalog import fields_for
from ..imageproc import analyze_text_crop, checkbox_score, image_quality, text_crop_for_ocr, warp_region
from ..normalize import parse
from ..schema import PageExtraction, Status
from ..status import decide_check, decide_radio, decide_text, known_threshold
from ..template import load_templates, printed_crop
from .common import field_result, finish

VERSION = "local-ocr-0.1"
MIN_PAGE_CONFIDENCE = 0.2  # genuine pages, even hard simulated photos, score >= 0.35


FALLBACK_MAX_SCORE = 0.75  # below every KNOWN bar: a fallback value is always confirmed by the midwife


def _plausible_fallback(need, crops, reads) -> dict:
    """Fields whose free OCR reading is not a plausible value (unparseable, out of range, not in the
    field's vocabulary): decode the field's most likely plausible value from the recogniser's own
    character probabilities (extraction/constrained.py). Pre-filled for review, never KNOWN."""
    from .. import constrained

    todo = []
    for f, _ in need:
        text, _score = reads.get(f.key, (None, 0.0))
        if not text or not constrained.candidates(f.key, f.dtype, f.range):
            continue
        p = parse(text, f)
        if p.ok and not p.issues:  # a reading that parses is kept, even in an unusual format
            continue
        todo.append(f)
    if not todo:
        return {}
    probs, charset = ocr.rec_probs([crops[f.key] for f in todo], model="small")
    out = {}
    for f, pr in zip(todo, probs):
        r = constrained.read(pr, charset, f.key, f.dtype, f.range)
        if r is None or r.fit < -12:
            continue
        text = constrained.canonical(r, f.dtype)
        if parse(text, f).ok:
            out[f.key] = (text, min(FALLBACK_MAX_SCORE, constrained.confidence(r)))
    return out


class LocalBackend:
    name = "local-ocr"

    def extract(self, img: np.ndarray) -> PageExtraction:
        t0 = time.time()
        quality = image_quality(img)
        lines = ocr.read_page(img)
        al = align(lines, img)
        if al is None or al.confidence < MIN_PAGE_CONFIDENCE:
            # not the specimen layout: maybe a page of the real pink booklet
            lay, score = booklet.classify(lines)
            if lay is not None:
                return self._booklet(img, lines, lay, score, quality, t0)
            # unknown layout: better no answer than a wrong one
            return finish("unknown", 0.0 if al is None else al.confidence, {}, quality, self.name, VERSION,
                          int((time.time() - t0) * 1000),
                          ["page not recognised as a known registry page: retake the photo or enter it manually"])

        tpl = load_templates()[al.page_type]
        page_conf = (0.7 + 0.3 * al.confidence) * (1.0 if quality.ok else 0.9)
        H = al.H
        fields = {}

        threshold = known_threshold(quality.sharpness)

        # 1) text fields: rectified crop -> ink analysis -> batched OCR on the inked ones
        todo = []
        for f in fields_for(al.page_type):
            geo = tpl.fields.get(f.key)
            if geo is None or f.kind not in ("text", "cell"):
                continue
            ink = analyze_text_crop(warp_region(img, H, geo["region"]),
                                    printed=printed_crop(al.page_type, geo["region"]))
            todo.append((f, ink))
        need = [(f, ink) for f, ink in todo if ink.present and not ink.is_dash]
        crops = {f.key: text_crop_for_ocr(i) for f, i in need}
        reads = dict(zip(crops, ocr.read_crops(list(crops.values())))) if crops else {}
        fallback = _plausible_fallback(need, crops, reads) if crops else {}
        for f, ink in todo:
            text, score = reads.get(f.key, (None, 0.0))
            issues = []
            if f.key in fallback:  # the free reading made no sense: the field's most likely plausible value
                text, score = fallback[f.key]
                issues = ["read as the closest plausible value: please confirm"]
            parsed = parse(text, f)
            d = decide_text(ink, text, score, parsed, page_conf, f.dtype, threshold)
            if issues and d.status == Status.NEEDS_REVIEW:
                d.issues = issues + [i for i in d.issues if i not in issues]
            keep = d.status in (Status.KNOWN, Status.NEEDS_REVIEW)
            fields[f.key] = field_result(f, d.status, d.confidence,
                                         value=parsed.value if keep else None,
                                         display=parsed.display if d.status != Status.NOT_PROVIDED else None,
                                         raw=text, issues=d.issues)

        # 2) checkboxes
        for f in fields_for(al.page_type):
            geo = tpl.fields.get(f.key)
            if geo is None:
                continue
            if f.kind == "check":
                ticked, d = decide_check(checkbox_score(img, H, geo["box"]), page_conf)
                fields[f.key] = field_result(f, d.status, d.confidence, value=ticked,
                                             display="☒" if ticked else "☐", issues=d.issues)
            elif f.kind == "radio":
                scores = {v: checkbox_score(img, H, b) for v, b in geo["options"].items()}
                val, d = decide_radio(scores, page_conf)
                fields[f.key] = field_result(f, d.status, d.confidence, value=val, display=val, issues=d.issues)

        # keep catalog order
        ordered = {f.key: fields[f.key] for f in fields_for(al.page_type) if f.key in fields}
        validate.check(al.page_type, ordered)
        ocr.trim_memory()
        warnings = list(al.warnings) + [f"image quality: {i}" for i in quality.issues]
        return finish(al.page_type, al.confidence, ordered, quality, self.name, VERSION,
                      int((time.time() - t0) * 1000), warnings, layout="specimen")

    def _booklet(self, img, lines, lay, score, quality, t0) -> PageExtraction:
        threshold = known_threshold(0)  # always a photo
        fields, warns, _, _ = booklet.extract(img, lines, lay, threshold)
        ordered = {f.key: fields[f.key] for f in fields_for(lay.page_type) if f.key in fields}
        validate.check(lay.page_type, ordered)
        ocr.trim_memory()
        warnings = [f"real booklet page ({lay.name}): handwriting is harder to read, more fields go to review"]
        warnings += warns + [f"image quality: {i}" for i in quality.issues]
        return finish(lay.page_type, score, ordered, quality, self.name, VERSION, int((time.time() - t0) * 1000),
                      warnings, layout=lay.name)
