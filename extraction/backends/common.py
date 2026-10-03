"""Helpers shared by all backends to build the output objects."""

from __future__ import annotations

from ..catalog import PAGE_TYPES, Field, fields_for
from ..schema import FieldResult, ImageQuality, PageExtraction, Status


def field_result(f: Field, status: Status, confidence: float, value=None, display=None, raw=None,
                 issues=None) -> FieldResult:
    return FieldResult(key=f.key, label_fr=f.label_fr, label_en=f.label_en, section=f.section, kind=f.kind,
                       dtype=f.dtype, value=value, display=display, raw=raw, status=status,
                       confidence=round(float(confidence), 3), issues=issues or [], question_fr=f.question_fr,
                       question_en=f.question_en)


def finish(page_type: str, page_conf: float, fields: dict[str, FieldResult], quality: ImageQuality,
           backend: str, version: str, ms: int, warnings: list[str]) -> PageExtraction:
    review = [k for k, r in fields.items() if r.status in (Status.NEEDS_REVIEW, Status.ILLEGIBLE)]
    review.sort(key=lambda k: fields[k].confidence)
    summary: dict[str, int] = {}
    for r in fields.values():
        summary[r.status.value] = summary.get(r.status.value, 0) + 1
    rec = fields.get("record_number")
    record_number = rec.value if rec is not None and rec.status in (Status.KNOWN, Status.NEEDS_REVIEW) else None
    if rec is not None and rec.status != Status.KNOWN:
        warnings = warnings + ["record number (patient code) needs confirmation before linking"]
    return PageExtraction(page_type=page_type, page_type_label=PAGE_TYPES.get(page_type, "Page non reconnue"),
                          page_type_confidence=round(page_conf, 3), record_number=record_number, fields=fields,
                          needs_review=review, summary=summary, image_quality=quality, backend=backend,
                          model_version=version, processing_ms=ms, warnings=warnings)


def empty_form(page_type: str, backend: str = "manual") -> PageExtraction:
    """All fields NOT_PROVIDED: the form the chat walks through for full manual entry
    (AI unavailable, or the midwife prefers to type)."""
    fields = {f.key: field_result(f, Status.NOT_PROVIDED, 0.0) for f in fields_for(page_type)}
    q = ImageQuality(ok=True, sharpness=0.0, brightness=0.0, issues=[])
    return finish(page_type, 1.0, fields, q, backend, "manual", 0, [])
