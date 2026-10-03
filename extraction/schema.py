"""Output contract of the extractor: what Person 2 (chat UI) and Person 3 (backend) receive.

One photo of one registry page -> one ``PageExtraction``. Serialise with
``result.model_dump_json(indent=2)``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field as PField

SCHEMA_VERSION = "1.0"


class Status(str, Enum):
    KNOWN = "KNOWN"  # read with confidence and passes the checks
    UNKNOWN = "UNKNOWN"  # the midwife wrote that the value is unknown ("?", "inconnu", "NSP")
    NOT_PROVIDED = "NOT_PROVIDED"  # nothing written on the paper
    ILLEGIBLE = "ILLEGIBLE"  # something is written but cannot be read
    NOT_APPLICABLE = "NOT_APPLICABLE"  # "—" written, or ruled out by another field
    NEEDS_REVIEW = "NEEDS_REVIEW"  # read, but low confidence or failed a plausibility check


class FieldResult(BaseModel):
    key: str
    label_fr: str
    label_en: str
    section: str
    kind: str  # text | cell | check | radio
    dtype: str  # str | int | float | date | bp | weeks | kg | grams | cm | temp | posneg | yesno | sex | days | code | bool | enum
    value: Any = None  # parsed value (int, float, ISO date, {"systolic", "diastolic"}, bool, option...)
    display: str | None = None  # human-readable value for the chat ("102/60", "18/01/2026")
    raw: str | None = None  # text exactly as read from the page
    status: Status
    confidence: float = PField(ge=0, le=1)
    issues: list[str] = []  # why it needs review, e.g. "out of range (34-42)"
    question_fr: str
    question_en: str


class ImageQuality(BaseModel):
    ok: bool
    sharpness: float  # variance of the Laplacian; < ~60 is blurry
    brightness: float  # 0..1 mean intensity
    issues: list[str] = []  # e.g. ["blurry", "too dark"] -> ask the midwife to retake


class PageExtraction(BaseModel):
    schema_version: str = SCHEMA_VERSION
    page_type: str  # one of catalog.PAGE_TYPES, or "unknown"
    page_type_label: str
    page_type_confidence: float
    record_number: str | None = None  # midwife's code ("N° de la fiche"), cover page only
    fields: dict[str, FieldResult]
    needs_review: list[str]  # keys to ask about, least confident first
    summary: dict[str, int]  # count per status
    image_quality: ImageQuality
    backend: str  # "local-ocr" | "claude" | "manual"
    model_version: str
    processing_ms: int
    warnings: list[str] = []

    def to_ui_fields(self, only: list[str] | None = None) -> dict[str, dict]:
        """Shape used by the WhatsApp mock-up (sample_0ne.HTML): {key: {label, value, conf, status}}."""
        keys = only or list(self.fields)
        return {
            k: {
                "label": self.fields[k].label_en,
                "value": self.fields[k].display if self.fields[k].display is not None else "—",
                "conf": round(self.fields[k].confidence, 2),
                "status": self.fields[k].status.value,
            }
            for k in keys
        }
