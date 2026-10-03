"""Request bodies. Responses are plain dicts built in services/record_service.py; the field
objects inside them follow extraction/schema.py (FieldResult)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Answer(BaseModel):
    """One answer from the midwife. Exactly one of: text, confirm, status."""
    text: Optional[str] = Field(None, description="the value she typed, e.g. '120/80'")
    confirm: bool = Field(False, description="the value read by the AI is right")
    status: Optional[Literal["UNKNOWN", "NOT_APPLICABLE", "NOT_PROVIDED", "ILLEGIBLE"]] = None

    @model_validator(mode="after")
    def exactly_one(self):
        given = [self.text is not None, self.confirm, self.status is not None]
        if sum(given) != 1:
            raise ValueError("give exactly one of: text, confirm, status")
        return self


class AnswersIn(BaseModel):
    answers: dict[str, Answer] = Field(..., examples=[{"record_number": {"confirm": True},
                                                      "facility_name": {"text": "CSCA Al Wifaq"}}])


class ManualEntryIn(BaseModel):
    page_type: str = Field(..., examples=["cover"])


class LinkIn(BaseModel):
    patient_code: Optional[str] = Field(None, examples=["2026-823-001"])
    create: bool = Field(False, description="create the profile if the code is new (or generate a code if none given)")


class ConnectivityIn(BaseModel):
    online: bool
