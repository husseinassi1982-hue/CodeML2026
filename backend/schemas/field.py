from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

class FieldStatus(str, Enum):
    KNOWN = "KNOWN"
    UKNOWN = "UKNOWN"
    NOT_PROVIDED = "NOT_PROVIDED"
    ILLEGIBLE = "ILLEGIBLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"

class ExtractField(BaseModel):
    value: Optional[Any] = None

    status: FieldStatus

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )