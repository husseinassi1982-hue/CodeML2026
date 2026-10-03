from pydantic import BaseModel

from schemas.delivery import DeliveryData
from schemas.patient import (
    MedicalHistory,
    ObstericHistory,
    PatientBackground,
)
from schemas.pregnancy import PregnancyData

class MaternalRecord(BaseModel):
    background: PatientBackground
    medical_history