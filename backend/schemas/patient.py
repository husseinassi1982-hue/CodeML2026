from pydantic import BaseModel

from schemas.field import ExtractField

class PatientBackground(BaseModel):
    age: ExtractField
    village: ExtractField
    consanguinity: ExtractField
    desired_pregnancy: ExtractField

class MedicalHistory(BaseModel):
    hypertension: ExtractField
    diabetes: ExtractField
    hereditary_diseases: ExtractField
    malinformations: ExtractField
    allergies: ExtractField

class ObstericHistory(BaseModel):
    abortions: ExtractField
    premature_deliveries: ExtractField
    fetal_deaths: ExtractField
    gravidity: ExtractField
    parity: ExtractField
    living_children: ExtractField