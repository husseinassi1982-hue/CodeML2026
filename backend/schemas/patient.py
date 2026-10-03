from pydantic import BaseModel

from schemas.field import ExtractField

class PatientBackground(BaseModel):
    age: ExtractField
    village: ExtractField
    consanguinity: ExtractField
    desired_pregnancy: ExtractField