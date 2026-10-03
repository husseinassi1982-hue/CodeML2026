from pydantic import BaseModel

from schemas.field import ExtractField

class PregnancyData(BaseModel):
    expected_delivery_date: ExtractField
    gestional_age: ExtractField
    weight: ExtractField
    blood_pressure: ExtractField
    temperature: ExtractField
    fetal_movement: ExtractField
    fetal_heart_rate: ExtractField
    hiv_test: ExtractField
    syphilis_test: ExtractField
    hepatitis_c_test: ExtractField
    treatment: ExtractField