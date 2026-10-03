from pydantic import BaseModel

from schemas.field import ExtractField

class DeliveryData(BaseModel):
    place: ExtractField
    date: ExtractField
    mode: ExtractField
    complication: ExtractField
    newborn_status: ExtractField
    newborn_sex: ExtractField
    new_born_weight: ExtractField
    head_circumference: ExtractField
    abnormalities: ExtractField