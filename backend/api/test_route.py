from fastapi import APIRouter

from schemas.patient import PatientBackground

router = APIRouter()


@router.post("/test-patient")
def test_patient(data: PatientBackground):
    return data