from fastapi import APIRouter

from database.database import get_state
from services import record_service as rs

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("")
def search(code: str):
    """Profiles whose code matches exactly or differs by one character (misread digit)."""
    return rs.search_patients(get_state(), code)


@router.get("/new-code")
def new_code():
    return {"code": rs.new_patient_code(get_state())}


@router.get("/{code}")
def get_patient(code: str):
    return rs.patient_view(get_state(), code)
