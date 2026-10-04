"""Patient profiles: search by code (with near matches), get a new code, look one up."""
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
    """A random code (M-1234) that no profile uses yet, to write on the paper registry."""
    return {"code": rs.new_patient_code(get_state())}


@router.get("/{code}")
def get_patient(code: str):
    """One profile: its visits (linked records) and the last one. 404 with near-match candidates if the
    code is unknown."""
    return rs.patient_view(get_state(), code)
