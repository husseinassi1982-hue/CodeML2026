from fastapi import APIRouter

from schemas.record import MaternalRecord

router = APIRouter()


@router.post("/records")
def create_record(record: MaternalRecord):
    return record