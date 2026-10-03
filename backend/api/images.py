from fastapi import APIRouter, Header
from fastapi.responses import Response

from database.database import get_state
from offline import AccessDenied
from services import record_service as rs

router = APIRouter(prefix="/records", tags=["images"])


@router.get("/{record_id}/image")
def original_photo(record_id: str, x_role: str = Header("", description="midwife or admin")):
    """The original photo, decrypted. Restricted: only the midwife and admins may see it."""
    s = get_state()
    rs.get_record(s, record_id)
    try:
        data = s.store.get_image(record_id, x_role)
    except AccessDenied as e:
        raise rs.ApiError(403, str(e))
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
