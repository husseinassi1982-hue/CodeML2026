from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, Form, Query, UploadFile

from database.database import get_state
from offline import State
from schemas.api import AnswersIn, LinkIn, ManualEntryIn
from services import record_service as rs
from services.ai_service import fixture_names, load_fixture

router = APIRouter(prefix="/records", tags=["records"])

MAX_PHOTO_BYTES = 20 * 1024 * 1024


@router.post("", status_code=201)
async def capture(photo: UploadFile = File(...), midwife_id: str = Form("mw-demo"),
                  captured_at: Optional[str] = Form(None, description="when the phone took the photo (ISO 8601), "
                                                                       "if it was kept offline and uploaded later"),
                  demo_page: Optional[str] = Form(None, description="demo only: use this saved real output "
                                                                     "instead of running OCR (see GET /demo/pages)")):
    """The phone uploads a photo of a registry page (right away, or later from its offline outbox).
    It's encrypted and saved, then queued (PENDING_AI), or parked as SUSPECTED_DUPLICATE if this
    exact photo was seen before."""
    data = await photo.read()
    if not data:
        raise rs.ApiError(422, "empty photo")
    if len(data) > MAX_PHOTO_BYTES:
        raise rs.ApiError(413, "photo too large (20 MB max)")
    taken = _parse_time(captured_at) if captured_at else None
    if demo_page is not None:
        load_fixture(demo_page)  # validates the name before we store anything
    state = get_state()
    rid = state.store.capture(data, midwife_id=midwife_id, captured_at=taken)
    if demo_page:
        state.store.update_fields(rid, {"demo_page": demo_page})
    return rs.record_view(state, rs.get_record(state, rid))


def _parse_time(text: str) -> str:
    """The phone's capture time (ISO 8601) as UTC ISO text, like the store's own timestamps."""
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise rs.ApiError(422, "captured_at must be an ISO 8601 date-time")
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    # A phone clock running ahead must not block its outbox: never later than now.
    return min(t, datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


@router.get("")
def list_records(state_: Optional[list[State]] = Query(None, alias="state")):
    """All records, oldest first; filter with ?state=TO_REVIEW&state=PENDING_AI."""
    s = get_state()
    return [rs.record_view(s, r) for r in s.store.list_by_state(*(state_ or list(State)))]


@router.get("/{record_id}")
def get_record(record_id: str):
    """Everything about one record: every field with status/confidence/questions, plus its history."""
    s = get_state()
    return rs.record_view(s, rs.get_record(s, record_id), full=True)


@router.patch("/{record_id}/fields")
def answer(record_id: str, body: AnswersIn):
    """The midwife's answers: a typed value, 'that's right', or UNKNOWN / NOT_APPLICABLE."""
    return rs.apply_answers(get_state(), record_id, {k: v.model_dump() for k, v in body.answers.items()})


@router.post("/{record_id}/manual")
def manual(record_id: str, body: ManualEntryIn):
    """Page not recognised or AI failed: start an empty form of this page type to fill by chat."""
    return rs.start_manual_entry(get_state(), record_id, body.page_type)


@router.post("/{record_id}/validate")
def validate(record_id: str):
    return rs.validate(get_state(), record_id)


@router.post("/{record_id}/retry")
def retry(record_id: str):
    """Send the record back to the AI (after a failure, or to re-read it)."""
    return rs.retry(get_state(), record_id)


@router.post("/{record_id}/link")
def link(record_id: str, body: LinkIn):
    """Attach a validated record to a patient. Unknown code -> 404 with near-match candidates."""
    return rs.link(get_state(), record_id, body.patient_code, body.create)


demo_router = APIRouter(prefix="/demo", tags=["demo"])


@demo_router.get("/pages")
def demo_pages():
    """Saved real extractor outputs usable as demo_page when capturing."""
    return fixture_names()
