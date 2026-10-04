"""The AI queue, sync to the health system, the demo network switch, and the status counts."""
from fastapi import APIRouter

from core import config
from database.database import get_state
from offline import State, retry_failed_sync, sync_saved
from schemas.api import ConnectivityIn
from services.ai_service import active_extractor, run_queue

router = APIRouter(tags=["queue & sync"])


def _counts() -> dict:
    """How many records are in each lifecycle state."""
    s = get_state()
    return {st.value: len(s.store.list_by_state(st)) for st in State}


@router.get("/status")
def status():
    """What the chat's header shows: online or not, which extractor, records per state, and how many
    records the health-system server received."""
    s = get_state()
    return {"online": s.net.is_online(), "extractor": active_extractor(),
            "ai_needs_network": config.AI_NEEDS_NETWORK, "counts": _counts(),
            "server_received": len(s.server_received)}


@router.post("/connectivity")
def set_connectivity(body: ConnectivityIn):
    """Demo switch: simulate losing / getting back the connection."""
    get_state().net.set_online(body.online)
    return status()


@router.post("/queue/process")
def process():
    """Run the AI on every queued photo (PENDING_AI), oldest first. Takes a few seconds per page."""
    return run_queue(get_state())


@router.post("/sync")
def sync():
    """Send SAVED records to the health-system server. While offline they just wait."""
    s = get_state()
    retry_failed_sync(s.store)

    def send(rec):  # the pretend server: the real one would receive the record over HTTPS
        s.server_received.append(rec["id"])

    return sync_saved(s.store, s.net, send)
