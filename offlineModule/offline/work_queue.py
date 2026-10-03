"""The offline queue: a waiting line for records that could not be processed/sent yet.

The queue is not a separate list in memory. It is simply "every record whose
state is PENDING_AI" (or SAVED, for the server sync). Because that lives in the
database, the queue survives the app being closed or crashing.

Golden rule: a record only moves forward AFTER the step succeeded. If anything
goes wrong half-way, the record keeps its old state and is picked up again.
"""
from .states import State
from .store import RecordStore

MAX_ATTEMPTS = 3


class Connectivity:
    """Simulated internet switch. The demo/UI flips it with set_online()."""

    def __init__(self, online: bool = True):
        self._online = online

    def set_online(self, online: bool) -> None:
        self._online = online

    def is_online(self) -> bool:
        return self._online


def process_queue(store: RecordStore, net: Connectivity, extract_fn,
                  max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Send every PENDING_AI record to the AI, oldest first.

    extract_fn(image_bytes, record_id) -> dict of fields (module A's job).
      - raise ConnectionError  -> internet problem: leave the record waiting,
                                  do not count it against the record.
      - raise anything else    -> the AI failed on this record: count an attempt;
                                  after max_attempts mark PROCESSING_FAILED.
    Returns a small report: {"processed": [...], "waiting": [...], "failed": [...]}.
    """
    report = {"processed": [], "waiting": [], "failed": []}
    for rec in store.list_by_state(State.PENDING_AI):
        rid = rec["id"]
        if not net.is_online():           # checked per record: connection can drop mid-way
            report["waiting"].append(rid)
            continue
        try:
            fields = extract_fn(store.load_image_for_processing(rid), rid)
        except ConnectionError:
            report["waiting"].append(rid)  # stays PENDING_AI, tried again next time
            continue
        except Exception as exc:
            attempts = store.bump_attempts(rid)
            if attempts >= max_attempts:
                store.transition(rid, State.PROCESSING_FAILED, error=str(exc),
                                 note=f"gave up after {attempts} attempts")
                report["failed"].append(rid)
            else:
                report["waiting"].append(rid)
            continue
        store.update_fields(rid, fields)
        store.transition(rid, State.AI_PROCESSED, note="AI returned fields")
        store.transition(rid, State.TO_REVIEW, note="ready for the midwife")
        report["processed"].append(rid)
    return report


def retry_failed_processing(store: RecordStore) -> list[str]:
    """Put PROCESSING_FAILED records back in the queue (e.g. a 'retry' button)."""
    ids = []
    for rec in store.list_by_state(State.PROCESSING_FAILED):
        store.reset_attempts(rec["id"])
        store.transition(rec["id"], State.PENDING_AI, note="manual retry")
        ids.append(rec["id"])
    return ids


def sync_saved(store: RecordStore, net: Connectivity, sync_fn) -> dict:
    """Send every SAVED record to the server. sync_fn(record_dict) raises on failure."""
    report = {"synced": [], "waiting": [], "failed": []}
    for rec in store.list_by_state(State.SAVED):
        rid = rec["id"]
        if not net.is_online():
            report["waiting"].append(rid)
            continue
        try:
            sync_fn(rec)
        except ConnectionError:
            report["waiting"].append(rid)
            continue
        except Exception as exc:
            store.transition(rid, State.SYNC_FAILED, error=str(exc))
            report["failed"].append(rid)
            continue
        store.transition(rid, State.SYNCED, note="server accepted")
        report["synced"].append(rid)
    return report


def retry_failed_sync(store: RecordStore) -> list[str]:
    ids = []
    for rec in store.list_by_state(State.SYNC_FAILED):
        store.transition(rec["id"], State.SAVED, note="manual sync retry")
        ids.append(rec["id"])
    return ids
