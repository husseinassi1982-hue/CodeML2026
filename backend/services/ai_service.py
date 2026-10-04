"""Runs the extraction pipeline on queued photos (the `extract_fn` the offline queue calls)."""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache

from core import config
from database.database import AppState
from offline import Connectivity, State, process_queue


class PermanentFailure(Exception):
    """The photo itself is the problem (blank, corrupt...). Retrying won't help: ask for a retake.
    The offline queue sees `permanent = True` and marks the record PROCESSING_FAILED at once."""
    permanent = True


@lru_cache(maxsize=1)
def ocr_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("rapidocr", "cv2", "pymupdf", "onnxruntime"))


def active_extractor() -> str:
    """What actually runs: the configured backend, or fixtures if the OCR isn't installed."""
    if config.EXTRACTOR in ("local", "vlm") and not ocr_available():
        return "fixture"
    return config.EXTRACTOR


def fixture_names() -> list[str]:
    return sorted(p.stem for p in config.FIXTURES_DIR.glob("*.json")
                  if not p.name.endswith(".ui.json") and not p.stem.startswith("manual_"))


def load_fixture(name: str) -> dict:
    if name not in fixture_names():
        raise ValueError(f"unknown demo page {name!r}; choose one of {fixture_names()}")
    return json.loads((config.FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))


def make_extract_fn(state: AppState):
    def extract_fn(image_bytes: bytes, record_id: str) -> dict:
        demo_page = state.store.get(record_id)["fields"].get("demo_page")
        mode = active_extractor()
        if demo_page or mode == "fixture":
            return {"extraction": load_fixture(demo_page or config.DEFAULT_FIXTURE)}

        from extraction import ExtractionError, extract
        try:
            result = extract(image_bytes, backend=mode)
        except ExtractionError as e:
            if e.code == "unreadable_image":
                raise PermanentFailure(f"unreadable_image: {e}") from e
            if mode in ("claude", "vlm"):
                # cloud down / no GPU model on this machine: wait in the queue, no attempt used
                raise ConnectionError(str(e)) from e
            raise  # local model failed: counts as an attempt, gives up after 3
        return {"extraction": result.model_dump(mode="json")}

    return extract_fn


_ALWAYS_ONLINE = Connectivity(online=True)


def run_queue(state: AppState) -> dict:
    """Process every PENDING_AI record. Local OCR runs offline; only the cloud backend
    (or DAYONE_AI_NEEDS_NETWORK=1) waits for the connection."""
    net = state.net if config.AI_NEEDS_NETWORK else _ALWAYS_ONLINE
    with state.queue_lock:
        report = process_queue(state.store, net, make_extract_fn(state))
        # A page the extractor couldn't recognise goes to manual review instead of the normal review.
        for rid in list(report["processed"]):
            ext = state.store.get(rid)["fields"].get("extraction") or {}
            if ext.get("page_type", "unknown") == "unknown":
                state.store.transition(rid, State.MANUAL_REVIEW_REQUIRED, note="page type not recognised")
    return report
