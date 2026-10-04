"""What the API does to a record: build the view the chat needs, apply the midwife's
answers, validate, link to a patient. Every state change goes through RecordStore.transition,
so the lifecycle rules in offlineModule/offline/states.py are always enforced."""
from __future__ import annotations

import random

from database.database import AppState
from offline import State
from services.patient_matching import candidates, normalise_code

EDITABLE_STATES = {State.TO_REVIEW, State.MANUAL_REVIEW_REQUIRED}
REVIEW_STATUSES = {"NEEDS_REVIEW", "ILLEGIBLE"}


class ApiError(Exception):
    """A request the API refuses: becomes an HTTP error with this status code and message. extra adds
    keys to the JSON body (e.g. near-match candidates)."""

    def __init__(self, status_code: int, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.extra = extra or {}


# ---------- views ----------
def _field_view(f: dict) -> dict:
    """The parts of one extracted field (FieldResult) the chat needs."""
    return {k: f.get(k) for k in ("key", "label_en", "label_fr", "kind", "dtype", "value", "display", "status",
                                  "confidence", "issues", "question_en", "question_fr", "corrected_by_midwife")}


def record_view(state: AppState, rec: dict, full: bool = False) -> dict:
    """A record as the chat sees it: state, page, summary, the questions to ask (least confident first)
    and the filled fields. full=True adds every field and the state history."""
    ext = rec["fields"].get("extraction") or {}
    fields = ext.get("fields", {})
    view = {
        "id": rec["id"],
        "state": rec["state"].value,
        "captured_at": rec["captured_at"],
        "updated_at": rec["updated_at"],
        "attempts": rec["attempts"],
        "last_error": rec["last_error"],
        "patient_code": state.patients.code_for_record(rec["id"]),
        "demo_page": rec["fields"].get("demo_page"),
        "extracted": bool(ext),
        "page_type": ext.get("page_type"),
        "page_type_label": ext.get("page_type_label"),
        "page_type_confidence": ext.get("page_type_confidence"),
        "record_number": ext.get("record_number"),
        "image_quality": ext.get("image_quality"),
        "summary": ext.get("summary"),
        "warnings": ext.get("warnings", []),
        "backend": ext.get("backend"),
        # questions to ask, least confident first
        "needs_review": [_field_view(fields[k]) for k in ext.get("needs_review", []) if k in fields],
        # everything that has something written, for the summary message
        "filled": [_field_view(f) for f in fields.values()
                   if f.get("display") not in (None, "", "☐") and f.get("status") not in ("NOT_PROVIDED",)],
    }
    if full:
        view["fields"] = {k: _field_view(f) for k, f in fields.items()}
        view["history"] = state.store.history(rec["id"])
    return view


def get_record(state: AppState, record_id: str) -> dict:
    """The stored record, or ApiError 404."""
    try:
        return state.store.get(record_id)
    except KeyError:
        raise ApiError(404, f"record {record_id} not found")


# ---------- the midwife's answers ----------
def _recompute(ext: dict) -> None:
    """After answers: refresh the review queue (least confident first), the count per status and the
    patient code read on the page."""
    fields = ext["fields"]
    review = [k for k, f in fields.items() if f["status"] in REVIEW_STATUSES]
    review.sort(key=lambda k: fields[k]["confidence"])
    ext["needs_review"] = review
    summary: dict[str, int] = {}
    for f in fields.values():
        summary[f["status"]] = summary.get(f["status"], 0) + 1
    ext["summary"] = summary
    rec = fields.get("record_number")
    if rec is not None:
        ext["record_number"] = rec["value"] if rec["status"] in ("KNOWN", "NEEDS_REVIEW") else None


YES = {"oui", "o", "yes", "y", "x", "1", "vrai", "true", "coché", "coche", "✓", "☑"}
NO = {"non", "n", "no", "0", "faux", "false", "☐"}
FREE_TEXT = {"str", "code"}  # anything goes; every other type must parse


def _typed_value(text: str, f, label: str):
    """Parse what the midwife typed for field definition f. Returns (status, value, display, issues);
    raises ApiError(422) with a helpful message when it doesn't fit the field, so the chat re-asks."""
    from extraction.normalize import UNKNOWN_MARKERS, parse

    low = text.strip().lower()
    if low in UNKNOWN_MARKERS:
        return "UNKNOWN", None, "?", []
    if f is None:  # page type unknown: no definition to check against
        return "KNOWN", text, text, []
    if f.kind == "check":
        if low in YES:
            return "KNOWN", True, "☑", []
        if low in NO:
            return "KNOWN", False, "☐", []
        raise ApiError(422, f"“{text}” — for {label} answer oui / non")
    if f.kind == "radio":
        opts = [o.value for o in (f.options or [])]
        hit = next((o for o in opts if o.lower() == low), None)
        if hit is None:
            raise ApiError(422, f"“{text}” — for {label} choose one of: {', '.join(opts)}")
        return "KNOWN", hit, hit, []
    p = parse(text, f)
    if p.marker == "unknown":
        return "UNKNOWN", None, "?", []
    if p.marker == "dash":
        return "NOT_APPLICABLE", None, "—", []
    if p.ok:
        return "KNOWN", p.value, p.display or text, list(p.issues)
    if f.dtype in FREE_TEXT:
        return "KNOWN", text, text, ["kept as typed"]
    hint = "; ".join(p.issues) if p.issues else f"expected a {f.dtype} value"
    raise ApiError(422, f"“{text}” doesn't look right for {label} ({hint})")


def _apply_one(field: dict, answer: dict, catalog_field) -> None:
    """Apply one answer to one field: confirm the value read, set a status (UNKNOWN, NOT_APPLICABLE...),
    or a typed value checked against the field's definition. Either way the field is then marked as
    corrected by the midwife."""
    if answer.get("confirm"):
        field.update(status="KNOWN", confidence=1.0, corrected_by_midwife=True)
        field["issues"] = [i for i in field.get("issues", []) if "confirm" not in i] + ["confirmed by midwife"]
        return
    if answer.get("status"):
        st = answer["status"]
        field.update(value=None, display={"UNKNOWN": "?", "NOT_APPLICABLE": "—"}.get(st), status=st,
                     confidence=1.0, corrected_by_midwife=True, issues=["set by midwife"])
        return
    text = (answer.get("text") or "").strip()
    if not text:
        raise ApiError(422, f"empty answer for {field['label_en']}")
    status, value, display, issues = _typed_value(text, catalog_field, field["label_en"])
    field.update(status=status, value=value, display=display, raw=text, confidence=1.0,
                 corrected_by_midwife=True, issues=["entered by midwife"] + issues)


def apply_answers(state: AppState, record_id: str, answers: dict[str, dict]) -> dict:
    """Apply the midwife's answers to a record under review. All or nothing: if one answer is refused
    (ApiError 422), none is saved. A page entered by hand then moves on to TO_REVIEW."""
    from extraction.catalog import fields_for

    rec = get_record(state, record_id)
    if rec["state"] not in EDITABLE_STATES:
        raise ApiError(409, f"record is {rec['state'].value}; fields can only be changed while it is under review")
    ext = rec["fields"].get("extraction")
    if not ext or not ext.get("fields"):
        raise ApiError(409, "nothing extracted yet: run the queue, or start manual entry")
    catalog = {f.key: f for f in fields_for(ext["page_type"])} if ext.get("page_type") != "unknown" else {}
    unknown_keys = [k for k in answers if k not in ext["fields"]]
    if unknown_keys:
        raise ApiError(422, f"unknown field(s): {unknown_keys}")
    for key, ans in answers.items():  # if any answer is rejected, nothing is saved
        _apply_one(ext["fields"][key], ans, catalog.get(key))
    _recompute(ext)
    state.store.set_fields(record_id, {**rec["fields"], "extraction": ext})
    if rec["state"] == State.MANUAL_REVIEW_REQUIRED:
        state.store.transition(record_id, State.TO_REVIEW, note="manual entry by midwife")
    return record_view(state, get_record(state, record_id))


def start_manual_entry(state: AppState, record_id: str, page_type: str) -> dict:
    """The AI couldn't handle the page: give the record an empty form of this page type, for the midwife
    to fill by chat (state MANUAL_REVIEW_REQUIRED)."""
    from extraction import PAGE_TYPES, empty_form

    if page_type not in PAGE_TYPES:
        raise ApiError(422, f"unknown page type; choose one of {sorted(PAGE_TYPES)}")
    rec = get_record(state, record_id)
    if rec["state"] not in (State.MANUAL_REVIEW_REQUIRED, State.PROCESSING_FAILED):
        raise ApiError(409, f"record is {rec['state'].value}; manual entry is for pages the AI couldn't handle")
    if rec["state"] == State.PROCESSING_FAILED:
        state.store.transition(record_id, State.MANUAL_REVIEW_REQUIRED, note="midwife chose manual entry")
    ext = empty_form(page_type).model_dump(mode="json")
    state.store.set_fields(record_id, {**rec["fields"], "extraction": ext})
    return record_view(state, get_record(state, record_id), full=True)


# ---------- lifecycle ----------
def validate(state: AppState, record_id: str) -> dict:
    """The midwife confirms the record: TO_REVIEW -> VALIDATED (409 in any other state)."""
    rec = get_record(state, record_id)
    if rec["state"] != State.TO_REVIEW:
        raise ApiError(409, f"record is {rec['state'].value}; only records under review can be validated")
    state.store.transition(record_id, State.VALIDATED, note="confirmed by midwife")
    return record_view(state, get_record(state, record_id))


def retry(state: AppState, record_id: str) -> dict:
    """Send a record back to the AI (PENDING_AI) with a fresh attempt count, if its state allows it."""
    rec = get_record(state, record_id)
    if State.PENDING_AI not in _allowed(rec["state"]):
        raise ApiError(409, f"record is {rec['state'].value}; it can't be sent back to the AI")
    state.store.reset_attempts(record_id)
    state.store.transition(record_id, State.PENDING_AI, note="re-run requested")
    return record_view(state, get_record(state, record_id))


def _allowed(s: State) -> set:
    """States a record may move to from state s (the lifecycle table in offline)."""
    from offline import ALLOWED
    return ALLOWED[s]


def new_patient_code(state: AppState) -> str:
    """A random code (M-1234) that no profile uses yet."""
    existing = set(state.patients.all_codes())
    while True:
        code = f"M-{random.randint(1000, 9999)}"
        if code not in existing:
            return code


def link(state: AppState, record_id: str, code: str | None, create: bool) -> dict:
    """Attach a validated record to a patient, then save it on the device (VALIDATED -> PATIENT_LINKED
    -> SAVED).

    Never guesses: an unknown code is refused (404 with near-match candidates) unless create=true, and
    create=true without a code generates a new one."""
    rec = get_record(state, record_id)
    if rec["state"] != State.VALIDATED:
        raise ApiError(409, f"record is {rec['state'].value}; validate it before linking")
    if create and not code:
        code = new_patient_code(state)
    if not code:
        raise ApiError(422, "patient_code is required (or create=true for a new profile)")
    code = normalise_code(code)
    if not state.patients.exists(code):
        if not create:
            raise ApiError(404, f"no patient with code {code}",
                           {"candidates": candidates(code, state.patients.all_codes())})
        state.patients.create(code)
    state.patients.link(record_id, code)
    state.store.transition(record_id, State.PATIENT_LINKED, note=f"linked to patient {code}")
    state.store.transition(record_id, State.SAVED, note="saved on device")
    return record_view(state, get_record(state, record_id))


def patient_view(state: AppState, code: str) -> dict:
    """One profile: number of visits, the last visit, and its records. 404 with near-match candidates if
    the code is unknown."""
    code = normalise_code(code)
    if not state.patients.exists(code):
        raise ApiError(404, f"no patient with code {code}",
                       {"candidates": candidates(code, state.patients.all_codes())})
    recs = [state.store.get(r) for r in state.patients.records_for(code)]
    return {
        "code": code,
        "visits": len(recs),
        "last_visit": max((r["captured_at"] for r in recs), default=None),
        "records": [{"id": r["id"], "state": r["state"].value, "captured_at": r["captured_at"],
                     "page_type": (r["fields"].get("extraction") or {}).get("page_type")} for r in recs],
    }


def search_patients(state: AppState, code: str) -> list[dict]:
    """Profiles whose code matches exactly or with one character different, with their visits and the
    pages on file."""
    out = []
    for c in candidates(code, state.patients.all_codes()):
        p = patient_view(state, c["code"])
        out.append({**c, "visits": p["visits"], "last_visit": p["last_visit"],
                    "pages": sorted({r["page_type"] for r in p["records"] if r["page_type"]})})
    return out
