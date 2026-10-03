"""Record lifecycle: the labels a record can have and which moves are legal.

A "state machine" is just this: a list of labels (States) plus a table that says
which label can follow which. Nothing in the project sets a state directly;
everything goes through `check_transition`, so an illegal jump like
CAPTURED -> SYNCED is rejected instead of silently corrupting a record.
"""
from enum import Enum


class State(str, Enum):
    # Happy path
    CAPTURED = "CAPTURED"                  # photo saved on the device
    PENDING_AI = "PENDING_AI"              # waiting in the queue for AI processing
    AI_PROCESSED = "AI_PROCESSED"          # AI returned structured fields
    TO_REVIEW = "TO_REVIEW"                # waiting for the midwife to check
    VALIDATED = "VALIDATED"                # midwife confirmed the fields
    PATIENT_LINKED = "PATIENT_LINKED"      # attached to a patient profile
    SAVED = "SAVED"                        # stored locally in final form
    SYNCED = "SYNCED"                      # sent to the server (terminal)
    # Failure states
    PROCESSING_FAILED = "PROCESSING_FAILED"
    SYNC_FAILED = "SYNC_FAILED"
    SUSPECTED_DUPLICATE = "SUSPECTED_DUPLICATE"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


S = State

# state -> set of states it is allowed to move to
ALLOWED = {
    S.CAPTURED: {S.PENDING_AI, S.SUSPECTED_DUPLICATE},
    S.PENDING_AI: {S.AI_PROCESSED, S.PROCESSING_FAILED},
    S.AI_PROCESSED: {S.TO_REVIEW},
    # PENDING_AI again = "retake / re-run the AI on this record"
    S.TO_REVIEW: {S.VALIDATED, S.PENDING_AI, S.MANUAL_REVIEW_REQUIRED},
    S.VALIDATED: {S.PATIENT_LINKED},
    S.PATIENT_LINKED: {S.SAVED},
    S.SAVED: {S.SYNCED, S.SYNC_FAILED},
    S.SYNCED: set(),  # terminal
    # Every failure state has a way out, so nothing gets stuck forever.
    S.PROCESSING_FAILED: {S.PENDING_AI, S.MANUAL_REVIEW_REQUIRED},
    S.SYNC_FAILED: {S.SAVED, S.MANUAL_REVIEW_REQUIRED},
    S.SUSPECTED_DUPLICATE: {S.PENDING_AI, S.MANUAL_REVIEW_REQUIRED},
    # Manual entry finished -> the midwife reviews it like any other record.
    S.MANUAL_REVIEW_REQUIRED: {S.TO_REVIEW, S.PENDING_AI},
}


class IllegalTransition(Exception):
    pass


def check_transition(current: State, new: State) -> None:
    if new not in ALLOWED[current]:
        raise IllegalTransition(f"{current.value} -> {new.value} is not allowed")
