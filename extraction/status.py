"""Turn the raw evidence for one field into (status, confidence).

The agent must never hide doubt, so every path ends in an explicit status:
  no ink in the field                -> NOT_PROVIDED
  a dash, or "?"/"inconnu" written    -> NOT_APPLICABLE / UNKNOWN
  ink but OCR could not read it      -> ILLEGIBLE
  read, but unparseable / implausible / low confidence -> NEEDS_REVIEW
  otherwise                          -> KNOWN

Thresholds are calibrated on the development patients (see tools/evaluate.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .imageproc import Ink
from .normalize import Parsed
from .schema import Status

KNOWN_THRESHOLD = 0.80  # below this a readable value is sent to the midwife for review (scans)
KNOWN_THRESHOLD_PHOTO = 0.88  # stricter for phone photos: OCR is over-confident on blurred text
PHOTO_SHARPNESS = 900  # image_quality().sharpness below this looks like a photo, not a scan


def known_threshold(sharpness: float) -> float:
    """Calibrated on the dev patients (tools/evaluate.py): on simulated photos 0.88 halves the
    silent errors of 0.80; on clean scans a higher bar only adds review work."""
    return KNOWN_THRESHOLD if sharpness >= PHOTO_SHARPNESS else KNOWN_THRESHOLD_PHOTO
ILLEGIBLE_OCR = 0.35  # OCR score under which the text is treated as unreadable
CHECK_THRESHOLD = 0.07  # fraction of box interior inked to call it ticked
CHECK_SPREAD = 0.06  # distance from the threshold that gives full confidence


@dataclass
class Decision:
    status: Status
    confidence: float
    issues: list[str]


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def decide_text(ink: Ink, ocr_text: str | None, ocr_score: float, parsed: Parsed, page_conf: float,
                dtype: str = "str", threshold: float = KNOWN_THRESHOLD) -> Decision:
    if not ink.present:
        conf = 0.97 if ink.fraction == 0 else 0.6  # faint marks below the ink threshold: less sure
        return Decision(Status.NOT_PROVIDED, _clip(conf * page_conf), [])
    if ink.is_dash or parsed.marker == "dash":
        return Decision(Status.NOT_APPLICABLE, _clip(0.9 * page_conf), [])
    if parsed.marker == "unknown":
        return Decision(Status.UNKNOWN, _clip(ocr_score * page_conf), [])
    if not ocr_text or ocr_score < ILLEGIBLE_OCR:
        return Decision(Status.ILLEGIBLE, _clip(0.3 * page_conf), ["handwriting could not be read"])
    if not parsed.ok:
        return Decision(Status.NEEDS_REVIEW, _clip(0.5 * ocr_score * page_conf), list(parsed.issues))

    conf = ocr_score * page_conf
    if dtype in ("yesno", "posneg", "sex") and parsed.snapped == 1.0:
        # an exact match to one of 2-3 allowed answers is strong evidence on its own;
        # OCR scores short words low even when they are right
        conf = (0.5 + 0.5 * ocr_score) * page_conf
    if parsed.snapped < 1.0:  # corrected to the nearest known term
        conf *= 0.6 + 0.4 * parsed.snapped
    issues = list(parsed.issues)
    if issues:
        return Decision(Status.NEEDS_REVIEW, _clip(min(conf, 0.5)), issues)
    if conf < threshold:
        return Decision(Status.NEEDS_REVIEW, _clip(conf), ["low reading confidence"])
    return Decision(Status.KNOWN, _clip(conf), [])


def check_confidence(score: float) -> float:
    return _clip(0.5 + abs(score - CHECK_THRESHOLD) / (2 * CHECK_SPREAD))


def decide_check(score: float, page_conf: float) -> tuple[bool, Decision]:
    ticked = score >= CHECK_THRESHOLD
    conf = check_confidence(score) * page_conf
    if conf < 0.65:
        return ticked, Decision(Status.NEEDS_REVIEW, _clip(conf), ["unclear whether the box is ticked"])
    return ticked, Decision(Status.KNOWN, _clip(conf), [])


def decide_radio(scores: dict[str, float], page_conf: float) -> tuple[str | None, Decision]:
    ticked = {k: s for k, s in scores.items() if s >= CHECK_THRESHOLD}
    conf = min(check_confidence(s) for s in scores.values()) * page_conf if scores else 0.0
    if not ticked:
        return None, Decision(Status.NOT_PROVIDED, _clip(conf), [])
    best = max(ticked, key=ticked.get)
    if len(ticked) > 1:
        return best, Decision(Status.NEEDS_REVIEW, _clip(min(conf, 0.5)),
                              [f"several boxes ticked: {', '.join(sorted(ticked))}"])
    if conf < 0.65:
        return best, Decision(Status.NEEDS_REVIEW, _clip(conf), ["unclear which box is ticked"])
    return best, Decision(Status.KNOWN, _clip(conf), [])
