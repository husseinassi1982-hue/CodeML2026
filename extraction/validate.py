"""Cross-field plausibility checks. A failed check never changes a value: it flags the
fields involved as NEEDS_REVIEW so the midwife confirms them."""

from __future__ import annotations

from datetime import date

from .catalog import VISIT_COLS, fields_for
from .schema import FieldResult, Status


def _flag(fr: FieldResult | None, issue: str, cap: float = 0.6):
    if fr is None or fr.status not in (Status.KNOWN, Status.NEEDS_REVIEW):
        return
    fr.issues.append(issue)
    fr.status = Status.NEEDS_REVIEW
    fr.confidence = min(fr.confidence, cap)


def _d(fr: FieldResult | None) -> date | None:
    if fr is None or fr.status not in (Status.KNOWN, Status.NEEDS_REVIEW) or not isinstance(fr.value, str):
        return None
    try:
        return date.fromisoformat(fr.value)
    except ValueError:
        return None


def _n(fr: FieldResult | None):
    if fr is None or fr.status not in (Status.KNOWN, Status.NEEDS_REVIEW):
        return None
    return fr.value if isinstance(fr.value, (int, float)) and not isinstance(fr.value, bool) else None


def check(page_type: str, fields: dict[str, FieldResult]) -> None:
    g = fields.get
    # the patient code links this visit to a woman's record: a misread digit would attach the
    # visit to someone else, so it is always confirmed by the midwife, however sure the OCR is
    for f in fields_for(page_type):
        if f.linking:
            _flag(g(f.key), "patient code: confirm before linking the visit", cap=0.79)

    if page_type == "cover":
        name, ftype = g("facility_name"), g("facility_type")
        if (name and ftype and ftype.status == Status.KNOWN and isinstance(name.value, str)
                and name.value.split(" ")[0] in ("DR", "CSC", "CSU", "CSCA", "CSUA")
                and name.value.split(" ")[0] != ftype.value):
            for k in ("facility_name", "facility_type"):
                _flag(g(k), "facility name and ticked facility type disagree")

    if page_type == "current_pregnancy":
        lmp, edd, post = _d(g("lmp_date")), _d(g("expected_delivery_date")), _d(g("term_exceeded_date"))
        if lmp and edd and abs((edd - lmp).days - 280) > 3:
            for k in ("lmp_date", "expected_delivery_date"):
                _flag(g(k), "due date should be last period + 280 days")
        if edd and post and abs((post - edd).days - 7) > 2:
            for k in ("expected_delivery_date", "term_exceeded_date"):
                _flag(g(k), "post-term date should be due date + 7 days")
        if lmp or edd:  # every visit/appointment date must fall within this pregnancy
            start = lmp.toordinal() if lmp else edd.toordinal() - 294
            end = (edd.toordinal() if edd else lmp.toordinal() + 280) + 30
            for ck, *_ in VISIT_COLS:
                for row in ("appointment_date", "attended_date"):
                    v = _d(g(f"visits.{ck}.{row}"))
                    if v and not start < v.toordinal() <= end:
                        _flag(g(f"visits.{ck}.{row}"), "date outside this pregnancy (last period → due date)")
        prev = None
        for ck, *_ in VISIT_COLS:
            seen = _d(g(f"visits.{ck}.attended_date"))
            ga = _n(g(f"visits.{ck}.gestational_age_weeks"))
            if seen and lmp:
                if ga is not None and abs((seen - lmp).days / 7 - ga) > 2.5:
                    _flag(g(f"visits.{ck}.gestational_age_weeks"),
                          f"does not match visit date ({(seen - lmp).days / 7:.0f} SA expected)")
            if seen and prev and seen < prev:
                _flag(g(f"visits.{ck}.attended_date"), "earlier than the previous visit")
            prev = seen or prev

    if page_type == "identification":
        gr, pa, lc = _n(g("gravidity")), _n(g("parity")), _n(g("living_children"))
        if gr is not None and pa is not None and pa > gr:
            for k in ("gravidity", "parity"):
                _flag(g(k), "parity cannot exceed gravidity")
        if pa is not None and lc is not None and lc > pa + 2:
            for k in ("parity", "living_children"):
                _flag(g(k), "more living children than deliveries")

    if page_type == "delivery":
        ga, w = _n(g("gestational_age_weeks")), _n(g("birth_weight_g"))
        if ga is not None and w is not None and ga >= 37 and w < 1500:
            _flag(g("birth_weight_g"), "very low weight for a term birth")
