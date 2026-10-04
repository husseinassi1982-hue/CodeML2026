"""Turn text read from the page into typed values.

``parse(raw, field)`` returns a ``Parsed``: the typed value, a display string, whether
parsing succeeded, and any issues (used to downgrade the status to NEEDS_REVIEW).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date

from .catalog import Field
from .lexicon import ALIASES, CLOSED_VOCAB, vocab_for
from .pdf_layout import fold

DASHES = re.compile(r"^[\s\-–—_~.]+$")
UNKNOWN_MARKERS = {"?", "??", "inconnu", "inconnue", "nsp", "ne sait pas", "non connu", "unknown", "ns"}

# characters OCR commonly confuses with digits, applied only to numeric fields
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "i": "1", "|": "1",
                           "!": "1", "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "g": "9", "q": "9",
                           "G": "6", "b": "6", "T": "7",
                           # French handwritten "1" has a long up-stroke and reads as Λ / A / ^
                           "A": "1", "Λ": "1", "^": "1", "N": "1"})
NOT_DONE = {"nf", "n.f", "non fait", "non faite", "pas fait"}
YES_WORDS = {"oui", "+", "positif", "present", "presente", "recu", "recue", "pris", "prise", "fait", "faite"}
NO_WORDS = {"non", "0", "o", "ø", "neg", "nég", "negatif", "absent", "absente", "aucun", "aucune"}


@dataclass
class Parsed:
    value: object = None
    display: str | None = None
    ok: bool = False
    issues: list[str] = field(default_factory=list)
    marker: str | None = None  # "dash" | "unknown" when the paper says so explicitly
    snapped: float = 1.0  # similarity when the text was snapped to the lexicon (1 = exact)


def _num(s: str) -> float | None:
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    return float(m.group().replace(",", ".")) if m else None


def _digits(s: str) -> str:
    """Fix OCR letter/digit confusions after removing a trailing unit ('SA', 'cm', 'kg', 'g/dL'...)."""
    s = re.sub(r"(?i)(\s*(sa|cm|mm|kg|g/dl|g/l|°c|℃|°|c|jours?|j|g))+\s*$", "", s.strip())
    return s.translate(DIGIT_FIX)


def _leftover_letters(fixed: str) -> bool:
    """True if letters survive digit fixing: the text was not really a number."""
    return bool(re.search(r"[^\d\s.,/\-+]", fixed))


def _parse_date(s: str) -> tuple[str | None, str | None, bool]:
    """-> (ISO date, display, repaired) ; repaired = a slash read as "1"/"l"/"|" was fixed."""
    t = re.sub(r"\s+", "", s)
    repaired = False
    m0 = re.match(r"^(\d\d)[1lI|](\d\d)[1lI|/](\d{4})$", t) or re.match(r"^(\d\d)/(\d\d)[1lI|](\d{4})$", t)
    if m0:
        t, repaired = f"{m0.group(1)}/{m0.group(2)}/{m0.group(3)}", True
    t = _digits(t)
    m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$", t)
    if not m:
        m2 = re.match(r"^(\d{2})(\d{2})(\d{4})$", t)  # slashes lost by OCR
        if not m2:
            return None, None, False
        m, repaired = m2, True
    d, mo, y = (int(x) for x in m.groups())
    if y < 100:
        y += 2000
    if not 1990 <= y <= 2035:  # a misread year ("2251") is not a date
        return None, None, False
    try:
        dt = date(y, mo, d)
    except ValueError:
        return None, None, False
    return dt.isoformat(), f"{d:02d}/{mo:02d}/{y}", repaired


def snap(text: str, vocab: list[str], cutoff: float = 0.78) -> tuple[str, float]:
    """Map OCR text onto the closest lexicon entry if it is close enough."""
    if not vocab:
        return text, 1.0
    f = fold(text)
    folded = {fold(v): v for v in vocab}
    if f in folded:
        return ALIASES.get(folded[f], folded[f]), 1.0
    scored = sorted(((difflib.SequenceMatcher(None, f, k).ratio(), k) for k in folded), reverse=True)
    ratio, best = scored[0]
    if ratio < cutoff:
        return text, 1.0
    if (len(scored) > 1 and ratio - scored[1][0] < 0.05
            and ALIASES.get(folded[scored[1][1]], folded[scored[1][1]]) != ALIASES.get(folded[best], folded[best])):
        ratio *= 0.6  # equally close to two different terms: do not pretend to know which
    return ALIASES.get(folded[best], folded[best]), ratio


def parse(raw: str | None, f: Field) -> Parsed:
    if raw is None or not raw.strip():
        return Parsed(ok=False)
    s = re.sub(r"\s+", " ", raw).strip()
    if DASHES.match(s):
        return Parsed(value=None, display="—", ok=True, marker="dash")
    if fold(s).strip(" .") in UNKNOWN_MARKERS:
        return Parsed(value=None, display="?", ok=True, marker="unknown")
    if fold(s).strip(" .") in NOT_DONE:
        return Parsed(value=None, display="NF (non fait)", ok=True, marker="not_done")

    p = _parse_typed(s, f)
    if p.ok and f.range and isinstance(p.value, (int, float)):
        lo, hi = f.range
        if not lo <= p.value <= hi:
            p.issues.append(f"out of plausible range ({lo:g}–{hi:g})")
    return p


def _parse_typed(s: str, f: Field) -> Parsed:
    d = f.dtype
    if d in ("str", "code"):
        if f.key == "facility_name":  # starts with the facility type: DR / CSC / CSU / CSCA / CSUA
            head, _, rest = s.partition(" ")
            if rest and len(head) <= 5:
                t, r = snap(head, ["DR", "CSC", "CSU", "CSCA", "CSUA"], cutoff=0.6)
                if t != head:
                    return Parsed(f"{t} {rest}", f"{t} {rest}", ok=True, snapped=r)
        if d == "code":
            v = re.sub(r"\s+", "", s).upper()
            return Parsed(v, v, ok=True)
        vocab = vocab_for(f)
        if vocab and re.fullmatch(r"[\wÀ-ÿ]?[aA4Λ][sS5]\.?", s.strip()) and fold(s.strip()) != "ras":
            return Parsed("RAS", "RAS", ok=True, snapped=0.7)  # "7AS", "DAS", "YAS": handwritten RAS
        v, ratio = snap(s, vocab)
        v = re.sub(r"\bAI(?= [A-Z])", "Al", v)  # Arabic article "Al" read as capital-i "AI"
        issues = []
        if vocab and fold(v) not in {fold(x) for x in vocab}:
            letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", v)
            if f.key in CLOSED_VOCAB:
                issues.append("not a known value for this field")
            elif 0 < len(letters) <= 4 and not re.search(r"\d", v):
                issues.append("unrecognised short word")
        return Parsed(v, v, ok=True, snapped=ratio, issues=issues)

    if d == "date":
        s = re.sub(r"^\s*M(?=\s*[/.\-])", "11", s)  # "11" written as two French 1s reads as "M"
        iso, disp, repaired = _parse_date(s)
        if not iso:
            return Parsed(issues=["could not read as a date (dd/mm/yyyy)"])
        return Parsed(iso, disp, ok=True, snapped=0.9 if repaired else 1.0)

    if d == "date_or_year":
        iso, disp, _ = _parse_date(s)
        if iso:
            return Parsed(iso, disp, ok=True)
        m = re.fullmatch(r"(19|20)\d\d", _digits(s).replace(" ", ""))
        return Parsed(m.group(), m.group(), ok=True) if m else Parsed(issues=["could not read as a date or year"])

    if d == "bp":
        t = re.sub(r"\s+", "", _digits(s))
        if "/" in t:
            m = re.match(r"^(\d{1,3})/(\d{1,3})$", t)
        else:  # slash lost by OCR: 10474 -> 104/74, 11071 -> 110/71
            m = re.match(r"^(\d{3})(\d{2,3})$", t)
        if not m:
            return Parsed(issues=["could not read as systolic/diastolic"])
        sys_, dia = int(m.group(1)), int(m.group(2))
        issues = []
        if sys_ < 30 and dia < 20:  # written in cmHg, e.g. 12/8: common practice, converted
            sys_, dia = sys_ * 10, dia * 10
        if not (60 <= sys_ <= 250 and 30 <= dia <= 150 and sys_ > dia):
            issues.append("implausible blood pressure")
        return Parsed({"systolic": sys_, "diastolic": dia}, f"{sys_}/{dia}", ok=True, issues=issues)

    if d == "weeks":  # "16SA+3j" = 16 weeks 3 days
        m = re.match(r"^\s*([\dAΛ]{1,2})\s*(?:S\s*A|SA|5A|S)?\s*\+\s*([\dAΛ])\s*[jJ]?", s)
        if m:
            w, dd = (int(x.translate(DIGIT_FIX)) for x in m.groups())
            if 0 <= dd <= 6:
                v = round(w + dd / 7, 1)
                return Parsed(v, f"{w} SA + {dd} j", ok=True)

    if d in ("int", "weeks", "grams", "cm", "days"):
        if d == "grams":  # the unit "g" is often read as "9" or ".9": "3485 g" -> "3485.9"
            s = re.sub(r"^\s*(\d{3,4})\s*[.,]?\s*[9gqa]\s*$", r"\1", s)
        t = _digits(s)
        n = _num(t)
        if n is None or _leftover_letters(t):
            return Parsed(issues=[f"could not read a number ({d})"])
        if d == "int" and not re.fullmatch(r"\s*\d+\s*", t):
            return Parsed(int(n), str(int(n)), ok=True, issues=["unexpected characters around the number"])
        unit = {"weeks": " SA", "grams": " g", "cm": " cm", "days": " j"}.get(d, "")
        v = int(n) if n == int(n) else n
        return Parsed(v, f"{v}{unit}", ok=True)

    if d in ("float", "kg", "temp"):
        t = _digits(s)
        n = _num(t)
        if n is None or _leftover_letters(t):
            return Parsed(issues=[f"could not read a number ({d})"])
        if d == "temp" and n > 100:  # decimal point lost: 372 -> 37.2
            n = n / 10
        unit = {"kg": " kg", "temp": " °C"}.get(d, "")
        return Parsed(n, f"{n:g}{unit}", ok=True)

    if d == "posneg":
        g = fold(s).replace(" ", "")
        if re.fullmatch(r"([mn][eéiy][gsyq5][.,]?)+", g):  # handwritten "nég" often reads "mes", "mig"
            return Parsed("negative", "Négatif", ok=True, snapped=0.7)
        if g.startswith(("neg", "nég")) or g in ("-", "n"):
            return Parsed("negative", "Négatif", ok=True)
        if g.startswith(("pos", "+")) or g.endswith("+"):
            return Parsed("positive", "Positif", ok=True)
        v, ratio = snap(s, ["Neg", "Pos"], cutoff=0.6)
        if v in ("Neg", "Pos"):
            return Parsed("negative" if v == "Neg" else "positive", "Négatif" if v == "Neg" else "Positif",
                          ok=True, snapped=ratio)
        return Parsed(issues=["expected Neg/Pos"])

    if d == "yesno":
        g = fold(s).strip(" .,")
        if g in ("oui", "non"):
            return Parsed(g == "oui", g.capitalize(), ok=True)
        if g in YES_WORDS or g.startswith(("recu", "rec")):
            return Parsed(True, "Oui", ok=True, snapped=1.0 if g in YES_WORDS else 0.8)
        if difflib.SequenceMatcher(None, g, "recu").ratio() >= 0.7 or re.match(r"^[rpkh][eéa][cçs]", g):
            return Parsed(True, "Oui", ok=True, snapped=0.75)  # "Reçu" (iron received), misread
        if g in NO_WORDS or g.startswith("neg"):
            return Parsed(False, "Non", ok=True)
        if g[:1] in ("n", "o") and len(g) <= 4:  # handwritten "Non" often reads "Nou"/"Nan"
            v = "Non" if g[0] == "n" else "Oui"
            return Parsed(v == "Oui", v, ok=True, snapped=difflib.SequenceMatcher(None, g, fold(v)).ratio())
        v, ratio = snap(s, ["Oui", "Non"], cutoff=0.5)
        if v in ("Oui", "Non"):
            return Parsed(v == "Oui", v, ok=True, snapped=ratio)
        return Parsed(issues=["expected Oui/Non"])

    if d == "sex":
        g = fold(s).strip(" .")
        if g in ("f", "feminin", "fille"):
            return Parsed("F", "F", ok=True)
        if g in ("m", "masculin", "garcon", "h"):
            return Parsed("M", "M", ok=True)
        return Parsed(issues=["expected F/M"])

    return Parsed(s, s, ok=True)
