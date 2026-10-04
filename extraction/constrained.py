"""Field-constrained reading: use what a field can contain to read it.

The recogniser (PP-OCRv6) outputs, for every slice of the crop, a probability for every
character; its usual reading takes the most likely character slice by slice ("NII+" for a
blood pressure). Here we instead score every *plausible value* of the field under those same
probabilities (CTC forward algorithm) and keep the most likely one ("11/7"):

* the grammar of the field: a blood pressure is S/D in cmHg or mmHg, a date a real date, a
  gestational age "16SA+3j", a weight a number in its range, a clinical cell a known word;
* lookalike characters count for each other (a French "1" is often read "A" or "l", "0" as "O",
  "/" as "1"), letters ignore case and accents;
* a weak prior from the organisers' synthetic data (extraction/data/field_priors.json): words
  seen in that column, the usual range of its numbers. It only breaks near-ties.

Nothing is invented when the ink does not fit: `fit` (how well the best value explains the
crop, compared with the free reading) and `margin` (best vs second best) feed the confidence,
so a crop whose ink matches no plausible value keeps a low score and goes to review.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import lexicon

PRIORS_PATH = Path(__file__).resolve().parent / "data" / "field_priors.json"
PRIOR_WEIGHT = 0.5  # log-prior scale: a tie-breaker, never stronger than a clear stroke
FEW_OPTIONS = 100  # fields with at most this many values (words, yes/no, +/-) ...
SMALL_WEIGHT = 2.0  # ... also listen to the small recogniser, which reads short words more steadily
# (on the real-photo fields: few-option fields 10/22 exact with the medium model alone, 12/22 with
#  medium + small; numbers and dates were best with the medium model alone, 6/22 vs 5/22)
TABLES = ("visits.", "previous_delivery.", "obstetric_anomalies.", "family_history.", "woman_history.")

# characters the recogniser confuses for each other in handwriting (target -> what may be read)
ALIASES = {
    "1": "1lI|i!AΛ^7", "0": "0OoDQ", "2": "2Zz", "3": "3", "4": "4", "5": "5Ss", "6": "6Gb",
    "7": "7T1", "8": "8B", "9": "9gq", "/": "/|lI1()\\", "+": "+tT", ",": ",.", ".": ".,", "-": "-—–_~",
}
WORD_COLUMNS = {  # free-text table cells that in practice hold a few clinical words
    "skeletal_anomalies", "conjunctivae", "breast_exam", "edema", "fetal_movements", "cervix",
    "presentation", "pelvis", "speculum_exam", "iron", "syphilis", "hiv", "hepatitis_b", "rai",
    "glycosuria", "albuminuria", "rubella", "toxoplasmosis", "reminder_visit",
}
EXTRA_WORDS = ["RAS", "NF", "Oui", "Non", "+", "-", "0", "nég", "neg", "Neg", "pos", "Reçu", "Fait",
               "colorées", "Colorées", "Décolorées", "Fermé", "Ouvert", "Céphalique", "Siège", "Normal",
               "Normaux", "Normales"]


def fold(c: str) -> str:
    return unicodedata.normalize("NFD", c.lower())[:1]


def column(key: str) -> str:
    return key.split(".")[-1] if key.startswith(TABLES) else key


@lru_cache(maxsize=1)
def priors() -> dict:
    try:
        return json.loads(PRIORS_PATH.read_text(encoding="utf-8"))
    except OSError:
        return {"words": {}, "numbers": {}}


# --- candidates -------------------------------------------------------------------------------


def _num_prior(col: str, values: list[float], i: int = 0) -> float:
    p = priors()["numbers"].get(col)
    if not p or i >= len(p["mean"]):
        return 0.0
    mu, sd = p["mean"][i], max(p["sd"][i], 1e-3) * 1.5  # widened: other clinics, other women
    return float(-0.5 * ((values[i] - mu) / sd) ** 2)


def candidates(key: str, dtype: str, rng=None) -> list[tuple[str, str, float]] | None:
    """(text as written, normalised value, log-prior) of every plausible value of this field,
    or None when the field is free text with no useful constraint."""
    return _candidates(column(key), dtype, tuple(rng) if rng else None)


@lru_cache(maxsize=64)
def _candidates(col: str, dtype: str, rng) -> list[tuple[str, str, float]] | None:
    out: dict[str, tuple[str, float]] = {}

    def add(text, value, prior=0.0):
        if text and (text not in out or out[text][1] < prior):
            out[text] = (value, prior)

    if dtype == "bp":
        for s in range(7, 23):
            for d in range(4, 14):
                if d < s:
                    pr = _num_prior(col, [s, d], 0) + _num_prior(col, [s, d], 1)
                    add(f"{s}/{d}", f"{s}/{d}", pr)
                    add(f"{s * 10}/{d * 10}", f"{s}/{d}", pr)
        for s in range(80, 200):
            for d in range(40, 130):
                if d + 10 <= s and (s % 10 or d % 10):
                    add(f"{s}/{d}", f"{s / 10:g}/{d / 10:g}",
                        _num_prior(col, [s / 10, d / 10], 0) + _num_prior(col, [s / 10, d / 10], 1))
    elif dtype == "date":
        # a date in the registry is recent: the two years before capture are the usual case,
        # older/later years stay possible at a cost (the ink must clearly say so)
        today = dt.date.today()
        day, end = dt.date(today.year - 4, 1, 1), dt.date(today.year + 1, 12, 31)
        while day <= end:
            d, m, y = day.day, day.month, day.year
            age = (today - day).days
            pr = 0.0 if 0 <= age <= 730 else -3.0
            iso = day.isoformat()
            for t in {f"{d}/{m}/{y % 100:02}", f"{d:02}/{m:02}/{y % 100:02}", f"{d}/{m}/{y}", f"{d:02}/{m:02}/{y}",
                      f"{d:02}/{m}/{y % 100:02}", f"{d}/{m:02}/{y % 100:02}"}:
                add(t, iso, pr)
            day += dt.timedelta(days=1)
    elif dtype == "weeks":
        lo, hi = (int(rng[0]), int(rng[1])) if rng else (4, 44)
        for w in range(lo, hi + 1):
            pr = _num_prior(col, [w])
            add(f"{w}SA", f"{w}", pr)
            add(f"{w}", f"{w}", pr - 1.0)
            for d in range(0, 7):
                v = f"{w}+{d}"
                for t in (f"{w}SA+{d}j", f"{w}SA+{d}", f"{w}SA{d}j"):
                    add(t, v, pr)
    elif dtype in ("kg", "int", "cm") or (dtype == "float" and rng):
        lo, hi = rng if rng else ((30, 160) if dtype == "kg" else (0, 60))
        units = {"kg": ["", "kg"], "cm": ["", "cm"]}.get(dtype, [""])
        if col.endswith("_cm"):
            units = ["", "cm"]
        if col == "age":
            units = ["", "ans", "jours", "j"]
        if dtype == "float":
            two = hi <= 5
            for a in range(int(lo), int(math.ceil(hi)) + 1):
                for b in range(100 if two else 10):
                    v = a + b / (100 if two else 10)
                    if lo <= v <= hi:
                        for sep in (",", "."):
                            frac = f"{b:02}" if two else f"{b}"
                            add(f"{a}{sep}{frac}", f"{v:g}", _num_prior(col, [v]))
                            if two and b % 10 == 0:
                                add(f"{a}{sep}{b // 10}", f"{v:g}", _num_prior(col, [v]))
                add(f"{a}", f"{a}", _num_prior(col, [a]) - 0.5)
        else:
            for n in range(int(lo), int(hi) + 1):
                for u in units:
                    add(f"{n}{u}", f"{n}", _num_prior(col, [n]))
                if dtype == "kg":
                    for k in range(10):
                        add(f"{n},{k}", f"{n}.{k}", _num_prior(col, [n + k / 10]))
                        add(f"{n}.{k}", f"{n}.{k}", _num_prior(col, [n + k / 10]))
        if dtype in ("kg", "cm") or col.endswith(("_kg", "_cm")):
            add("NF", "NF")
    elif col == "platelets":
        for n in range(80, 600):
            for t in (f"{n}000", f"{n}000/mm3", f"{n}k"):
                add(t, f"{n}000")
    elif dtype in ("yesno", "posneg", "enum") or col in WORD_COLUMNS:
        # (region / province are left to normalize.py's fuzzy snapping: midwives abbreviate them)
        vocab = list(dict.fromkeys(lexicon.CLINICAL + EXTRA_WORDS + list(priors()["words"].get(col, {}))))
        counts = priors()["words"].get(col, {})
        total = sum(counts.values()) + len(vocab)
        for w in vocab:
            add(w, w, math.log((counts.get(w, 0) + 1) / total))
    else:
        return None
    return [(t, v, p) for t, (v, p) in out.items()]


def _key(text: str) -> str:
    return "".join(fold(c) for c in text if not c.isspace())


@lru_cache(maxsize=64)
def _plausible_set(col: str, dtype: str, rng) -> frozenset:
    return frozenset(_key(t) for t, _, _ in (_candidates(col, dtype, rng) or []))


def is_plausible(key: str, dtype: str, rng, text: str) -> bool | None:
    """Is this reading one of the field's plausible values? None when the field has no grammar."""
    c = _candidates(column(key), dtype, tuple(rng) if rng else None)
    if not c:
        return None
    return _key(text or "") in _plausible_set(column(key), dtype, tuple(rng) if rng else None)


# --- CTC scoring --------------------------------------------------------------------------------


class Emissions:
    """log P(symbol | slice) for the symbols the candidates use, lookalikes merged."""

    def __init__(self, probs: np.ndarray, charset: list[str]):
        self.p, self.charset = probs, charset
        self.T = probs.shape[0]
        self._index: dict[str, int] = {}
        self._cols: list[np.ndarray] = []
        blank = probs[:, 0] + (probs[:, charset.index(" ")] if " " in charset else 0)
        self._cols.append(np.log(np.clip(blank, 1e-12, 1)))

    @staticmethod
    @lru_cache(maxsize=4)
    def _groups(charset_id: int, charset: tuple) -> dict[str, list[int]]:
        g: dict[str, list[int]] = {}
        for i, ch in enumerate(charset):
            if i and len(ch) == 1:
                g.setdefault(fold(ch), []).append(i)
        return g

    def sym(self, c: str) -> int:
        if c not in self._index:
            groups = self._groups(id(self.charset), tuple(self.charset))
            if c in ALIASES:
                idx = sorted({i for a in ALIASES[c] for i in groups.get(fold(a), []) if self.charset[i] in ALIASES[c]
                              or (a.isalpha() and fold(self.charset[i]) == fold(a))})
            else:
                idx = groups.get(fold(c), [])
            col = self.p[:, idx].sum(1) if idx else np.zeros(self.T, np.float32)
            self._cols.append(np.log(np.clip(col, 1e-12, 1)))
            self._index[c] = len(self._cols) - 1
        return self._index[c]

    def matrix(self) -> np.ndarray:
        return np.stack(self._cols, 1)  # (T, K), column 0 = blank


def ctc_logprob(E: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """log P(label sequence | crop) for N label sequences of the same length L (CTC forward)."""
    N, L = labels.shape
    T = E.shape[0]
    S = 2 * L + 1
    ext = np.zeros((N, S), np.int64)
    ext[:, 1::2] = labels
    skip = np.zeros((N, S), bool)
    if L > 1:
        skip[:, 3::2] = labels[:, 1:] != labels[:, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return _forward(E, ext, skip, N, S, T)


def _forward(E, ext, skip, N, S, T):
    neg = -np.inf
    alpha = np.full((N, S), neg)
    alpha[:, 0] = E[0, 0]
    alpha[:, 1] = E[0, ext[:, 1]]
    for t in range(1, T):
        a1 = alpha
        a2 = np.concatenate([np.full((N, 1), neg), alpha[:, :-1]], 1)
        a3 = np.concatenate([np.full((N, 2), neg), alpha[:, :-2]], 1)
        a3 = np.where(skip, a3, neg)
        m = np.maximum(np.maximum(a1, a2), a3)
        m_safe = np.where(np.isfinite(m), m, 0.0)
        alpha = m_safe + np.log(np.exp(a1 - m_safe) + np.exp(a2 - m_safe) + np.exp(a3 - m_safe)) + E[t, ext]
        alpha = np.where(np.isfinite(m), alpha, neg)
    return np.logaddexp(alpha[:, -1], alpha[:, -2]) if S > 1 else alpha[:, -1]


@dataclass
class Reading:
    text: str        # as written (best candidate)
    value: str       # normalised value
    logp: float      # log P(text | crop)
    fit: float       # logp - log P(free reading): ~0 = the value explains the ink as well as anything
    margin: float    # posterior of the best value among all candidates (0..1)
    top: list        # [(text, posterior)] best 3


# words written diagonally across a whole section (short marks like "—" or "NF" are not: they would
# win on any unclear stroke simply because they are short)
SPAN_WORDS = ["RAS", "R.A.S", "Néant", "Normal", "Normale", "Normaux", "Rien", "Négatif", "Aucun", "Aucune"]


def read_words(probs: np.ndarray, charset: list[str], words: list[str],
               probs_small: np.ndarray | None = None) -> Reading | None:
    """The most likely of a few words (e.g. what is written diagonally across a section)."""
    cands = [(w, w, 0.0) for w in words]
    scores = _scores(probs, charset, cands)
    free = float(np.log(np.clip(probs.max(1), 1e-12, 1)).sum())
    scale = 1
    if probs_small is not None:
        small = _scores(probs_small, charset, cands)
        scores = np.where(np.isfinite(small), scores + SMALL_WEIGHT * small, scores)
        free += SMALL_WEIGHT * float(np.log(np.clip(probs_small.max(1), 1e-12, 1)).sum())
        scale = 1 + SMALL_WEIGHT
    if not np.isfinite(scores).any():
        return None
    order = np.argsort(-scores)
    post = np.exp(scores - scores[order[0]])
    post /= post.sum()
    b = order[0]
    return Reading(words[b], words[b], float(scores[b]), float(scores[b] - free) / scale, float(post[b]),
                   [(words[i], round(float(post[i]), 3)) for i in order[:3]])


def few_options(key: str, dtype: str, rng=None) -> bool:
    c = candidates(key, dtype, rng)
    return bool(c) and len(c) <= FEW_OPTIONS


def _scores(probs: np.ndarray, charset: list[str], cands) -> np.ndarray:
    em = Emissions(probs, charset)
    labels = [[em.sym(c) for c in t] for t, _, _ in cands]
    E = em.matrix()
    T = E.shape[0]
    scores = np.full(len(cands), -np.inf)
    by_len: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        if len(lab) <= T:
            by_len.setdefault(len(lab), []).append(i)
    for L, idx in by_len.items():
        scores[idx] = ctc_logprob(E, np.array([labels[i] for i in idx], np.int64))
    return scores


def read(probs: np.ndarray, charset: list[str], key: str, dtype: str, rng=None,
         probs_small: np.ndarray | None = None) -> Reading | None:
    """probs: the medium recogniser's; probs_small: the small one's, used on few-option fields."""
    cands = candidates(key, dtype, rng)
    if not cands:
        return None
    scores = _scores(probs, charset, cands)
    if probs_small is not None and len(cands) <= FEW_OPTIONS:
        small = _scores(probs_small, charset, cands)
        scores = np.where(np.isfinite(small), scores + SMALL_WEIGHT * small, scores)
        free_small = float(np.log(np.clip(probs_small.max(1), 1e-12, 1)).sum())
    else:
        free_small = 0.0
    if not np.isfinite(scores).any():
        return None
    total = scores + PRIOR_WEIGHT * np.array([p for _, _, p in cands])
    order = np.argsort(-total)
    post = np.exp(total - total[order[0]])
    post /= post.sum()
    best = order[0]
    free = float(np.log(np.clip(probs.max(1), 1e-12, 1)).sum())  # the greedy path's probability
    if probs_small is not None and len(cands) <= FEW_OPTIONS:
        free += SMALL_WEIGHT * free_small
    scale = 1 + SMALL_WEIGHT if probs_small is not None and len(cands) <= FEW_OPTIONS else 1  # fit per model
    return Reading(cands[best][0], cands[best][1], float(scores[best]), float(scores[best] - free) / scale,
                   float(post[best]), [(cands[i][0], round(float(post[i]), 3)) for i in order[:3]])


def canonical(r: Reading, dtype: str) -> str:
    """The value in the form normalize.parse reads best (the midwife's "25SA3j" -> "25SA+3j",
    "21ans" -> "21"); words and other values stay as written."""
    if dtype == "weeks":
        w, _, d = r.value.partition("+")
        return f"{w}SA+{d}j" if d else f"{w}SA"
    if dtype in ("int", "kg", "cm", "float") and r.value != "NF":
        return r.value
    return r.text


def confidence(r: Reading) -> float:
    """0..1. High only when the value is clearly the best plausible one (margin) and explains the
    ink about as well as the free reading did (fit). Thresholds chosen a priori, see RESULTS."""
    fit = 1.0 / (1.0 + math.exp(-(r.fit + 6.0) / 2.0))  # ~1 when fit > -2, ~0.5 at -6, ~0 below -12
    return float(r.margin * fit)
