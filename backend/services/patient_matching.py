"""Find patient profiles that could match a code the midwife typed or the OCR read.

A single misread digit would attach a visit to another woman, so we never link
automatically: we show exact and near matches (one character different, or two swapped)
and the midwife chooses.
"""
from __future__ import annotations


def normalise_code(code: str) -> str:
    """Codes compare without spaces or case: ' m-1234 ' -> 'M-1234'."""
    return "".join(code.split()).upper()


def _distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (optimal string alignment): counts a swap of neighbours as 1."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def candidates(code: str, existing: list[str], max_distance: int = 1) -> list[dict]:
    """Exact match first, then near matches. [{code, distance}]"""
    target = normalise_code(code)
    out = []
    for c in existing:
        dist = _distance(target, normalise_code(c))
        if dist <= max_distance:
            out.append({"code": c, "distance": dist})
    return sorted(out, key=lambda x: (x["distance"], x["code"]))
