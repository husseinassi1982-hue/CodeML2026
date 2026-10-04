"""Build extraction/data/field_priors.json: what each field usually contains, from the organisers'
synthetic data only (specimen ground truth + maternal_registry_synthetic.csv), never from the
real booklet photos. Used by extraction/constrained.py as a weak prior: words seen in a column,
and the typical range of its numbers.

    python -m tools.build_priors
"""

import csv
import json
import re
from collections import Counter, defaultdict

import numpy as np

from tools.paths import DATA, REPO

TABLES = ("visits.", "previous_delivery.", "obstetric_anomalies.", "family_history.", "woman_history.")
OUT = REPO / "extraction" / "data" / "field_priors.json"


def column(key: str) -> str:
    """Table cells share their column's prior: visits.m9.blood_pressure -> blood_pressure."""
    return key.split(".")[-1] if key.startswith(TABLES) else key


def numbers(col: str, raw: str) -> list[float]:
    raw = raw.replace(",", ".")
    if col == "blood_pressure":
        m = re.match(r"\s*(\d+)\s*/\s*(\d+)", raw)
        if not m:
            return []
        s, d = float(m.group(1)), float(m.group(2))
        if s > 30:  # mmHg -> cmHg, the unit the prior is kept in
            s, d = s / 10, d / 10
        return [s, d]
    m = re.search(r"\d+(\.\d+)?", raw)
    return [float(m.group())] if m else []


def main():
    gt = json.loads((REPO / "evaluation" / "ground_truth.json").read_text(encoding="utf-8"))
    words: dict[str, Counter] = defaultdict(Counter)
    nums: dict[str, list] = defaultdict(list)
    for page in gt["pages"].values():
        for key, f in page["fields"].items():
            raw = f.get("raw")
            if not isinstance(raw, str) or not raw.strip() or "�" in raw:  # NanumPen gaps
                continue
            col = column(key)
            n = numbers(col, raw)
            if n:
                nums[col].append(n)
            elif not re.search(r"\d", raw):
                words[col][raw.strip()] += 1

    # the synthetic CSV: more numbers for the same quantities (mmHg -> cmHg for blood pressure)
    rows = list(csv.DictReader(open(DATA / "maternal_registry_synthetic.csv", encoding="utf-8")))
    csv_cols = {"age (years)": "age", "gravidity (number)": "gravidity", "parity (number)": "parity",
                "living children (number)": "living_children", "hemoglobin (g/dl)": "hemoglobin_g_dl",
                "gestational age at enrollment (weeks)": "gestational_age_weeks"}
    for src, col in csv_cols.items():
        nums[col] += [[float(r[src])] for r in rows if r.get(src) not in (None, "")]
    nums["blood_pressure"] += [[float(r["mean systolic bp (mmhg)"]) / 10, float(r["mean diastolic bp (mmhg)"]) / 10]
                               for r in rows if r.get("mean systolic bp (mmhg)")]

    out = {"note": "Weak priors per field column, from the organisers' synthetic data only (specimen ground "
                   "truth + synthetic CSV), never from the real booklet photos. words = values written in "
                   "that column and how often; numbers = mean and sd of each number in it (blood pressure "
                   "in cmHg: [systolic, diastolic]). Built by tools/build_priors.py.",
           "words": {c: dict(v.most_common()) for c, v in sorted(words.items())},
           "numbers": {}}
    for c, vals in sorted(nums.items()):
        a = np.array(vals, float)
        out["numbers"][c] = {"n": int(len(a)), "mean": [round(float(x), 3) for x in a.mean(0)],
                             "sd": [round(float(x), 3) for x in a.std(0)]}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{OUT.relative_to(REPO)}: {len(out['words'])} word columns, {len(out['numbers'])} number columns")


if __name__ == "__main__":
    main()
