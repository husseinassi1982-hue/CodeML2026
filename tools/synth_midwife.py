"""Synthetic training crops in the midwife's own formats (no real data involved).

    python -m tools.synth_midwife --n 6000          # -> eval_out/vlm_dataset/synth_*.jsonl + eval_out/vlm_dataset_v2.zip

The specimen pages teach the model the registry, but their values are written the synthetic way
("115/63" mmHg, "31 SA", "Normales"). The real booklet uses other conventions: cmHg "11/7",
"16SA+3j", "NF", "Reçu", "nég", "colorées", comma decimals. This writes such values, drawn letter by
letter in free handwriting fonts (eval_out/fonts, SIL OFL, from github.com/google/fonts) with random
size, slant, wobble, ink colour, rows overflow and the French "1" (long up-stroke), onto EMPTY field
crops cut from the specimen pages (pink paper, printed lines, the right field label in the prompt).
Everything is synthetic: safe for Colab.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from extraction.catalog import PAGE_TYPES, fields_for
from tools.paths import REPO
from tools.vlm_common import prompt

DATA = REPO / "eval_out" / "vlm_dataset"
FONTS = REPO / "eval_out" / "fonts"
TABLES = ("visits.", "previous_delivery.", "obstetric_anomalies.", "family_history.", "woman_history.")


def column(key: str) -> str:
    return key.split(".")[-1] if key.startswith(TABLES) else key


# --- values, the midwife's way --------------------------------------------------------------------

def _date(r):
    d = dt.date(2024, 1, 1) + dt.timedelta(days=r.randrange(3 * 365))
    f = r.random()
    if f < 0.5:
        return f"{d.day:02}/{d.month:02}/{d.year % 100:02}"
    if f < 0.7:
        return f"{d.day}/{d.month}/{d.year % 100:02}"
    return f"{d.day:02}/{d.month:02}/{d.year}"


def _weeks(r):
    w, d = r.randint(6, 41), r.randint(0, 6)
    return r.choices([f"{w}SA+{d}j", f"{w}SA", f"{w} SA+{d}j", f"{w}SA+{d}", f"{w}SA {d}j"], [5, 2, 1, 1, 1])[0]


def _bp(r):
    s = r.randint(9, 14)
    d = r.randint(5, min(9, s - 2))
    return f"{s * 10}/{d * 10}" if r.random() < 0.15 else f"{s}/{d}"


def _kg(r):
    if r.random() < 0.07:
        return "NF"
    n = r.randint(45, 95)
    v = f"{n},5" if r.random() < 0.1 else str(n)
    return v + ("kg" if r.random() < 0.3 else "")


VALUES = {
    "blood_pressure": _bp,
    "gestational_age_weeks": _weeks,
    "weight_kg": _kg,
    "fundal_height_cm": lambda r: f"{r.randint(10, 40)}" + ("cm" if r.random() < 0.4 else ""),
    "fetal_heart_rate": lambda r: str(r.randint(110, 165)),
    "hemoglobin_g_dl": lambda r: f"{r.randint(9, 13)}{r.choice([',', ','] + ['.'])}{r.randint(0, 9)}",
    "glycemia_g_l": lambda r: f"0{r.choice([',', '.'])}{r.randint(65, 99)}" if r.random() < 0.8 else f"1,{r.randint(0, 25):02}",
    "platelets": lambda r: f"{r.randint(150, 400)}000" + ("/mm3" if r.random() < 0.5 else ""),
    "age": lambda r: f"{r.randint(16, 45)}" + (" ans" if r.random() < 0.4 else ""),
    "gravidity": lambda r: str(r.randint(1, 8)), "parity": lambda r: str(r.randint(0, 7)),
    "living_children": lambda r: str(r.randint(0, 7)),
}
YESNO = ["Oui", "Non", "+", "0", "Reçu", "oui", "non", "NF", "Oui", "Non"]
POSNEG = ["nég", "Neg", "négatif", "-", "0", "+", "Pos", "NF", "nég", "Négatif"]
CLINICAL = ["RAS", "RAS", "RAS", "colorées", "Colorées", "décolorées", "Normal", "Normaux", "Fermé", "Céph",
            "NF", "R.A.S", "ras", "Normales", "Pâles", "Rien"]
CLINICAL_BY_COL = {  # what each examination row usually holds
    "skeletal_anomalies": ["RAS"] * 6 + ["R.A.S", "ras", "Normal", "NF", "Rien"],
    "conjunctivae": ["colorées", "Colorées", "colorées", "décolorées", "Normales", "Pâles", "RAS", "NF"],
    "breast_exam": ["RAS"] * 5 + ["Normaux", "Normal", "NF", "ras"],
    "cervix": ["Fermé", "RAS", "Fermé", "long fermé", "NF", "ras"],
    "presentation": ["Céph", "Céphalique", "Siège", "RAS", "NF", "Pos"],
    "pelvis": ["RAS", "Normal", "NF", "ras"],
    "speculum_exam": ["RAS", "NF", "Normal", "ras"],
}
YESNO_COLS = {"iron", "edema", "fetal_movements", "reminder_visit"}
POSNEG_COLS = {"syphilis", "hiv", "hepatitis_b", "glycosuria", "albuminuria", "rai", "rubella", "toxoplasmosis"}
CLINICAL_COLS = {"skeletal_anomalies", "conjunctivae", "breast_exam", "cervix", "presentation", "pelvis", "speculum_exam"}


def value_for(col: str, key: str, r: random.Random) -> str | None:
    if col in VALUES:
        return VALUES[col](r)
    if col.endswith("_date") or col in ("appointment_date", "attended_date"):
        return _date(r)
    if col in YESNO_COLS:
        return r.choice(YESNO)
    if col in POSNEG_COLS:
        return r.choice(POSNEG)
    if col in CLINICAL_BY_COL:
        return r.choice(CLINICAL_BY_COL[col])
    if col in CLINICAL_COLS:
        return r.choice(CLINICAL)
    return None  # free text (names of places, notes...): left to the specimen data


@lru_cache(maxsize=1)
def field_info() -> dict[str, tuple[str, str, str]]:
    """column -> (label_fr, label_en, dtype) of one representative field (for borrowed backgrounds)."""
    out = {}
    for pt in PAGE_TYPES:
        for f in fields_for(pt):
            if f.kind in ("text", "cell") and not f.pii:
                out.setdefault(column(f.key), (f.label_fr, f.label_en, f.dtype))
    return out


# --- drawing ----------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=None)
def covers(path: str, ch: str) -> bool:
    f = font(path, 40)
    try:
        return f.getmask(ch).getbbox() is not None if ch.strip() else True
    except Exception:
        return False


def draw_text(text: str, height: int, r: random.Random) -> np.ndarray:
    """Alpha mask (uint8) of `text` written by hand: letter by letter, jitter, French 1."""
    fonts = [str(p) for p in sorted(FONTS.glob("*.ttf"))]
    ok = [p for p in fonts if all(covers(p, c) for c in text)]
    path = r.choice(ok or fonts)
    size = max(12, int(height * r.uniform(0.75, 1.1)))
    W, H = int(size * 0.9 * len(text) + 4 * size), int(size * 2)
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    x, base = size * 0.3, H * 0.25
    french_one = r.random() < 0.6
    stroke = r.choice([0, 0, 0, 1])
    for ch in text:
        s = max(10, int(size * r.uniform(0.9, 1.1)))
        f = font(path, s)
        y = base + r.uniform(-0.06, 0.06) * size
        if ch == "1" and french_one:  # the French 1: long up-stroke then the stem
            w = s * 0.35
            pts = [(x, y + s * 0.45), (x + w * 0.8, y + s * 0.12), (x + w * 0.9, y + s * 0.9)]
            d.line(pts, fill=255, width=max(2, s // 14) + stroke, joint="curve")
            x += w + s * r.uniform(0.05, 0.15)
            continue
        if ch == " ":
            x += s * r.uniform(0.25, 0.45)
            continue
        d.text((x, y), ch, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        x += f.getlength(ch) * r.uniform(0.85, 1.05)
    a = np.array(layer)
    ys, xs = np.nonzero(a)
    if not len(xs):
        return a[:1, :1]
    a = a[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    # slant + slight rotation + elastic wobble
    h, w = a.shape
    shear = r.uniform(-0.35, 0.15)
    M = np.float32([[1, shear, max(0, -shear * h)], [0, 1, 0]])
    a = cv2.warpAffine(a, M, (int(w + abs(shear) * h) + 2, h))
    a = cv2.warpAffine(a, cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), r.uniform(-4, 4), 1.0),
                       (a.shape[1], a.shape[0]))
    dx = cv2.GaussianBlur(np.random.default_rng(r.randrange(1 << 30)).uniform(-1, 1, a.shape).astype(np.float32), (0, 0), 6) * 25
    dy = cv2.GaussianBlur(np.random.default_rng(r.randrange(1 << 30)).uniform(-1, 1, a.shape).astype(np.float32), (0, 0), 6) * 25
    gx, gy = np.meshgrid(np.arange(a.shape[1], dtype=np.float32), np.arange(a.shape[0], dtype=np.float32))
    return cv2.remap(a, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0)


INKS = [(160, 60, 30), (150, 70, 60), (140, 50, 90), (120, 40, 110), (60, 50, 45), (170, 90, 40)]  # BGR ballpoint


def compose(bg: np.ndarray, text: str, r: random.Random) -> np.ndarray:
    out = bg.copy().astype(np.float32)
    H, W = out.shape[:2]
    row_h = H / (1 + 2 * 0.6 * 0.9)  # crop = field + margins (tools/vlm_common.MARGIN_Y)
    a = draw_text(text, int(row_h * r.uniform(0.7, 1.15)), r).astype(np.float32) / 255
    if a.shape[1] > W * 0.95:  # too long for the field: shrink to fit
        s = W * 0.95 / a.shape[1]
        a = cv2.resize(a, (max(1, int(a.shape[1] * s)), max(1, int(a.shape[0] * s))))
    if a.shape[0] > H:
        a = cv2.resize(a, (max(1, int(a.shape[1] * H / a.shape[0])), H))
    h, w = a.shape
    x = int(r.uniform(0.02, max(0.03, 1 - w / W - 0.02)) * W)
    y = int(H / 2 - h / 2 + r.uniform(-0.25, 0.25) * row_h)
    y = max(0, min(H - h, y))
    ink = np.array(r.choice(INKS), np.float32) * r.uniform(0.8, 1.2)
    strength = r.uniform(0.65, 0.95)
    region = out[y:y + h, x:x + w]
    al = (a * strength)[..., None]
    out[y:y + h, x:x + w] = region * (1 - al) + ink * al
    out = np.clip(out, 0, 255).astype(np.uint8)
    if r.random() < 0.6:
        out = cv2.GaussianBlur(out, (0, 0), r.uniform(0.3, 0.9))
    if r.random() < 0.5:
        out = np.clip(out + np.random.default_rng(r.randrange(1 << 30)).normal(0, r.uniform(2, 6), out.shape), 0, 255).astype(np.uint8)
    q = r.randint(40, 90)
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


# --- dataset -----------------------------------------------------------------------------------------

def backgrounds(split: str) -> list[dict]:
    rows = [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    return [r for r in rows if r["answer"] == "EMPTY" and r["variant"] in ("clean", "mild", "medium")]


def make(split: str, n: int, r: random.Random) -> list[dict]:
    bgs = backgrounds(split)
    by_col: dict[str, list[dict]] = {}
    for b in bgs:
        by_col.setdefault(column(b["key"]), []).append(b)
    visit_bgs = [b for b in bgs if b["key"].startswith("visits.")]
    info = field_info()
    targets = [c for c in info if value_for(c, c, random.Random(0)) is not None]
    out_rows = []
    (DATA / f"synth_{split}").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        col = r.choice(targets)
        pool = by_col.get(col) or visit_bgs or bgs
        bg = r.choice(pool)
        text = value_for(col, bg["key"], r)
        img = cv2.imread(str(DATA / bg["image"]))
        crop = compose(img, text, r)
        label_fr, label_en, dtype = info[col]
        name = f"synth_{split}/{i:05d}__{col}.jpg"
        cv2.imwrite(str(DATA / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out_rows.append(dict(image=name, split=f"synth_{split}", key=f"synth.{col}", dtype=dtype, page=bg["page"],
                             variant="synth_midwife", prompt=prompt(label_fr, label_en, dtype), answer=text))
    with open(DATA / f"synth_{split}.jsonl", "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="synthetic training crops")
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not list(FONTS.glob("*.ttf")):
        raise SystemExit(f"no fonts in {FONTS}")
    r = random.Random(a.seed)
    tr = make("train", a.n, r)
    va = make("val", a.n_val, r)
    print(f"synth_train: {len(tr)}  synth_val: {len(va)}")
    z = REPO / "eval_out" / "vlm_dataset_v2.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(DATA.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(DATA.parent))
    print(f"-> {z.relative_to(REPO)} ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
