"""Real booklet support: the pink Moroccan "Fiche de surveillance de la grossesse et du post
partum" as midwives actually use it (data/Paper Registry/1-*.jpg).

Its layout differs from the organisers' specimen pages (pages split differently, the visit
table spread over two facing pages, the right page without row labels), and every photo is
taken at a different angle with a curved spine. So instead of a fixed template, fields are
located on each photo from what is printed on it:

  * the photo is deskewed using the printed table lines;
  * printed labels are found by OCR (row labels, column headers, "Age :", "Groupage :"...);
  * table cells are the band between the grid lines around a row label and a column header;
  * pages without row labels use the row positions of their facing page;
  * checkboxes are found as small printed squares next to their label; circled options as a
    hand-drawn ring around printed text;
  * a word written diagonally across several rows ("RAS" for a whole section) is read once and
    reported for every row it crosses, for review.

Field keys are the catalog keys, so the output is the same PageExtraction as for any page.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from rapidfuzz import fuzz

from .catalog import get_field
from .imageproc import _ink, _remove_lines, analyze_text_crop, text_crop_for_ocr
from .ocr import OcrLine
from .pdf_layout import fold

# --- layout specifications -------------------------------------------------------------


@dataclass
class Inline:  # handwriting to the right of a printed label
    key: str
    anchor: str
    nth: int = 0
    stop: tuple[str, ...] = ()  # labels that end the field on the same line
    between: tuple[str, str] | None = None  # fallback: the line between these two labels


@dataclass
class Check:  # a printed square next to a label
    key: str
    anchor: str
    side: str = "right"
    nth: int = 0


@dataclass
class CheckRow:  # numbered boxes on one line ("VAT : 1 □ 2 □ ...") -> keys in x order
    keys: list[str]
    anchor: str


@dataclass
class Radio:  # one of several labelled squares
    key: str
    options: list[tuple[str, str]]  # (value, label)
    side: str = "right"


@dataclass
class Circled:  # options printed on one line; the chosen one is circled by hand
    key: str
    anchor: str  # label that starts the line ("Groupage")
    options: list[tuple[str, str]]  # (value, printed text)


@dataclass
class Table:
    prefix: str  # key = f"{prefix}.{col}.{row}" (or f"{prefix}.{col}.{row}" via row/col keys)
    rows: list[tuple[str, str]]  # (row key, row label); label "" = row given by position
    cols: list[tuple[str, str]]  # (col key, header label)
    key_fmt: str = "{prefix}.{col}.{row}"
    col_order_pattern: str | None = None  # headers matched by this pattern, assigned in x order
    row_fractions: list[float] | None = None  # rows without labels: centre as fraction of the table


@dataclass
class Column:  # a whole column of an unlabelled table (e.g. "Médicaux") read as one value
    key: str
    header: str


@dataclass
class Layout:
    name: str
    page_type: str
    signature: list[str]
    fields: list = field(default_factory=list)
    pii: list[str] = field(default_factory=list)  # labels whose handwriting is an identifier


VISIT_ROWS_BOOKLET = [
    ("appointment_date", "Rendez vous"), ("attended_date", "Venue le"),
    ("reminder_visit", "Visites de relance"), ("gestational_age_weeks", "Age probable de la grossesse"),
    ("weight_kg", "Poids"), ("blood_pressure", "TA"),
    ("skeletal_anomalies", "Anomalies du squelette (à préciser)"), ("conjunctivae", "Etat des conjonctives"),
    ("breast_exam", "Examen des seins"), ("edema", "Œdèmes"), ("fetal_movements", "Mouvements actifs"),
    ("fundal_height_cm", "HU"), ("fetal_heart_rate", "BCF"), ("speculum_exam", "Examen au speculum"),
    ("cervix", "Etat du col"), ("presentation", "Présentation"), ("pelvis", "Bassin"),
    ("glycosuria", "Glucosurie"), ("albuminuria", "Albuminurie"), ("rubella", "Rubéole"),
    ("toxoplasmosis", "Toxoplasmose"), ("syphilis", "Syphilis(TPHA/VDRL)"), ("hepatitis_b", "Ag HBs"),
    ("hiv", "Sérologie VIH"), ("hemoglobin_g_dl", "Hémoglobine"), ("platelets", "Plaquettes"),
    ("glycemia_g_l", "Bilan glycémique"), ("rai", "RAI (si Rhésus négatif)"), ("iron", "Fer"),
    ("examined_by", "EXAMEN FAIT PAR"),
]

LAYOUTS = [
    Layout(
        "booklet_cover", "cover",
        ["FICHE DE SURVEILLANCE DE LA GROSSESSE", "ET DU POST PARTUM", "Type de l'établissement sanitaire",
         "Mode de la couverture", "Nom/Prénom de la parturiente", "Grossesse classée à risque",
         "Si grossesse à risque, préciser le type de risque"],
        [
            Inline("record_number", "N° de la fiche"),
            Inline("region", "Région", stop=("Province",)),
            Inline("province", "Province"),
            Inline("facility_name", "Nom de l'établissement sanitaire",
                   between=("Région", "Type de l'établissement sanitaire")),
            Radio("facility_type", [("DR", "DR"), ("CSC", "CSC"), ("CSU", "CSU"), ("CSCA", "CSCA"), ("CSUA", "CSUA")]),
            Radio("coverage_mode", [("fixed", "Fixe"), ("mobile", "Mobile")]),
            Check("high_risk", "Grossesse classée à risque"),
            Check("risk.anemia", "- Anémie"), Check("risk.hypertension", "-H.T.A"),
            Check("risk.diabetes", "- Diabète"), Check("risk.cardiopathy", "- Cardiopathie"),
            Check("risk.metrorrhagia", "- Métrorragie"), Check("risk.infection", "- Infection"),
            Check("risk.preeclampsia", "- Pré-éclampsie"), Check("risk.eclampsia", "- Eclampsie"),
        ],
        pii=["Nom/Prénom de la parturiente"],
    ),
    Layout(
        "booklet_identification", "identification",
        ["IDENTIFICATION", "Niveau d'instruction", "Nom du Mari", "Consanguinité", "Grossesse désirée",
         "Mari et famille du mari", "ANTÉCÉDENTS DE LA FEMME", "Médicaux", "Chirurgicaux", "Gynécologiques"],
        [
            Inline("age", "Age"),
            Inline("education_level", "Niveau d'instruction"),
            Inline("profession", "Profession", nth=0),
            Inline("husband_profession", "Profession", nth=1),
            Check("consanguinity", "Consanguinité"),
            Check("desired_pregnancy", "Grossesse désirée"),
            Table("family_history",
                  [("hypertension", "HTA"), ("diabetes", "Diabète"), ("hereditary", "Maladies héréditaires"),
                   ("malformations", "Malformations"), ("allergies", "Allergie(s)")],
                  [("woman_family", "Famille de la femme"), ("husband_family", "Mari et famille du mari")]),
            Column("woman_history.medical.hypertension", "Médicaux"),
            Column("woman_history.surgical.hypertension", "Chirurgicaux"),
            Column("woman_history.gynecological.hypertension", "Gynécologiques"),
        ],
        pii=["CIN", "Adresse", "Téléphone", "Nom du Mari"],
    ),
    Layout(
        "booklet_obstetric_history", "identification",
        ["ANTÉCÉDENTS OBSTÉTRICAUX", "Anomalies du déroulement des grossesses antérieures",
         "DÉROULEMENT DES ACCOUCHEMENTS ANTÉRIEURS", "Accouchement 1", "Nombre d'enfants vivants",
         "Vaccinée contre la rubéole", "Frottis cervical"],
        [
            Table("obstetric_anomalies",
                  [("abortion", "Avortement"), ("premature_delivery", "prématuré"), ("fetal_death", "in utéro"),
                   ("other", "Autres à préciser")],
                  [("count", "Nombre"), ("date", "Date"), ("place", "Lieu"), ("gestational_age", "Age gestationnel (SA)")]),
            Table("previous_delivery",
                  [("date", "l'accouchement"), ("mode", "l'extraction"), ("cesarean_indication", "préciser l'indication"),
                   ("complication", "Si complication de"), ("newborn_weight_g", "Poids du (des)"),
                   ("newborn_complication", "Si complication du")],
                  [(str(i), f"Accouchement {i}") for i in range(1, 6)]),
            Inline("gravidity", "Gestation", stop=("Parité",)),
            Inline("parity", "Parité", stop=("Nombre d'enfants vivants",)),
            Inline("living_children", "Nombre d'enfants vivants"),
            CheckRow([f"tetanus_dose_{i}" for i in range(1, 6)], "VAT"),
            Check("rubella_vaccinated", "Vaccinée contre la rubéole"),
            Inline("rubella_vaccination_date", "Le", nth=0),
            Check("hepatitis_b_vaccinated", "Vaccinée contre l'hépatite B"),
            Inline("hepatitis_b_vaccination_date", "Le", nth=1),
            Inline("cervical_screening", "Frottis cervical / I V A (moins de 3 ans)"),
        ],
    ),
    Layout(
        "booklet_pregnancy_left", "current_pregnancy",
        ["GROSSESSE ACTUELLE", "Groupage", "Prestations", "Rendez vous", "Age probable de la grossesse",
         "EXAMEN CLINIQUE", "Toucher vaginal", "EXAMEN BIOLOGIQUE", "Syphilis(TPHA/VDRL)", "1er trimestre"],
        [
            Inline("lmp_date", "DDR|DOR|D.D.R"),
            Inline("height_cm", "Taille", stop=("Groupage",)),
            Circled("blood_group", "Groupage", [("A", "A"), ("B", "B"), ("O", "O"), ("AB", "AB")]),
            Circled("rhesus", "Groupage", [("negative", "Rhésus (-)"), ("positive", "Rhésus (+)")]),
            Table("visits", VISIT_ROWS_BOOKLET,
                  [("t1_v1", "Visite 1"), ("t1_v2", "Visite 2"), ("t1_v3", "Visite 3")]),
        ],
    ),
    Layout(
        "booklet_pregnancy_right", "current_pregnancy",
        ["DATE PREVUE D'ACCOUCHEMENT", "DATE DE DÉPACEMENT DE TERME", "2ème trimestre", "3ème trimestre",
         "7ème mois", "8ème mois", "9ème mois"],
        [
            Inline("expected_delivery_date", "DATE PREVUE D'ACCOUCHEMENT"),
            Inline("term_exceeded_date", "DATE DE DÉPACEMENT DE TERME"),
            Table("visits", [(k, "") for k, _ in VISIT_ROWS_BOOKLET],
                  [("t2_v1", "Visite 1"), ("t2_v2", "Visite 2"), ("t2_v3", "Visite 3"),
                   ("m7", "ème mois"), ("m8", "ème mois"), ("m9", "ème mois")],
                  col_order_pattern="eme mois"),
        ],
    ),
]
LAYOUT_BY_NAME = {l.name: l for l in LAYOUTS}

# Row centres of the visit table relative to the four shaded section bands (EXAMEN CLINIQUE,
# EXAMEN BIOLOGIQUE, TRAITEMENT, EXAMEN FAIT PAR): (k, f) means y = band[k] + f * (band[k+1] - band[k]).
# Measured on the left page, whose rows are labelled, by tools/booklet_rows.py; used on the right page.
ROW_ANCHORS: dict[str, tuple[int, float]] = {
    "appointment_date": (0, -0.2656), "attended_date": (0, -0.201), "reminder_visit": (0, -0.1399),
    "gestational_age_weeks": (0, -0.0696), "weight_kg": (0, 0.0835), "blood_pressure": (0, 0.1511),
    "skeletal_anomalies": (0, 0.2134), "conjunctivae": (0, 0.3021), "breast_exam": (0, 0.383),
    "edema": (0, 0.4577), "fetal_movements": (0, 0.5277), "fundal_height_cm": (0, 0.6153),
    "fetal_heart_rate": (0, 0.6859), "speculum_exam": (0, 0.753), "cervix": (0, 0.8071),
    "presentation": (0, 0.8782), "pelvis": (0, 0.9438), "glycosuria": (1, 0.0573), "albuminuria": (1, 0.1158),
    "rubella": (1, 0.1845), "toxoplasmosis": (1, 0.2508), "syphilis": (1, 0.3123), "hepatitis_b": (1, 0.3835),
    "hiv": (1, 0.4493), "hemoglobin_g_dl": (1, 0.5236), "platelets": (1, 0.5944), "glycemia_g_l": (1, 0.666),
    "rai": (1, 0.7447), "iron": (2, 0.1843), "examined_by": (2, 0.943),
}

# --- page analysis ---------------------------------------------------------------------


@dataclass
class Seg:  # a printed line: horizontal y = a*x + b over [lo, hi] (vertical: x = a*y + b)
    a: float
    b: float
    lo: float
    hi: float

    def at(self, t: float) -> float:
        return self.a * t + self.b


@dataclass
class Page:
    img: np.ndarray  # deskewed
    lines: list[OcrLine]  # OCR lines in deskewed coordinates
    ink: np.ndarray
    hl: list[Seg]
    vl: list[Seg]
    angle: float
    th: float = 24.0  # typical height of a printed text line (px)


def _rotate_lines(lines: list[OcrLine], M: np.ndarray) -> list[OcrLine]:
    out = []
    for l in lines:
        q = np.c_[l.quad, np.ones(4)] @ M.T
        out.append(OcrLine(l.text, l.score, q))
    return out


def _segments(mask: np.ndarray, horizontal: bool, min_len: int) -> list[Seg]:
    k = np.ones((1, 35), np.uint8) if horizontal else np.ones((35, 1), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    segs = []
    for i in range(1, n):
        L = stats[i, cv2.CC_STAT_WIDTH] if horizontal else stats[i, cv2.CC_STAT_HEIGHT]
        thick = stats[i, cv2.CC_STAT_HEIGHT] if horizontal else stats[i, cv2.CC_STAT_WIDTH]
        if L < min_len or thick > 14 or stats[i, cv2.CC_STAT_AREA] > 8 * L:
            continue  # page edge against a dark background, shaded band, not a printed line
        ys, xs = np.nonzero(lab == i)
        t, v = (xs, ys) if horizontal else (ys, xs)
        a, b = np.polyfit(t, v, 1)
        resid = float(np.sqrt(np.mean((v - (a * t + b)) ** 2)))
        if abs(a) > 0.1 or resid > 4.0:
            continue  # slanted or curved: a pen stroke, not a printed line
        segs.append(Seg(float(a), float(b), float(t.min()), float(t.max())))
    # merge collinear pieces of the same printed line
    segs.sort(key=lambda s: s.at((s.lo + s.hi) / 2))
    merged: list[Seg] = []
    for s in segs:
        for m_ in merged:
            gap = max(s.lo, m_.lo) - min(s.hi, m_.hi)  # < 0 when they overlap
            mid = (max(s.lo, m_.lo) + min(s.hi, m_.hi)) / 2
            if gap < 25 and abs(s.at(mid) - m_.at(mid)) < 4 and abs(s.a - m_.a) < 0.02:
                w1, w2 = m_.hi - m_.lo, s.hi - s.lo
                m_.a = (m_.a * w1 + s.a * w2) / (w1 + w2)
                m_.b = (m_.b * w1 + s.b * w2) / (w1 + w2)
                m_.lo, m_.hi = min(m_.lo, s.lo), max(m_.hi, s.hi)
                break
        else:
            merged.append(Seg(s.a, s.b, s.lo, s.hi))
    return merged


def analyse(img: np.ndarray, lines: list[OcrLine]) -> Page:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = _ink(gray)
    hs = _segments(ink, True, 120)
    angles = [math.degrees(math.atan(s.a)) for s in hs if s.hi - s.lo > 200]
    angle = float(np.median(angles)) if angles else 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rimg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rink = _ink(cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY))
    paper = cv2.GaussianBlur(cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY), (0, 0), 9) > 110
    hl = [sg for sg in _segments(rink, True, 60) if _on_paper(paper, sg, True)]
    vl = [sg for sg in _segments(rink, False, 60) if _on_paper(paper, sg, False)]
    rl = _rotate_lines(lines, M)
    hs_ = [l.quad[:, 1].max() - l.quad[:, 1].min() for l in rl if l.score >= 0.9 and len(l.text) >= 4]
    th = float(np.median(hs_)) if hs_ else 24.0
    return Page(rimg, rl, rink, hl, vl, angle, th)


def _on_paper(paper: np.ndarray, sg: Seg, horizontal: bool) -> bool:
    h, w = paper.shape
    pts = []
    for t in np.linspace(sg.lo, sg.hi, 7):
        x, y = (t, sg.at(t)) if horizontal else (sg.at(t), t)
        if 0 <= int(y) < h and 0 <= int(x) < w:
            pts.append(paper[int(y), int(x)])
    return bool(pts) and float(np.mean(pts)) > 0.7


# --- label finding ----------------------------------------------------------------------


@dataclass
class Found:
    x0: float
    y0: float
    x1: float  # estimated end of the label itself (the OCR line may continue with handwriting)
    y1: float
    line_x1: float
    text: str = ""  # the whole OCR line
    used: int = 0  # characters of the line taken by the label

    @property
    def remainder(self) -> str:
        """What the OCR read after the label on the same line (often the handwritten value)."""
        return self.text[self.used:].strip(" :;.") if self.text else ""

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2

    @property
    def h(self):
        return self.y1 - self.y0


def _match(line_fold: str, lab_fold: str) -> tuple[float, int]:
    """(score, number of characters of the line taken by the label)."""
    full = fuzz.ratio(line_fold, lab_fold)
    best, used = full, len(line_fold)
    if len(line_fold) > len(lab_fold) + 1:
        for n in (len(lab_fold) - 1, len(lab_fold), len(lab_fold) + 1, len(lab_fold) + 2):
            s = fuzz.ratio(line_fold[:n], lab_fold) * 0.98
            if s > best:
                best, used = s, n
    return best, used


def find_labels(page: Page, text: str, min_score: float = 78, x_max: float | None = None) -> list[Found]:
    """Occurrences of a printed label, top-to-bottom then left-to-right. When a near-exact match
    exists, weaker look-alikes ("Prestations" for "Présentation") are dropped."""
    scored = _find_scored(page, text, min_score)
    if x_max is not None:
        scored = [(f, s) for f, s in scored if f.xc < x_max]
    if not scored:
        return []
    top = max(s for _, s in scored)
    keep = [f for f, s in scored if s >= top - 5]
    keep.sort(key=lambda f: (round(f.yc / 12), f.x0))
    return keep


def _find_scored(page: Page, text: str, min_score: float) -> list[tuple[Found, float]]:
    out = []
    for alt in text.split("|"):
        out += _find_scored_one(page, alt, min_score)
    return out


def _find_scored_one(page: Page, text: str, min_score: float) -> list[tuple[Found, float]]:
    lf = fold(text)
    out = []
    for l in page.lines:
        tf = fold(l.text)
        if not tf:
            continue
        s, used = _match(tf, lf)
        if s < min_score and len(lf) >= 4:
            # label in the middle of a merged line ("Groupage:B0 AB Rhésus (-) Rhésus (+)")
            pr = fuzz.partial_ratio(lf, tf) if len(tf) > len(lf) else 0
            if pr >= 90:
                i = tf.find(lf[:4])
                if i >= 0:
                    x0, x1 = l.quad[:, 0].min(), l.quad[:, 0].max()
                    cw = (x1 - x0) / max(1, len(tf))
                    y0, y1 = l.quad[:, 1].min(), l.quad[:, 1].max()
                    out.append((Found(x0 + i * cw, y1 - page.th, x0 + (i + len(lf)) * cw, y1, x1), pr * 0.85))
            continue
        if s < min_score:
            continue
        x0, x1 = l.quad[:, 0].min(), l.quad[:, 0].max()
        y0, y1 = l.quad[:, 1].min(), l.quad[:, 1].max()
        lx1 = x0 + (x1 - x0) * min(1.0, used / max(1, len(tf)))
        if used < len(tf) and y1 - y0 > 1.4 * page.th:
            y0 = y1 - page.th  # merged with taller handwriting: the printed label sits on the baseline
        out.append((Found(x0, y0, lx1, y1, x1, text=l.text, used=used), s))
    return out


def find_label(page: Page, text: str, nth: int = 0, min_score: float = 78, x_max: float | None = None) -> Found | None:
    if nth == 0:
        fs = find_labels(page, text, min_score, x_max)
    else:  # repeated label: every decent match, in reading order
        sc = [(f, s) for f, s in _find_scored(page, text, max(min_score, 82)) if x_max is None or f.xc < x_max]
        fs = sorted((f for f, _ in sc), key=lambda f: (round(f.yc / 12), f.x0))
    return fs[nth] if nth < len(fs) else None


# --- geometry helpers ------------------------------------------------------------------


def _row_band(page: Page, y: float, x: float, x_probe: float | None = None):
    """Printed horizontal lines just above and below point (x, y)."""
    above = [s for s in page.hl if s.lo - 30 <= x <= s.hi + 30 and s.at(x) < y - 2]
    below = [s for s in page.hl if s.lo - 30 <= x <= s.hi + 30 and s.at(x) > y + 2]
    a = max(above, key=lambda s: s.at(x)) if above else None
    b = min(below, key=lambda s: s.at(x)) if below else None
    return a, b


def _col_band(page: Page, x: float, y: float):
    left = [s for s in page.vl if s.lo - 40 <= y <= s.hi + 40 and s.at(y) < x - 2]
    right = [s for s in page.vl if s.lo - 40 <= y <= s.hi + 40 and s.at(y) > x + 2]
    l = max(left, key=lambda s: s.at(y)) if left else None
    r = min(right, key=lambda s: s.at(y)) if right else None
    return l, r


@dataclass
class Region:
    key: str
    rect: tuple[int, int, int, int]  # x0, y0, x1, y1 in deskewed pixels
    kind: str = "text"  # text | check | radio | circled
    extra: dict = field(default_factory=dict)


def _clip_rect(page: Page, x0, y0, x1, y1):
    h, w = page.img.shape[:2]
    return int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))


def _inline_region(page: Page, spec: Inline) -> Region | None:
    lab = find_label(page, spec.anchor, spec.nth)
    if spec.between and (lab is None or not _is_between(page, lab, spec.between)):
        a, b = (find_label(page, t) for t in spec.between)
        if a is None or b is None:
            return None
        # the whole line (printed label + handwriting) between the two neighbours; the label
        # text is stripped from the reading later
        yc = (a.yc + b.yc) / 2
        return Region(spec.key, _clip_rect(page, a.x0 - 5, yc - 1.3 * page.th, page.img.shape[1] - 5, yc + 1.0 * page.th),
                      "text", {"strip_label": spec.anchor})
    if lab is None:
        return None
    th = page.th
    x_start = lab.x1 + 2
    x_stop = page.img.shape[1] - 5
    for st in spec.stop:
        for f in find_labels(page, st):
            if abs(f.yc - lab.yc) < lab.h * 1.2 and f.x0 > lab.x1:
                x_stop = min(x_stop, f.x0 - 3)
    # a printed vertical line (table cell border) ends the field too
    for s in page.vl:
        xv = s.at(lab.yc)
        if s.lo < lab.yc - 0.3 * th and s.hi > lab.yc + 0.3 * th and lab.x1 + 5 < xv < x_stop:
            x_stop = xv - 2
    # handwriting is usually larger than print and sits a little above the line
    y0, y1 = lab.yc - 1.4 * th, lab.yc + 0.9 * th
    xm = (x_start + x_stop) / 2
    a, b = _row_band(page, lab.yc, xm)
    if a is not None and lab.yc - a.at(xm) < 2.2 * th:  # inside a table cell
        y0 = max(y0, a.at(xm) + 2)
    if b is not None and b.at(xm) - lab.yc < 2.2 * th:
        y1 = min(y1, b.at(xm) - 1)
    return Region(spec.key, _clip_rect(page, x_start, y0, x_stop, y1), "text", {"line_reading": lab.remainder})


def _sensitive_ink(page: Page, rect) -> np.ndarray:
    """Thin or faint strokes (box borders, hand-drawn rings): a more sensitive darkness threshold."""
    x0, y0, x1, y1 = rect
    gray = cv2.cvtColor(page.img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(cv2.dilate(gray.astype(np.uint8), np.ones((15, 15), np.uint8)), (0, 0), 5).astype(np.float32)
    return (gray / np.maximum(bg, 1) < 0.86).astype(np.uint8)


def _is_between(page: Page, lab: Found, labels: tuple[str, str]) -> bool:
    a, b = (find_label(page, t) for t in labels)
    return a is not None and b is not None and a.yc < lab.yc < b.yc


def _square_near(page: Page, lab: Found, side: str, max_dx: float = 140) -> tuple[int, int, int, int] | None:
    """A small printed square on the label's line, nearest on the given side."""
    th = page.th
    y0, y1 = int(lab.yc - 1.3 * th), int(lab.yc + 1.3 * th)
    if side == "right":
        x0, x1 = int(lab.x1 + 1), int(lab.x1 + max_dx)
    else:
        x0, x1 = int(lab.x0 - max_dx), int(lab.x0 - 1)
    x0, y0, x1, y1 = _clip_rect(page, x0, y0, x1, y1)
    if y1 <= y0 or x1 <= x0:
        return None
    roi = cv2.morphologyEx(_sensitive_ink(page, (x0, y0, x1, y1)), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    cnts, hier = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    best, best_d = None, None
    lo, hi = 0.4 * th, 1.4 * th
    for c in cnts:
        x, y, w, hh = cv2.boundingRect(c)
        if not (lo <= w <= hi and lo <= hh <= hi and 0.6 <= w / hh <= 1.6):
            continue
        if cv2.contourArea(c) < 0.8 * w * hh:  # round letters (o, e, d...) are not square
            continue
        approx = cv2.approxPolyDP(c, 0.08 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        cx = x0 + x + w / 2
        d = abs(cx - (lab.x1 if side == "right" else lab.x0))
        if best_d is None or d < best_d:
            best, best_d = (x0 + x, y0 + y, x0 + x + w, y0 + y + hh), d
    if best is None:
        # a ticked box: the cross/tick runs over the border, so look for a compact mark instead
        n, _, st, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        for i in range(1, n):
            x, y, w, hh, area = (st[i, j] for j in range(5))
            if lo <= w <= 1.8 * th and lo <= hh <= 1.8 * th and area > 0.25 * w * hh:
                cx = x0 + x + w / 2
                d = abs(cx - (lab.x1 if side == "right" else lab.x0))
                if d < 0.9 * max_dx and (best_d is None or d < best_d):
                    best, best_d = (x0 + x, y0 + y, x0 + x + w, y0 + y + hh), d
    return best


def _tick_score(page: Page, box) -> float:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    inset = max(2, int(0.2 * min(w, h)))
    roi = page.ink[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
    return float(roi.mean()) if roi.size else 0.0


# --- extraction ------------------------------------------------------------------------


@dataclass
class Located:
    regions: list[Region]
    spans: list[tuple[list[str], tuple[int, int, int, int]]]  # (keys crossed, bbox) of diagonal writing
    pii: list[tuple[int, int, int, int]]
    warnings: list[str]


def locate(page: Page, layout: Layout) -> Located:
    regions: list[Region] = []
    warnings: list[str] = []
    spans: list = []
    for spec in layout.fields:
        if isinstance(spec, Inline):
            r = _inline_region(page, spec)
            if r:
                regions.append(r)
            else:
                warnings.append(f"label not found: {spec.anchor}")
        elif isinstance(spec, Check):
            lab = find_label(page, spec.anchor, spec.nth)
            box = _square_near(page, lab, spec.side, max_dx=170) if lab else None
            regions.append(Region(spec.key, box or (0, 0, 0, 0), "check", {"found": box is not None}))
        elif isinstance(spec, CheckRow):
            lab = find_label(page, spec.anchor)
            boxes = []
            if lab:
                x = lab.x1
                for _ in range(len(spec.keys)):
                    probe = Found(x, lab.y0, x, lab.y1, x)
                    b = _square_near(page, probe, "right", max_dx=110)
                    if not b:
                        break
                    boxes.append(b)
                    x = b[2] + 2
            for i, k in enumerate(spec.keys):
                b = boxes[i] if i < len(boxes) else None
                regions.append(Region(k, b or (0, 0, 0, 0), "check", {"found": b is not None}))
        elif isinstance(spec, Radio):
            opts = {}
            for value, label in spec.options:
                lab = find_label(page, label, min_score=90 if len(label) <= 4 else 78)
                opts[value] = _square_near(page, lab, spec.side, max_dx=80) if lab else None
            regions.append(Region(spec.key, (0, 0, 0, 0), "radio", {"options": opts}))
        elif isinstance(spec, Circled):
            lab = find_label(page, spec.anchor)
            regions.append(Region(spec.key, (0, 0, 0, 0), "circled", {"anchor": lab, "options": spec.options}))
        elif isinstance(spec, Column):
            regions += _column_region(page, spec, warnings)
        elif isinstance(spec, Table):
            regs, sp = _table_regions(page, spec, warnings)
            regions += regs
            spans += sp
    pii = []
    for lab_text in layout.pii:
        lab = find_label(page, lab_text)
        if lab:
            r = _inline_region(page, Inline("_pii", lab_text))
            if r:
                pii.append(r.rect)
    return Located(regions, spans, pii, warnings)


def _column_region(page: Page, spec: Column, warnings) -> list[Region]:
    hdr = find_label(page, spec.header)
    if hdr is None:
        warnings.append(f"header not found: {spec.header}")
        return []
    l, r = _col_band(page, hdr.xc, hdr.yc + 3 * hdr.h)
    x0 = l.at(hdr.yc + 3 * hdr.h) + 3 if l else hdr.x0 - 20
    x1 = r.at(hdr.yc + 3 * hdr.h) - 3 if r else hdr.x1 + 20
    _, below = _row_band(page, hdr.yc, hdr.xc)
    y0 = below.at(hdr.xc) + 3 if below else hdr.y1 + 5
    lows = [s.at(hdr.xc) for s in page.hl if s.lo - 20 <= hdr.xc <= s.hi + 20 and s.at(hdr.xc) > y0]
    y1 = max(lows) - 3 if lows else y0 + 300
    return [Region(spec.key, _clip_rect(page, x0, y0, x1, y1))]


def _table_regions(page: Page, spec: Table, warnings):
    # columns
    cols: list[tuple[str, Found]] = []
    if spec.col_order_pattern:
        fixed = [(k, lab) for k, lab in spec.cols if lab != "ème mois"]
        for k, lab in fixed:
            f = find_label(page, lab)
            if f:
                cols.append((k, f))
        ordered = sorted([f for l in page.lines for f in [_found_of(l)]
                          if spec.col_order_pattern in fold(l.text).replace("ê", "e")], key=lambda f: f.x0)
        month_keys = [k for k, lab in spec.cols if lab == "ème mois"]
        cols += list(zip(month_keys, ordered))
    else:
        for k, lab in spec.cols:
            f = find_label(page, lab, min_score=85 if lab.startswith(("Visite", "Accouchement")) else 78)
            if f:
                cols.append((k, f))
            else:
                warnings.append(f"column header not found: {lab}")
    if not cols:
        return [], []
    hdr_y = float(np.median([f.yc for _, f in cols]))

    # rows
    rows: list[tuple[str, float, float, float]] = []  # key, y_center, x_label, label_h
    if all(lab == "" for _, lab in spec.rows):
        rows = _rows_by_position(page, spec, cols, warnings)
    else:
        x_lim = min(f.x0 for _, f in cols)
        for k, lab in spec.rows:
            fs = [f for f in find_labels(page, lab, 80 if len(lab) > 3 else 95, x_max=x_lim) if f.yc > hdr_y + 5]
            if not fs:
                continue
            f = fs[0]
            rows.append((k, f.yc, f.xc, f.h))
    regions, spans = [], []
    cell_rects: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    cols = sorted(cols, key=lambda c: c[1].xc)
    cx = [f.xc for _, f in cols]
    rows = sorted(rows, key=lambda r: r[1])
    for j, (rk, ry, rx, rh) in enumerate(rows):
        for i, (ck, cf) in enumerate(cols):
            # column edges: halfway to the neighbouring headers, snapped to a printed line if close
            w = (cx[i + 1] - cx[i]) if i + 1 < len(cx) else (cx[i] - cx[i - 1]) if i > 0 else 2 * (cf.x1 - cf.x0)
            wl = (cx[i] - cx[i - 1]) if i > 0 else w
            xl, xr = cx[i] - wl / 2, cx[i] + w / 2
            a_, b_ = _row_band(page, ry, rx)
            slope = a_.a if a_ else (b_.a if b_ else 0.0)
            y_at = ry + slope * (cx[i] - rx)
            xl = _snap_v(page, xl, y_at, 0.3 * wl) + 3
            xr = _snap_v(page, xr, y_at, 0.3 * w) - 3
            # row edges: halfway to the neighbouring rows, snapped likewise
            up = (ry - rows[j - 1][1]) if j > 0 else (rows[j + 1][1] - ry if j + 1 < len(rows) else 2 * rh)
            dn = (rows[j + 1][1] - ry) if j + 1 < len(rows) else up
            up, dn = min(up, 2.5 * rh), min(dn, 2.5 * rh)
            yt = _snap_h(page, y_at - up / 2, cx[i], 0.35 * up) + 2
            yb = _snap_h(page, y_at + dn / 2, cx[i], 0.35 * dn) - 1
            key = spec.key_fmt.format(prefix=spec.prefix, col=ck, row=rk)
            rect = _clip_rect(page, xl, yt, xr, yb)
            cell_rects[(ck, rk)] = rect
            regions.append(Region(key, rect, "cell", {"col": ck, "row": rk}))
    spans = _find_spans(page, spec, cols, cell_rects)
    return regions, spans


def _snap_h(page: Page, y: float, x: float, tol: float) -> float:
    cands = [sg.at(x) for sg in page.hl if sg.lo - 30 <= x <= sg.hi + 30 and abs(sg.at(x) - y) <= tol]
    return min(cands, key=lambda v: abs(v - y)) if cands else y


def _snap_v(page: Page, x: float, y: float, tol: float) -> float:
    cands = [sg.at(y) for sg in page.vl if sg.lo - 40 <= y <= sg.hi + 40 and abs(sg.at(y) - x) <= tol]
    return min(cands, key=lambda v: abs(v - x)) if cands else x


def _found_of(l: OcrLine) -> Found:
    x0, x1 = l.quad[:, 0].min(), l.quad[:, 0].max()
    y0, y1 = l.quad[:, 1].min(), l.quad[:, 1].max()
    return Found(x0, y0, x1, y1, x1)


def bands(page: Page) -> list[float]:
    """y centres of the wide shaded section bands of a table, top to bottom."""
    g = cv2.cvtColor(page.img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(cv2.dilate(g.astype(np.uint8), np.ones((61, 61), np.uint8)), (0, 0), 20).astype(np.float32)
    norm = g / np.maximum(bg, 1)
    shade = ((norm > 0.45) & (norm < 0.88)).astype(np.uint8)
    shade = cv2.morphologyEx(shade, cv2.MORPH_OPEN, np.ones((9, 61), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(shade)
    ys = sorted(st[i, 1] + st[i, 3] / 2 for i in range(1, n) if st[i, 2] > 250 and 8 < st[i, 3] < 60)
    merged: list[float] = []
    for y in ys:
        if merged and y - merged[-1] < 40:
            merged[-1] = (merged[-1] + y) / 2
        else:
            merged.append(float(y))
    return merged


def _rows_by_position(page: Page, spec: Table, cols, warnings):
    """Right page: no row labels. Rows sit at the same place relative to the shaded section
    bands as on the facing (left) page."""
    b = [y for y in bands(page) if y > min(f.yc for _, f in cols)]
    if len(b) != 4 or not ROW_ANCHORS:
        warnings.append(f"could not place the unlabelled rows ({len(b)} section bands found, 4 expected)")
        return []
    xc = float(np.median([f.xc for _, f in cols]))
    rh = (b[1] - b[0]) / 14
    rows = []
    for k, _ in spec.rows:
        if k in ROW_ANCHORS:
            i, f = ROW_ANCHORS[k]
            y = b[i] + f * (b[i + 1] - b[i]) if i + 1 < len(b) else b[i]
            rows.append((k, y, xc, rh))
    return rows


def _line_mask(page: Page) -> np.ndarray:
    """Every detected printed line, drawn thick (cached on the page)."""
    if getattr(page, "_lines_cache", None) is None:
        m = np.zeros(page.ink.shape, np.uint8)
        for sg in page.hl:
            cv2.line(m, (int(sg.lo), int(sg.at(sg.lo))), (int(sg.hi), int(sg.at(sg.hi))), 1, 6)
        for sg in page.vl:
            cv2.line(m, (int(sg.at(sg.lo)), int(sg.lo)), (int(sg.at(sg.hi)), int(sg.hi)), 1, 6)
        page._lines_cache = m
    return page._lines_cache


def _near_line_mask(page: Page) -> np.ndarray:
    if getattr(page, "_near_cache", None) is None:
        page._near_cache = cv2.dilate(_line_mask(page), np.ones((9, 9), np.uint8))
    return page._near_cache


def _find_spans(page: Page, spec: Table, cols, cell_rects):
    """Handwriting written diagonally across several rows of one column."""
    spans = []
    for ck, _ in cols:
        rects = [(rk, r) for (c, rk), r in cell_rects.items() if c == ck and r[3] > r[1]]
        if len(rects) < 2:
            continue
        x0 = min(r[0] for _, r in rects)
        x1 = max(r[2] for _, r in rects)
        y0 = min(r[1] for _, r in rects)
        y1 = max(r[3] for _, r in rects)
        row_h = float(np.median([r[3] - r[1] for _, r in rects]))
        crop = page.ink[y0:y1, x0:x1].copy() & (1 - _line_mask(page)[y0:y1, x0:x1])
        crop, _ = _remove_lines(crop, h_len=int(0.6 * (x1 - x0)), v_len=int(3 * row_h))
        crop = cv2.dilate(crop, np.ones((7, 7), np.uint8))  # join the separate letters of a slanted word
        n, lab, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = (stats[i, j] for j in range(5))
            if h < 1.6 * row_h or area < 60 or w < 15:
                continue
            m = cv2.moments((lab == i).astype(np.uint8))
            ang = abs(math.degrees(0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])))
            if not 20 <= ang <= 75:
                continue  # written along a row or a stack of separate values, not diagonal
            gx0, gy0, gx1, gy1 = x0 + x, y0 + y, x0 + x + w, y0 + y + h
            crossed = [rk for rk, r in rects if min(r[3], gy1) - max(r[1], gy0) > 0.35 * (r[3] - r[1])]
            if len(crossed) >= 2:
                keys = [spec.key_fmt.format(prefix=spec.prefix, col=ck, row=rk) for rk in crossed]
                spans.append((keys, (gx0, gy0, gx1, gy1)))
    return spans


# --- classification --------------------------------------------------------------------


def classify(lines: list[OcrLine]) -> tuple[Layout | None, float]:
    folded = [fold(l.text) for l in lines]
    best, best_s = None, 0.0
    for lay in LAYOUTS:
        hits = 0
        for sig in lay.signature:
            sf = fold(sig)
            if any(_match(t, sf)[0] >= 80 or (len(sf) >= 6 and len(t) > len(sf) and fuzz.partial_ratio(sf, t) >= 90)
                   for t in folded):
                hits += 1
        s = hits / len(lay.signature)
        if s > best_s:
            best, best_s = lay, s
    return (best, best_s) if best_s >= 0.6 else (None, best_s)


# --- reading ---------------------------------------------------------------------------

SEPARATORS = re.compile(r"^[\s/_|.:;,\-–]+|[\s/_|.:;,]+$")


def _clean_reading(text: str, strip_label: str | None = None) -> str:
    t = text or ""
    if strip_label:  # whole-line crop: drop the printed label at the start
        lf, tf = fold(strip_label), fold(t)
        best_n, best_s = 0, 0.0
        for n in range(max(1, len(lf) - 6), min(len(tf), len(lf) + 6) + 1):
            sc = fuzz.ratio(tf[:n], lf)
            if sc > best_s:
                best_n, best_s = n, sc
        if best_s >= 60:
            t = t[best_n:]
    t = SEPARATORS.sub("", t)  # printed "/__/__/" separators and stray punctuation
    return re.sub(r"\s+", " ", t).strip()


def _same(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(a, str) and isinstance(b, str):
        return re.sub(r"\W", "", fold(a)) == re.sub(r"\W", "", fold(b))
    return a == b


@dataclass
class Reading:
    text: str
    score: float
    source: str


def _choose(field_, readings: list[Reading]):
    """Pick a reading and a confidence from several OCR opinions. Agreement between independent
    readings is the strongest evidence we have on real handwriting."""
    from .normalize import parse

    cands = []
    for r in readings:
        if not r.text:
            continue
        p = parse(r.text, field_)
        cands.append((r, p))
    if not cands:
        return None, None, 0.0, 0
    best, best_key = None, None
    for r, p in cands:
        agree = sum(1 for r2, p2 in cands if r2 is not r and (
            (p.ok and p2.ok and p.marker == p2.marker and (p.marker or _same(p.value, p2.value)))
            or _same(r.text, r2.text)))
        key = (agree, p.ok and not p.issues, p.snapped, r.score)
        if best_key is None or key > best_key:
            best, best_key = (r, p), key
    r, p = best
    agree = best_key[0]
    base = max(rr.score for rr, pp in cands if rr is r or _same(rr.text, r.text) or (pp.ok and _same(pp.value, p.value)))
    if agree >= 1:
        conf = 0.55 + 0.45 * base
    else:
        conf = 0.75 * r.score
    if p.snapped < 1.0:
        conf *= 0.6 + 0.4 * p.snapped
    return r, p, conf, agree


def extract(img: np.ndarray, lines: list[OcrLine], layout: Layout, threshold: float):
    """-> (fields dict, warnings, pii rectangles, page). FieldResult objects use catalog keys."""
    from . import ocr
    from .backends.common import field_result
    from .schema import Status

    page = analyse(img, lines)
    loc = locate(page, layout)
    scale = max(1.0, page.th / 11.0)  # pixels per printed point, roughly
    fields = {}
    span_of: dict[str, int] = {}
    for si, (keys, _) in enumerate(loc.spans):
        for k in keys:
            span_of.setdefault(k, si)

    # remove the pixels of diagonal writing from the cells it crosses
    span_mask = np.zeros(page.ink.shape, np.uint8)
    for _, (x0, y0, x1, y1) in loc.spans:
        span_mask[y0:y1, x0:x1] = page.ink[y0:y1, x0:x1]
    span_mask = cv2.dilate(span_mask, np.ones((3, 3), np.uint8))

    text_regions = [r for r in loc.regions if r.kind in ("text", "cell") and r.rect[2] > r.rect[0] + 4
                    and r.rect[3] > r.rect[1] + 4]
    inks, crops = {}, {}
    free_ink = page.ink & (1 - _line_mask(page)) & (1 - span_mask)
    for r in text_regions:
        ink = _word_crop(page, free_ink, r.rect)
        inks[r.key] = ink
        if ink.present and not ink.is_dash:
            crops[r.key] = ink.clean
    keys = list(crops)
    small = dict(zip(keys, ocr.read_crops([crops[k] for k in keys], model="small")))
    # without the medium model there is no independent second reading: no agreement bonus
    medium = dict(zip(keys, ocr.read_crops([crops[k] for k in keys], model="medium"))) if ocr.medium_available() else {}
    span_reads = ocr.read_crops([_span_crop(page, b) for _, b in loc.spans], model="medium") if loc.spans else []

    for r in loc.regions:
        try:
            f = get_field(layout.page_type, r.key)
        except KeyError:
            continue
        if f.pii:
            continue
        if r.kind in ("text", "cell"):
            fields[f.key] = _decide_text_field(f, r, inks.get(r.key), small.get(r.key), medium.get(r.key),
                                               span_reads[span_of[r.key]] if r.key in span_of else None,
                                               threshold, field_result, Status)
        elif r.kind == "check":
            if not r.extra.get("found"):
                fields[f.key] = field_result(f, Status.NEEDS_REVIEW, 0.3, issues=["checkbox not found on the photo"])
                continue
            sc = _tick_score(page, r.rect)
            ticked = sc >= 0.12
            conf = min(1.0, 0.5 + abs(sc - 0.12) / 0.16)
            st = Status.KNOWN if conf >= 0.75 else Status.NEEDS_REVIEW
            fields[f.key] = field_result(f, st, conf, value=ticked, display="☒" if ticked else "☐",
                                         issues=[] if st == Status.KNOWN else ["unclear whether the box is ticked"])
        elif r.kind == "radio":
            opts = r.extra["options"]
            scores = {v: _tick_score(page, b) for v, b in opts.items() if b}
            if not scores:
                fields[f.key] = field_result(f, Status.NEEDS_REVIEW, 0.3, issues=["checkboxes not found on the photo"])
                continue
            ticked = [v for v, sc in scores.items() if sc >= 0.12]
            missing = [v for v, b in opts.items() if not b]
            if len(ticked) == 1:
                st = Status.KNOWN if not missing else Status.NEEDS_REVIEW
                fields[f.key] = field_result(f, st, 0.9 if not missing else 0.6, value=ticked[0], display=ticked[0],
                                             issues=[] if not missing else [f"boxes not found: {', '.join(missing)}"])
            elif not ticked:
                st = Status.NOT_PROVIDED if not missing else Status.NEEDS_REVIEW
                fields[f.key] = field_result(f, st, 0.85 if not missing else 0.5,
                                             issues=[] if not missing else [f"boxes not found: {', '.join(missing)}"])
            else:
                best = max(ticked, key=scores.get)
                fields[f.key] = field_result(f, Status.NEEDS_REVIEW, 0.5, value=best, display=best,
                                             issues=[f"several boxes ticked: {', '.join(ticked)}"])
        elif r.kind == "circled":
            fields[f.key] = _decide_circled(page, f, r, field_result, Status)
    return fields, loc.warnings, loc.pii, page


@dataclass
class WordInk:
    present: bool
    is_dash: bool
    fraction: float
    clean: np.ndarray | None  # crop of the whole words, other ink painted out


def _word_crop(page: Page, free_ink: np.ndarray, rect) -> WordInk:
    """Handwriting is often taller than the booklet's rows: take every handwritten word whose
    centre falls inside the field, whole, even where it spills over the row lines."""
    x0, y0, x1, y1 = rect
    th = page.th
    ext = int(0.9 * th)
    H, W = free_ink.shape
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0 - ext), min(W, x1), min(H, y1 + ext)
    sub = free_ink[sy0:sy1, sx0:sx1]
    if sub.size == 0:
        return WordInk(False, False, 0.0, None)
    words = cv2.dilate(sub, np.ones((max(3, int(0.25 * th)), max(5, int(0.35 * th))), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(words, connectivity=8)
    keep = []
    min_area = max(12, int(0.02 * th * th))
    near_line = _near_line_mask(page)[sy0:sy1, sx0:sx1]
    for i in range(1, n):
        cy = sy0 + cen[i][1]
        comp = (lab == i) & (sub > 0)
        real = int(comp.sum())
        if not (y0 <= cy <= y1 and real >= min_area):
            continue
        # a sliver of a printed row line the line mask did not quite cover is not handwriting
        if st[i, cv2.CC_STAT_HEIGHT] <= 0.45 * th and near_line[comp].mean() > 0.6:
            continue
        # long flat ink across most of the field, or hugging its top/bottom edge: a band edge or
        # a printed line, not a handwritten dash
        ch, cw = st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_WIDTH]
        if ch <= 0.45 * th and (cw > 0.6 * (x1 - x0) or not (y0 + 0.2 * (y1 - y0) <= cy <= y1 - 0.2 * (y1 - y0))):
            continue
        keep.append(i)
    if not keep:
        return WordInk(False, False, float(sub.mean()), None)
    mask = np.isin(lab, keep)
    ys, xs = np.nonzero(mask & (sub > 0))
    if len(xs) == 0:
        return WordInk(False, False, 0.0, None)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    w, h = bx1 - bx0, by1 - by0
    is_dash = len(keep) == 1 and h <= 0.35 * th and w >= 0.5 * th and w > 2.5 * h
    crop = page.img[sy0 + by0:sy0 + by1, sx0 + bx0:sx0 + bx1].copy()
    other = (page.ink[sy0 + by0:sy0 + by1, sx0 + bx0:sx0 + bx1] > 0) & ~mask[by0:by1, bx0:bx1]
    if other.any():
        crop[other] = np.median(crop.reshape(-1, 3), axis=0)
    pad = max(4, int(0.2 * th))
    crop = cv2.copyMakeBorder(crop, pad, pad, pad + 4, pad + 4, cv2.BORDER_REPLICATE)
    return WordInk(True, bool(is_dash), float(len(xs)) / max(1, sub.size), _for_recogniser(crop))


def _for_recogniser(crop: np.ndarray) -> np.ndarray:
    """Grey, divided by the local paper colour: removes the pink cast and shadows. On the hand-
    labelled real-handwriting crops this lifted character accuracy from 57% to 65% (medium model)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(cv2.dilate(g.astype(np.uint8), np.ones((15, 15), np.uint8)), (0, 0), 5).astype(np.float32)
    n = np.clip(g / np.maximum(bg, 1), 0, 1)
    return cv2.cvtColor((n * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _span_crop(page: Page, box) -> np.ndarray:
    """Diagonal writing, rotated to horizontal for the recogniser."""
    x0, y0, x1, y1 = box
    crop = page.img[y0:y1, x0:x1]
    m = page.ink[y0:y1, x0:x1].astype(np.uint8)
    mo = cv2.moments(m)
    ang = math.degrees(0.5 * math.atan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])) if mo["m00"] else 0.0
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rot = cv2.warpAffine(crop, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
    return _for_recogniser(cv2.copyMakeBorder(rot, 8, 8, 12, 12, cv2.BORDER_REPLICATE))


def _decide_text_field(f, r: Region, ink, small, medium, span_read, threshold, field_result, Status):
    strip_label = r.extra.get("strip_label")
    readings = []
    raw_small = small[0] if small else ""
    raw_medium = medium[0] if medium else ""
    if small:
        readings.append(Reading(_clean_reading(small[0], strip_label), small[1], "small"))
    if medium:
        readings.append(Reading(_clean_reading(medium[0], strip_label), medium[1], "medium"))
    if r.kind == "text" and (small or medium) and all(_separator_only(t) for t in (raw_small, raw_medium)):
        line = r.extra.get("line_reading") or ""
        if _separator_only(_clean_reading(line)) or not _clean_reading(line):
            return field_result(f, Status.NOT_PROVIDED, 0.6)
    line = r.extra.get("line_reading")
    if line:
        readings.append(Reading(_clean_reading(line), 0.8, "page-line"))

    if r.kind == "text" and ink is not None and ink.present \
            and all(_separator_only(rd.text) for rd in readings if rd.source != "page-line") \
            and not any(rd.text and not _separator_only(rd.text) for rd in readings if rd.source == "page-line"):
        return field_result(f, Status.NOT_PROVIDED, 0.6)  # only the printed "/__/" brackets were seen
    if (ink is None or not ink.present) and not any(rd.text for rd in readings if rd.source == "page-line"):
        if span_read is not None:
            return _span_result(f, span_read, field_result, Status)
        conf = 0.9 if ink is not None and ink.fraction == 0 else 0.7
        return field_result(f, Status.NOT_PROVIDED, conf)
    if ink is not None and ink.is_dash:
        return field_result(f, Status.NOT_APPLICABLE, 0.85, display="—")
    rd, p, conf, agree = _choose(f, readings)
    if rd is None:
        if span_read is not None:
            return _span_result(f, span_read, field_result, Status)
        return field_result(f, Status.ILLEGIBLE, 0.25, issues=["handwriting could not be read"])
    raw = rd.text
    if p.marker == "dash":
        return field_result(f, Status.NOT_APPLICABLE, 0.85, display="—", raw=raw)
    if p.marker == "not_done":
        return field_result(f, Status.NOT_APPLICABLE, min(0.9, conf), display=p.display, raw=raw)
    if p.marker == "unknown":
        return field_result(f, Status.UNKNOWN, conf, display="?", raw=raw)
    if not p.ok:
        return field_result(f, Status.NEEDS_REVIEW, min(conf, 0.45), raw=raw,
                            issues=list(p.issues) or ["could not interpret the handwriting"])
    issues = list(p.issues)
    if issues:
        return field_result(f, Status.NEEDS_REVIEW, min(conf, 0.5), value=p.value, display=p.display, raw=raw,
                            issues=issues)
    if conf < threshold:
        why = "the two OCR readings disagree" if agree == 0 else "low reading confidence"
        return field_result(f, Status.NEEDS_REVIEW, conf, value=p.value, display=p.display, raw=raw, issues=[why])
    return field_result(f, Status.KNOWN, conf, value=p.value, display=p.display, raw=raw)


def _separator_only(text: str) -> bool:
    """Only the printed "/__/" brackets: slashes, bars, underscores (read as 1, l, I...)."""
    t = re.sub(r"\s", "", text or "")
    if not t:
        return True
    return (len(t) <= 4 and bool(re.fullmatch(r"[/|1lI_\-.()\[\]]*", t))
            and (bool(re.search(r"[/|_]", t)) or len(t) >= 2))


def _span_result(f, span_read, field_result, Status):
    from .normalize import parse

    text, score = span_read
    p = parse(text, f) if text else None
    disp = p.display if p is not None and p.ok and p.display else (text or None)
    return field_result(f, Status.NEEDS_REVIEW, min(0.6, score), value=p.value if p is not None and p.ok else None,
                        display=disp, raw=text, issues=["written across several rows: confirm it applies to this row"])


def _option_centres(lab: Found, options: list[tuple[str, str]]) -> dict[str, float]:
    """x centre of each printed option on the anchor's OCR line (by character position)."""
    tf = fold(lab.text)
    x0, x1 = lab.x0, lab.line_x1
    cw = (x1 - x0) / max(1, len(tf))
    out, pos = {}, lab.used
    for v, txt in options:
        t = fold(txt)
        i = tf.find(t, pos)
        if i < 0 and len(t) > 3:
            i = tf.find(t[:6], pos)
        if i >= 0:
            out[v] = x0 + (i + len(t) / 2) * cw
            pos = i + 1
    return out


# Where each option of the "Groupage : A B O AB Rhésus (-) Rhésus (+)" line is printed, in printed-
# text heights after the end of "Groupage" (measured on the booklet photo 1-4.jpg).
CIRCLE_OFFSETS = {"A": 0.45, "B": 1.2, "O": 2.1, "AB": 3.5, "negative": 5.8, "positive": 9.3}


def _decide_circled(page: Page, f, r: Region, field_result, Status):
    """The midwife circles the printed option. Find hand-drawn rings on the line and read the
    printed text inside each one."""
    from . import ocr

    lab: Found | None = r.extra.get("anchor")
    options = r.extra["options"]
    if lab is None:
        return field_result(f, Status.NEEDS_REVIEW, 0.3, issues=["line not found on the photo"])
    th = page.th
    y0, y1 = int(lab.yc - 1.5 * th), int(lab.yc + 1.3 * th)
    x0, x1 = int(lab.x1 - 0.5 * th), int(page.img.shape[1] - 5)
    x0, y0, x1, y1 = _clip_rect(page, x0, y0, x1, y1)
    roi = cv2.morphologyEx(_sensitive_ink(page, (x0, y0, x1, y1)), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labm, st, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    crops, boxes = [], []
    for i in range(1, n):
        x, y, w, h, area = (st[i, j] for j in range(5))
        if w >= 0.9 * th and h >= 0.8 * th and area < 0.45 * w * h:  # an open ring, not a filled blob
            bx0, by0, bx1, by1 = x0 + x, y0 + y, x0 + x + w, y0 + y + h
            crops.append(_for_recogniser(cv2.copyMakeBorder(page.img[by0:by1, bx0:bx1], 6, 6, 8, 8,
                                                            cv2.BORDER_REPLICATE)))
            boxes.append((bx0, by0, bx1, by1))
    reads = ocr.read_crops(crops, model="medium") if crops else []
    chosen = set()
    values = {v for v, _ in options}
    for (text, score), (bx0, _, bx1, _) in zip(reads, boxes):
        t = fold(text).replace(" ", "")
        hit = None
        if "+" in t or "-" in t or "rh" in t or "sus" in t:  # a Rhésus option, "(+)" alone is enough
            for v, opt in options:
                o = fold(opt).replace(" ", "")
                if ("+" in t and "+" in o) or ("-" in t and "-" in o):
                    hit = v
        else:
            core = re.sub(r"[^a-z0]", "", t).replace("0", "o")
            for v, opt in options:
                if core and core == fold(opt):
                    hit = v
        if hit is None:  # unreadable inside the ring: use where it sits on the printed line
            off = ((bx0 + bx1) / 2 - lab.x1) / th
            v, d = min(CIRCLE_OFFSETS.items(), key=lambda kv: abs(kv[1] - off))
            if abs(d - off) < 0.6:
                hit = v
        if hit in values:
            chosen.add(hit)
    if not chosen:
        return field_result(f, Status.NOT_PROVIDED if not crops else Status.NEEDS_REVIEW, 0.5,
                            issues=[] if not crops else ["a circled option was seen but not recognised"])
    if len(chosen) > 1:
        return field_result(f, Status.NEEDS_REVIEW, 0.4, value=sorted(chosen)[0],
                            issues=[f"several options circled: {', '.join(sorted(chosen))}"])
    v = chosen.pop()
    return field_result(f, Status.NEEDS_REVIEW, 0.75, value=v, display=v,
                        issues=["circled option read from the photo: please confirm"])
