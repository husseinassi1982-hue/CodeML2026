"""Field catalog: every field we extract, per registry page type.

This is the schema the whole team agrees on. Each field has a stable key, French and
English labels, a data type (drives parsing and validation), and a description of where
it sits on the printed form (an "anchor" label printed on the page).

Geometry kinds
--------------
text   handwriting to the right of a printed label (region ends at the next printed item)
cell   handwriting in a table cell, located by a row label and a column header
check  a single checkbox next to a printed label (side = where the box is: left/right)
radio  several checkboxes of which at most one should be ticked; value = the ticked option

Fields with ``pii=True`` are direct identifiers (name, national ID, phone, address,
husband's name). They are listed only so the image redactor knows where they are:
their values are NEVER extracted, returned or stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- page types ---------------------------------------------------------------------

PAGE_TYPES = {
    "cover": "Fiche de surveillance (couverture)",
    "identification": "Identification et antécédents",
    "current_pregnancy": "Grossesse actuelle",
    "delivery": "Déroulement de l'accouchement",
    "postpartum_early_mother": "Post-partum précoce — mère",
    "postpartum_early_newborn": "Post-partum précoce — nouveau-né",
    "postpartum_late_mother": "Post-partum tardif — mère",
    "postpartum_late_newborn": "Post-partum tardif — nouveau-né",
}

# Order of the 8 pages inside one registry booklet (used for the specimen PDF).
PAGE_ORDER = list(PAGE_TYPES)


@dataclass
class Option:
    value: str
    anchor: str
    nth: int = 0
    side: str = "left"


@dataclass
class Field:
    key: str
    label_fr: str
    label_en: str
    kind: str  # text | cell | check | radio
    dtype: str = "str"
    anchor: str | None = None  # text / check
    nth: int = 0
    side: str = "left"  # check: where the box is relative to its label
    placement: str = "right"  # text: right of the label | "below" it | "after_box" (right of its checkbox)
    row: tuple[str, int] | None = None  # cell: (row label, nth)
    col: tuple[str, int] | None = None  # cell: (column header, nth)
    col_next: tuple[str, int] | None = None  # cell: header of the next column (band end)
    options: list[Option] = field(default_factory=list)  # radio
    pii: bool = False
    range: tuple[float, float] | None = None  # plausible numeric range
    section: str = ""
    linking: bool = False  # used to link visits to a patient (the midwife's code)

    @property
    def question_fr(self) -> str:
        if self.kind in ("check",):
            return f"La case « {self.label_fr} » est-elle cochée ?"
        if self.kind == "radio":
            return f"Quelle option est cochée pour « {self.label_fr} » ?"
        return f"Quelle est la valeur de « {self.label_fr} » ?"

    @property
    def question_en(self) -> str:
        if self.kind in ("check",):
            return f"Is the box “{self.label_en}” ticked?"
        if self.kind == "radio":
            return f"Which option is ticked for “{self.label_en}”?"
        return f"What is the value of “{self.label_en}”?"


# --- small helpers to keep the catalog readable -------------------------------------


def T(key, fr, en, anchor, dtype="str", nth=0, **kw):
    return Field(key, fr, en, "text", dtype, anchor=anchor, nth=nth, **kw)


def C(key, fr, en, anchor, nth=0, side="left", **kw):
    return Field(key, fr, en, "check", "bool", anchor=anchor, nth=nth, side=side, **kw)


def R(key, fr, en, options, **kw):
    return Field(key, fr, en, "radio", "enum", options=options, **kw)


def O(value, anchor, nth=0, side="left"):
    return Option(value, anchor, nth, side)


def table(prefix, section, rows, cols, row_band=None):
    """Expand a table into one ``cell`` field per (row, column).

    rows: list of (row_key, row_label_fr, row_label_en, anchor, nth, dtype, range)
    cols: list of (col_key, col_label_fr, col_label_en, header_anchor, nth[, dtype, range])
          a column dtype, when given, overrides the row dtype
    """
    out = []
    for ci, (ck, cfr, cen, canchor, cnth, *cdt) in enumerate(cols):
        nxt = (cols[ci + 1][3], cols[ci + 1][4]) if ci + 1 < len(cols) else None
        for rk, rfr, ren, ranchor, rnth, dtype, rng in rows:
            out.append(
                Field(
                    key=f"{prefix}.{ck}.{rk}",
                    label_fr=f"{rfr} — {cfr}",
                    label_en=f"{ren} — {cen}",
                    kind="cell",
                    dtype=cdt[0] if cdt else dtype,
                    row=(ranchor, rnth),
                    col=(canchor, cnth),
                    col_next=nxt,
                    range=cdt[1] if cdt else rng,
                    section=section,
                )
            )
    return out


# --- 1. cover -----------------------------------------------------------------------

COVER = [
    T("record_number", "N° de la fiche", "Record number", "N° de la fiche :", "code",
      section="identification", linking=True),
    T("region", "Région", "Region", "Région :", section="identification"),
    T("province", "Province", "Province", "Province :", section="identification"),
    T("facility_name", "Établissement sanitaire", "Health facility", "Nom de l'établissement sanitaire :",
      section="identification"),
    R("facility_type", "Type d'établissement", "Facility type",
      [O("DR", "DR", side="right"), O("CSC", "CSC", side="right"), O("CSU", "CSU", side="right"),
       O("CSCA", "CSCA", side="right"), O("CSUA", "CSUA", side="right")], section="identification"),
    R("coverage_mode", "Mode de couverture", "Coverage mode",
      [O("fixed", "Fixe"), O("mobile", "Mobile")], section="identification"),
    T("patient_name", "Nom/Prénom de la parturiente", "Patient name", "Nom/Prénom de la parturiente :",
      pii=True),
    C("high_risk", "Grossesse classée à risque", "High-risk pregnancy", "Grossesse classée à risque :",
      section="risk"),
    C("risk.anemia", "Risque : anémie", "Risk: anaemia", "Anémie", section="risk"),
    C("risk.hypertension", "Risque : HTA", "Risk: hypertension", "H.T.A", section="risk"),
    C("risk.diabetes", "Risque : diabète", "Risk: diabetes", "Diabète", section="risk"),
    C("risk.cardiopathy", "Risque : cardiopathie", "Risk: heart disease", "Cardiopathie", section="risk"),
    C("risk.metrorrhagia", "Risque : métrorragie", "Risk: metrorrhagia", "Métrorragie", section="risk"),
    C("risk.infection", "Risque : infection", "Risk: infection", "Infection", section="risk"),
    C("risk.preeclampsia", "Risque : pré-éclampsie", "Risk: pre-eclampsia", "Pré-éclampsie", section="risk"),
    C("risk.eclampsia", "Risque : éclampsie", "Risk: eclampsia", "Eclampsie", section="risk"),
    T("risk.other", "Autre risque", "Other risk", "Autres à préciser :", section="risk"),
]

# --- 2. identification and history ----------------------------------------------------

_FAMILY_ROWS = [
    ("hypertension", "HTA", "Hypertension", "HTA", 0, "str", None),
    ("diabetes", "Diabète", "Diabetes", "Diabète", 0, "str", None),
    ("hereditary", "Maladies héréditaires", "Hereditary diseases", "Maladies héréditaires", 0, "str", None),
    ("malformations", "Malformations", "Malformations", "Malformations", 0, "str", None),
    ("allergies", "Allergie(s)", "Allergies", "Allergie(s)", 0, "str", None),
]
_FAMILY_COLS = [
    ("woman_family", "famille de la femme", "woman's family", "Famille de la femme", 0),
    ("husband_family", "mari/famille", "husband/family", "Mari/famille", 0),
]
_WOMAN_HISTORY_COLS = [
    ("medical", "médicaux", "medical", "Médicaux", 0),
    ("surgical", "chirurgicaux", "surgical", "Chirurgicaux", 0),
    ("gynecological", "gynécologiques", "gynaecological", "Gynécologiques", 0),
]
_ANOMALY_ROWS = [
    ("abortion", "Avortement", "Abortion", "Avortement", 0, "str", None),
    ("premature_delivery", "Accouchement prématuré", "Premature delivery", "Accouchement prématuré", 0, "str", None),
    ("fetal_death", "Mort fœtale in utéro", "Intra-uterine fetal death", "Mort fœtale in utéro", 0, "str", None),
    ("other", "Autres", "Other", "Autres à préciser :", 0, "str", None),
]
_ANOMALY_COLS = [
    ("count", "nombre", "count", "Nombre", 0, "int", (0, 20)),
    ("date", "date", "date (or year)", "Date", 0, "date_or_year", None),
    ("place", "lieu", "place", "Lieu", 0, "str", None),
    ("gestational_age", "âge gestationnel (SA)", "gestational age (weeks)", "Age gestationnel (SA)", 0, "weeks",
     (4, 44)),
]
_PREV_DELIVERY_ROWS = [
    ("date", "Date", "Date", "Date", 1, "date", None),
    ("mode", "Modalité d'extraction", "Delivery mode", "Modalité d'extraction", 0, "str", None),
    ("cesarean_indication", "Indication de césarienne", "Caesarean indication", "Si césarienne : indication", 0,
     "str", None),
    ("complication", "Complication", "Complication", "Complication (type)", 0, "str", None),
    ("newborn_weight_g", "Poids nouveau-né", "Newborn weight (g)", "Poids nouveau-né(s)", 0, "grams", (300, 6500)),
    ("newborn_complication", "Complication nouveau-né", "Newborn complication", "Compl. nouveau-né (type)", 0,
     "str", None),
]
_PREV_DELIVERY_COLS = [(f"{i}", f"accouch. {i}", f"delivery {i}", f"Accouch. {i}", 0) for i in range(1, 6)]

IDENTIFICATION = (
    [
        T("age", "Âge", "Age (years)", "Age :", "int", range=(12, 55), section="profile"),
        T("national_id", "CIN", "National ID", "CIN :", pii=True),
        T("education_level", "Niveau d'instruction", "Education level", "Niveau d'instruction :", section="profile"),
        T("profession", "Profession", "Occupation", "Profession :", nth=0, section="profile"),
        T("address", "Adresse", "Address", "Adresse :", pii=True),
        T("phone", "Téléphone", "Phone", "Téléphone :", pii=True),
        T("husband_name", "Nom du mari", "Husband's name", "Nom du Mari :", pii=True),
        T("husband_profession", "Profession du mari", "Husband's occupation", "Profession :", nth=1,
          section="profile"),
        C("consanguinity", "Consanguinité", "Consanguinity", "Consanguinité", section="profile"),
        C("desired_pregnancy", "Grossesse désirée", "Desired pregnancy", "Grossesse désirée", section="profile"),
    ]
    + table("family_history", "family_history", _FAMILY_ROWS, _FAMILY_COLS)
    + table("woman_history", "medical_history", _FAMILY_ROWS[:1], _WOMAN_HISTORY_COLS)
    + table("obstetric_anomalies", "obstetric_history", _ANOMALY_ROWS, _ANOMALY_COLS)
    + table("previous_delivery", "obstetric_history", _PREV_DELIVERY_ROWS, _PREV_DELIVERY_COLS)
    + [
        T("gravidity", "Gestation", "Gravidity", "Gestation :", "int", range=(1, 20), section="obstetric_history"),
        T("parity", "Parité", "Parity", "Parité :", "int", range=(0, 20), section="obstetric_history"),
        T("living_children", "Enfants vivants", "Living children", "Nombre d'enfants vivants :", "int",
          range=(0, 20), section="obstetric_history"),
    ]
    + [C(f"tetanus_dose_{i}", f"VAT dose {i}", f"Tetanus vaccine dose {i}", str(i), section="vaccination")
       for i in range(1, 6)]
    + [
        C("rubella_vaccinated", "Vaccinée contre la rubéole", "Rubella vaccinated", "Vaccinée contre la rubéole",
          section="vaccination"),
        T("rubella_vaccination_date", "Date vaccin rubéole", "Rubella vaccination date", "Le", "date", nth=0,
          section="vaccination"),
        C("hepatitis_b_vaccinated", "Vaccinée contre l'hépatite B", "Hepatitis B vaccinated",
          "Vaccinée contre l'hépatite B", section="vaccination"),
        T("hepatitis_b_vaccination_date", "Date vaccin hépatite B", "Hepatitis B vaccination date", "Le", "date",
          nth=1, section="vaccination"),
        T("cervical_screening", "Frottis cervical / IVA", "Cervical screening", "Frottis cervical / IVA (moins de 3 ans) :",
          section="medical_history"),
    ]
)

# --- 3. current pregnancy (longitudinal visit table) ------------------------------------

VISIT_COLS = [
    ("t1_v1", "1er trim. visite 1", "1st trim. visit 1", "Visite 1", 0),
    ("t1_v2", "1er trim. visite 2", "1st trim. visit 2", "Visite 2", 0),
    ("t1_v3", "1er trim. visite 3", "1st trim. visit 3", "Visite 3", 0),
    ("t2_v1", "2e trim. visite 1", "2nd trim. visit 1", "Visite 1", 1),
    ("t2_v2", "2e trim. visite 2", "2nd trim. visit 2", "Visite 2", 1),
    ("t2_v3", "2e trim. visite 3", "2nd trim. visit 3", "Visite 3", 1),
    ("m7", "7e mois", "month 7", "7ème mois", 0),
    ("m8", "8e mois", "month 8", "8ème mois", 0),
    ("m9", "9e mois", "month 9", "9ème mois", 0),
]
VISIT_ROWS = [
    ("appointment_date", "Rendez-vous", "Appointment date", "Rendez-vous", 0, "date", None),
    ("attended_date", "Venue le", "Date attended", "Venue le", 0, "date", None),
    ("reminder_visit", "Visite de relance", "Reminder visit", "Visites de relance", 0, "yesno", None),
    ("gestational_age_weeks", "Âge probable (SA)", "Gestational age (weeks)", "Age probable", 0, "weeks", (4, 44)),
    ("weight_kg", "Poids (kg)", "Weight (kg)", "Poids (kg)", 0, "kg", (30, 160)),
    ("blood_pressure", "TA", "Blood pressure", "TA", 0, "bp", None),
    ("skeletal_anomalies", "Anomalies squelette", "Skeletal anomalies", "Anomalies squelette", 0, "str", None),
    ("conjunctivae", "État des conjonctives", "Conjunctivae", "État des conjonctives", 0, "str", None),
    ("breast_exam", "Examen des seins", "Breast exam", "Examen des seins", 0, "str", None),
    ("edema", "Œdèmes", "Oedema", "Œdèmes", 0, "yesno", None),
    ("fetal_movements", "Mouvements actifs", "Fetal movements", "Mouvements actifs", 0, "yesno", None),
    ("fundal_height_cm", "HU (cm)", "Fundal height (cm)", "HU (cm)", 0, "int", (5, 45)),
    ("fetal_heart_rate", "BCF", "Fetal heart rate (bpm)", "BCF", 0, "int", (90, 200)),
    ("speculum_exam", "Examen au spéculum", "Speculum exam", "Examen au spéculum", 0, "str", None),
    ("cervix", "TV : état du col", "Cervix", "TV : état du col", 0, "str", None),
    ("presentation", "TV : présentation", "Presentation", "TV : présentation", 0, "str", None),
    ("pelvis", "TV : bassin", "Pelvis", "TV : bassin", 0, "str", None),
    ("glycosuria", "Glucosurie", "Glycosuria", "Glucosurie", 0, "posneg", None),
    ("albuminuria", "Albuminurie", "Proteinuria", "Albuminurie", 0, "posneg", None),
    ("rubella", "Rubéole", "Rubella serology", "Rubéole", 0, "str", None),
    ("toxoplasmosis", "Toxoplasmose", "Toxoplasmosis serology", "Toxoplasmose", 0, "str", None),
    ("syphilis", "Syphilis (TPHA/VDRL)", "Syphilis test", "Syphilis (TPHA/VDRL)", 0, "posneg", None),
    ("hepatitis_b", "Ag HBs", "Hepatitis B (HBsAg)", "Ag HBs", 0, "posneg", None),
    ("hiv", "Sérologie VIH", "HIV test", "Sérologie VIH", 0, "posneg", None),
    ("hemoglobin_g_dl", "Hémoglobine", "Haemoglobin (g/dL)", "Hémoglobine", 0, "float", (5, 16)),
    ("platelets", "Plaquettes", "Platelets", "Plaquettes", 0, "str", None),
    ("glycemia_g_l", "Bilan glycémique", "Glycaemia (g/L)", "Bilan glycémique", 0, "float", (0.4, 4)),
    ("rai", "RAI (si Rh négatif)", "Antibody screen (if Rh-)", "RAI (si Rh négatif)", 0, "posneg", None),
    ("iron", "Fer", "Iron supplement", "Fer", 0, "yesno", None),
    ("examined_by", "Examen fait par", "Examined by", "Examen fait par", 0, "str", None),
]

CURRENT_PREGNANCY = [
    T("lmp_date", "DDR", "Last menstrual period", "DDR :", "date", section="dating"),
    T("height_cm", "Taille", "Height (cm)", "Taille :", "cm", range=(120, 200), section="dating"),
    R("blood_group", "Groupage", "Blood group",
      [O("A", "A"), O("B", "B"), O("O", "O"), O("AB", "AB")], section="dating"),
    R("rhesus", "Rhésus", "Rhesus", [O("negative", "Rh-"), O("positive", "Rh+")], section="dating"),
    T("expected_delivery_date", "Date prévue d'accouchement", "Expected delivery date",
      "DATE PRÉVUE D'ACCOUCHEMENT :", "date", section="dating"),
    T("term_exceeded_date", "Date de dépassement de terme", "Post-term date", "DATE DE DÉPASSEMENT DE TERME :",
      "date", section="dating"),
] + table("visits", "visits", VISIT_ROWS, VISIT_COLS)

# --- 4. delivery ----------------------------------------------------------------------

DELIVERY = [
    T("patient_name", "Patiente", "Patient name", "Patiente :", pii=True),
    R("setting", "Lieu", "Place of delivery",
      [O("supervised", "En milieu surveillé"), O("home", "A domicile")], section="delivery"),
    R("facility", "Type de structure", "Facility",
      [O("birth_house", "Maison d'accouchement"), O("maternity", "Maternité"),
       O("private_clinic", "Clinique privée")], section="delivery"),
    T("facility_other", "Autre structure", "Other facility", "Autres :", nth=0, section="delivery"),
    C("home_skilled_attendant", "Assisté par un personnel qualifié", "Skilled attendant at home",
      "Assisté par un personnel qualifié", section="delivery"),
    T("home_other", "Autre (domicile)", "Other (home)", "Autres :", nth=1, section="delivery"),
    T("delivery_date", "Date de l'accouchement", "Delivery date", "Date de l'accouchement :", "date",
      section="delivery"),
    R("mode", "Mode de l'accouchement", "Delivery mode",
      [O("vaginal_spontaneous", "Voie basse non instrumentale"), O("vaginal_instrumental", "Voie basse instrumentale"),
       O("cesarean_planned", "Césarienne : Programmée"), O("cesarean_emergency", "Urgence")], section="delivery"),
    C("forceps", "Forceps", "Forceps", "Forceps", section="delivery"),
    C("vacuum", "Ventouse", "Vacuum", "Ventouse", section="delivery"),
    C("episiotomy", "Avec épisiotomie", "Episiotomy", "Avec épisiotomie", section="delivery"),
    T("cesarean_indication", "Indication de césarienne", "Caesarean indication", "Préciser l'indication :",
      section="delivery"),
    C("complications", "Présence de complications", "Complications", "Présence de complications", side="right",
      section="complications"),
    C("complication_at_delivery", "Au moment de l'accouchement", "At delivery", "Au moment de l'accouchement",
      section="complications"),
    C("complication_postpartum", "Suites de couches", "Postpartum period", "Suites de couches",
      section="complications"),
    C("complication.preeclampsia", "Pré-éclampsie", "Pre-eclampsia", "Pré-éclampsie", section="complications"),
    C("complication.eclampsia", "Eclampsie", "Eclampsia", "Eclampsie", section="complications"),
    C("complication.hemorrhage", "Hémorragie", "Haemorrhage", "Hémorragie", section="complications"),
    C("complication.infection", "Infection", "Infection", "Infection", section="complications"),
    C("complication.other", "Autres", "Other", "Autres", section="complications"),
    T("complication_other_text", "Autre complication", "Other complication", "Si autres à préciser :",
      section="complications"),
    R("newborn_status", "État du nouveau-né", "Newborn status",
      [O("alive", "Vivant"), O("stillborn", "Mort-né"), O("death_under_24h", "Décès < 24 heures")],
      section="newborn"),
    T("newborn_sex", "Sexe", "Sex", "Sexe :", "sex", section="newborn"),
    T("birth_weight_g", "Poids à la naissance", "Birth weight (g)", "Poids à la naissance :", "grams",
      range=(300, 6500), section="newborn"),
    T("head_circumference_cm", "Périmètre crânien", "Head circumference (cm)", "Périmètre crânien à la naissance :",
      "cm", range=(20, 45), section="newborn"),
    T("anomaly", "Anomalie", "Abnormality", "Anomalie à préciser :", section="newborn"),
    T("gestational_age_weeks", "Âge gestationnel", "Gestational age (weeks)", "Âge gestationnel :", "weeks",
      range=(20, 45), section="newborn"),
]

# --- 5/7. postpartum — mother -------------------------------------------------------


def _postpartum_mother(early: bool):
    timing = (
        [O("day_7_8", "Entre le 7ème et 8ème jour après l'accouchement", side="right"),
         O("after_day_8", "Après le 8ème jour de l'accouchement", side="right")]
        if early else
        [O("day_40_50", "Entre le 40ème et 50ème jour après l'accouchement", side="right"),
         O("after_day_50", "Après le 50ème jour de l'accouchement", side="right")]
    )
    return [
        T("consultation_date", "Date de la consultation", "Consultation date", "Date de la consultation :", "date",
          section="consultation"),
        R("timing", "Moment de la consultation", "Consultation timing", timing, section="consultation"),
        T("temperature_c", "T°", "Temperature (°C)", "T°", "temp", range=(34, 42), section="vitals"),
        T("blood_pressure", "TA", "Blood pressure", "TA", "bp", section="vitals"),
        T("pulse_bpm", "Pouls", "Pulse (bpm)", "Pouls", "int", range=(40, 160), section="vitals"),
        T("weight_kg", "Poids", "Weight (kg)", "Poids", "kg", range=(30, 160), section="vitals"),
        R("conjunctivae", "État des conjonctives", "Conjunctivae",
          [O("normal", "Normales"), O("pale", "Décolorées")], section="exam"),
        C("uterine_globe", "Présence du globe utérin", "Uterine globe present", "Présence du globe utérin",
          section="exam"),
        C("lochia.bland", "Lochies fades", "Lochia: bland", "Fade", section="exam"),
        C("lochia.foul", "Lochies fétides", "Lochia: foul-smelling", "fétide", section="exam"),
        C("lochia.clear", "Lochies claires", "Lochia: clear", "claires", section="exam"),
        C("lochia.bloody", "Lochies sanglantes", "Lochia: bloody", "sanglantes", section="exam"),
        C("lochia.yellowish", "Lochies jaunâtres", "Lochia: yellowish", "Jaunâtres", section="exam"),
        R("perineum", "État du périnée", "Perineum",
          [O("normal", "Normal", 0), O("episiotomy", "Épisiotomie"), O("tear", "Déchirure")], section="exam"),
        C("episiotomy_repaired", "Épisiotomie réparée", "Episiotomy repaired", "Réparée", section="exam"),
        R("sphincters", "État des sphincters", "Sphincters",
          [O("normal", "Normal", 1), O("abnormal", "Anormal")], section="exam"),
        C("cesarean", "Césarienne", "Caesarean", "Césarienne :", section="exam"),
        T("scar_state", "État de la cicatrice", "Scar condition", "Etat de la cicatrice :", section="exam"),
        R("breasts", "État des seins", "Breasts",
          [O("normal", "Normal", 2), O("lymphangitis", "lymphangite"), O("mastitis_abscess", "mastite et abcès")],
          section="exam"),
        C("calves.normal", "Mollets normaux", "Calves: normal", "Normal", nth=3, section="exam"),
        C("calves.red", "Mollets rouges", "Calves: red", "Rouges", section="exam"),
        C("calves.warm", "Mollets chauds", "Calves: warm", "Chauds", section="exam"),
        C("calves.painful", "Douloureux à la dorsiflexion", "Calves: painful on dorsiflexion",
          "Douloureux à la dorsiflexion", section="exam"),
        C("complications", "Présence de complication", "Complications", "Présence de complication :", side="right",
          section="complications"),
        C("complication.hemorrhage", "Hémorragie", "Haemorrhage", "Hémorragie", section="complications"),
        C("complication.infection", "Infection", "Infection", "Infection", section="complications"),
        C("complication.eclampsia", "Eclampsie", "Eclampsia", "Eclampsie", section="complications"),
        C("complication.phlebitis", "Phlébite", "Phlebitis", "Phlébite", section="complications"),
        C("complication.breast", "Complications mammaires", "Breast complications", "Complications mammaires",
          section="complications"),
        C("complication.anemia", "Anémie", "Anaemia", "Anémie", section="complications"),
        C("complication.other", "Autres", "Other", "Autres", section="complications"),
        C("taking_medication", "Prise de médicaments", "Taking medication", "Notion de prise de médicaments :",
          side="right", section="treatment"),
        T("medication_details", "Médicaments pris", "Medication details", "Notion de prise de médicaments :",
          placement="after_box", section="treatment"),
        C("treatment.iron", "Fer", "Iron", "Fer", section="treatment"),
        C("treatment.vitamin_a", "Vitamine A", "Vitamin A", "Vitamine A", section="treatment"),
        T("treatment.other", "Autre traitement", "Other treatment", "Autres à préciser :", section="treatment"),
        T("next_appointment_date", "Prochain rendez-vous", "Next appointment", "Prochain rendez-vous le", "date",
          section="treatment"),
        C("family_planning.wants_method", "Désire une méthode", "Wants contraception", "Désire utiliser une méthode",
          section="family_planning"),
        R("family_planning.method", "Méthode", "Method", [O("pill", "pilule"), O("iud", "DIU")],
          section="family_planning"),
        T("family_planning.method_other", "Autre méthode", "Other method", "Autre à préciser :",
          section="family_planning"),
        C("family_planning.prescribed", "Prescription faite", "Prescribed", "Prescription faite",
          section="family_planning"),
        C("family_planning.referred", "Référée", "Referred", "Référée :", section="family_planning"),
        T("family_planning.reason_declined", "Pourquoi pas de méthode", "Reason for declining contraception",
          "Si la mère ne désire pas une méthode contraceptive : Pourquoi ?", placement="below",
          section="family_planning"),
    ]


# --- 6/8. postpartum — newborn ------------------------------------------------------

POSTPARTUM_NEWBORN = [
    T("consultation_date", "Date de la consultation", "Consultation date", "Date de la consultation :", "date",
      section="consultation"),
    T("age", "Âge", "Age", "Age", "days", range=(0, 120), section="vitals"),
    T("temperature_c", "Température", "Temperature (°C)", "Température", "temp", range=(34, 42), section="vitals"),
    T("weight_g", "Poids", "Weight (g)", "Poids", "grams", range=(500, 9000), section="vitals"),
    T("length_cm", "Taille", "Length (cm)", "Taille", "cm", range=(30, 75), section="vitals"),
    T("head_circumference_cm", "Périmètre crânien", "Head circumference (cm)", "Périmètre crânien", "cm",
      range=(25, 45), section="vitals"),
    C("premature", "Nouveau-né prématuré", "Premature newborn", "Nouveau-né prématuré", section="vitals"),
    C("hypotrophic", "Nouveau-né hypotrophe", "Small for gestational age", "Nouveau-né hypotrophe", section="vitals"),
    R("feeding", "Allaitement", "Feeding",
      [O("exclusive_breastfeeding", "exclusivement au sein"), O("formula", "Artificiel"), O("mixed", "mixte")],
      section="feeding"),
    C("danger.convulsions", "Convulsions", "Convulsions", "Convulsions", section="danger_signs"),
    C("danger.refuses_feeding", "Refus de téter", "Refuses to feed", "Refus de téter", section="danger_signs"),
    C("danger.hematemesis", "Hématémèses", "Haematemesis", "Hématémèses", section="danger_signs"),
    C("danger.melena", "Mélaenas", "Melaena", "Mélaenas", section="danger_signs"),
    C("danger.diarrhea", "Diarrhée", "Diarrhoea", "Diarrhée", section="danger_signs"),
    C("danger.jaundice", "Ictère", "Jaundice", "Ictère", nth=0, section="danger_signs"),
    C("danger.chest_indrawing", "Tirage sous costal", "Chest indrawing", "Tirage sous costal",
      section="danger_signs"),
    C("danger.cough", "Toux", "Cough", "Toux", section="danger_signs"),
    C("danger.abnormal_breathing", "Rythme respiratoire anormal", "Abnormal breathing rate",
      "Rythme respiratoire anormal", section="danger_signs"),
    C("danger.fever", "Fièvre", "Fever", "Fièvre", section="danger_signs"),
    C("danger.hypothermia", "Hypothermie", "Hypothermia", "Hypothermie", section="danger_signs"),
    T("danger.other", "Autre signe de danger", "Other danger sign", "Autres à préciser :", nth=0,
      section="danger_signs"),
    C("trauma.cephalhematoma", "Bosse sérosanguine / céphalhématome", "Cephalhaematoma",
      "Bosse sérosanguine ou céphalohématome", section="trauma"),
    C("trauma.hip_dislocation", "Luxation congénitale de la hanche", "Congenital hip dislocation",
      "Luxation congénitale de la hanche", section="trauma"),
    C("trauma.limb_mobility", "Mobilité d'un membre diminuée", "Reduced limb mobility",
      "Diminution ou absence de la mobilité d'un membre", section="trauma"),
    T("trauma.other", "Autre traumatisme", "Other trauma", "Autres à préciser :", nth=1, section="trauma"),
    R("breastfeeding_assessment", "Évaluation de l'allaitement", "Breastfeeding assessment",
      [O("normal", "Normal"), O("problems", "A problèmes")], section="feeding"),
    C("vaccine.bcg", "BCG administré", "BCG given", "BCG", section="vaccination"),
    C("vaccine.hepatitis_b", "HB administré", "Hepatitis B vaccine given", "HB", section="vaccination"),
    C("vitamin_d", "Supplémentation en vitamine D", "Vitamin D supplement", "Supplémentation en vitamine D",
      section="vaccination"),
    C("complication.jaundice", "Ictère", "Jaundice", "Ictère", nth=1, section="complications"),
    C("complication.infection", "Infection", "Infection", "Infection", section="complications"),
    C("complication.conjunctivitis", "Conjonctivite", "Conjunctivitis", "Conjonctivite", section="complications"),
    C("complication.trauma", "Traumatisme", "Trauma", "Traumatisme", section="complications"),
    C("complication.malformation", "Malformation", "Malformation", "Malformation", section="complications"),
    C("complication.other", "Autres", "Other", "Autres", section="complications"),
    T("seen_by", "Vu par", "Seen by", "Vu par :", section="follow_up"),
    T("decision", "Décision prise", "Decision", "Décision prise :", section="follow_up"),
    T("treatment", "Traitement prescrit", "Treatment", "Traitement prescrit :", section="follow_up"),
    C("transfer", "Transfert", "Transferred", "Transfert", section="follow_up"),
    T("referral_facility", "Établissement de référence", "Referral facility", "Préciser l'établissement de référence :",
      section="follow_up"),
    T("next_visit_date", "Visite de suivi le", "Next follow-up visit", "Revenir pour une visite de suivi nécessaire le",
      "date", section="follow_up"),
]

CATALOG: dict[str, list[Field]] = {
    "cover": COVER,
    "identification": IDENTIFICATION,
    "current_pregnancy": CURRENT_PREGNANCY,
    "delivery": DELIVERY,
    "postpartum_early_mother": _postpartum_mother(early=True),
    "postpartum_early_newborn": POSTPARTUM_NEWBORN,
    "postpartum_late_mother": _postpartum_mother(early=False),
    "postpartum_late_newborn": POSTPARTUM_NEWBORN,
}


def fields_for(page_type: str, include_pii: bool = False) -> list[Field]:
    return [f for f in CATALOG[page_type] if include_pii or not f.pii]


def get_field(page_type: str, key: str) -> Field:
    for f in CATALOG[page_type]:
        if f.key == key:
            return f
    raise KeyError(f"{page_type}.{key}")


def _check_unique_keys():
    for pt, fs in CATALOG.items():
        keys = [f.key for f in fs]
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f"duplicate keys in {pt}: {dupes}"


_check_unique_keys()
