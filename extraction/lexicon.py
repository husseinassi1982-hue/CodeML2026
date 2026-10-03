"""Domain vocabulary used to snap OCR near-misses ("Normawx" -> "Normaux").

Generic French clinical shorthand used on Moroccan maternal registries, the 12 official
regions of Morocco, its provinces/prefectures, and common occupations. Deliberately
general: no patient-specific text or staff names from the specimen data.
"""

from __future__ import annotations

from .catalog import Field

CLINICAL = [
    "RAS", "Néant", "Aucun", "Aucune", "Normal", "Normale", "Normales", "Normaux", "Anormal", "Anormale",
    "Pâles", "Décolorées", "Fermé", "Ouvert", "Effacé", "Céphalique", "Siège", "Transverse", "Immune",
    "Non immune", "Oui", "Non", "Neg", "Pos", "Positif", "Négatif", "Propre", "Propre, sèche", "Infectée",
    "Suintante", "Voie basse", "Césarienne", "Forceps", "Ventouse", "Maternité", "Domicile", "Hôpital",
    "Clinique", "Mère", "Père", "Oncle", "Tante", "Frère", "Sœur", "Grand-mère", "Grand-père",
    "Cycles réguliers", "Cycles irréguliers", "Non fait", "Fait", "Souffrance fœtale", "SFA",
    "Utérus cicatriciel", "Pré-éclampsie", "Pré-éclampsie sévère", "Éclampsie", "Hémorragie", "Infection",
    "Anémie", "HTA", "Diabète", "Asthme", "Asthme léger", "Épilepsie", "Cardiopathie", "Appendicectomie",
    "Cholécystectomie", "Fer", "Acide folique", "Vitamine A", "Vitamine D", "Sage-femme", "Infirmière",
    "Médecin", "Poursuivre l'allaitement exclusif", "Dystocie", "Macrosomie", "Prématurité",
]

EDUCATION = ["Aucun", "Analphabète", "Primaire", "Collège", "Lycée", "Supérieur", "Universitaire",
             "Secondaire", "Alphabétisée"]

OCCUPATIONS = [
    "Femme au foyer", "Sans profession", "Agricultrice", "Agriculteur", "Étudiante", "Étudiant", "Couturière",
    "Commerçante", "Commerçant", "Employée", "Employé", "Fonctionnaire", "Ouvrière", "Ouvrier", "Chauffeur",
    "Mécanicien", "Maçon", "Enseignante", "Enseignant", "Infirmière", "Coiffeuse", "Artisan", "Artisane",
    "Menuisier", "Électricien", "Plombier", "Pêcheur", "Berger", "Militaire", "Policier", "Gardien",
    "Journalier", "Vendeuse", "Vendeur", "Cultivateur", "Éleveur", "Retraité", "Chômeur", "Sans emploi",
]

REGIONS = [
    "Tanger-Tétouan-Al Hoceïma", "Oriental", "Fès-Meknès", "Rabat-Salé-Kénitra", "Béni Mellal-Khénifra",
    "Casablanca-Settat", "Marrakech-Safi", "Drâa-Tafilalet", "Souss-Massa", "Guelmim-Oued Noun",
    "Laâyoune-Sakia El Hamra", "Dakhla-Oued Ed-Dahab",
]

PROVINCES = [
    "Tanger-Assilah", "M'diq-Fnideq", "Tétouan", "Fahs-Anjra", "Larache", "Al Hoceïma", "Chefchaouen", "Ouezzane",
    "Oujda-Angad", "Nador", "Driouch", "Jerada", "Berkane", "Taourirt", "Guercif", "Figuig", "Fès", "Meknès",
    "El Hajeb", "Ifrane", "Moulay Yacoub", "Sefrou", "Boulemane", "Taounate", "Taza", "Rabat", "Salé",
    "Skhirate-Témara", "Kénitra", "Khémisset", "Sidi Kacem", "Sidi Slimane", "Béni Mellal", "Azilal",
    "Fquih Ben Salah", "Khénifra", "Khouribga", "Casablanca", "Mohammédia", "El Jadida", "Nouaceur", "Médiouna",
    "Benslimane", "Berrechid", "Settat", "Sidi Bennour", "Marrakech", "Chichaoua", "Al Haouz", "El Kelâa des Sraghna",
    "Essaouira", "Rehamna", "Safi", "Youssoufia", "Errachidia", "Ouarzazate", "Midelt", "Tinghir", "Zagora",
    "Agadir Ida-Outanane", "Inezgane-Aït Melloul", "Chtouka-Aït Baha", "Taroudant", "Tiznit", "Tata", "Guelmim",
    "Assa-Zag", "Tan-Tan", "Sidi Ifni", "Laâyoune", "Boujdour", "Tarfaya", "Es-Semara", "Oued Ed-Dahab",
    "Aousserd",
]


# fields whose value must be one of the list (anything else is a misreading)
CLOSED_VOCAB = {"region", "province", "education_level"}


def vocab_for(f: Field) -> list[str]:
    k = f.key
    if k == "region":
        return REGIONS
    if k == "province":
        return PROVINCES
    if k == "education_level":
        return EDUCATION
    if k in ("profession", "husband_profession"):
        return OCCUPATIONS
    if k in ("facility_name", "record_number") or k.endswith(("examined_by", "seen_by")) or "platelets" in k:
        return []
    return CLINICAL
