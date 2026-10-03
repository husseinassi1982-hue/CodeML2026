"""Where the organisers' participant package lives. Override with DAYONE_DATA=/path/to/data."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DAYONE_DATA", REPO / "data"))
REGISTRY = DATA / "Paper Registry"
SPECIMEN_PDF = REGISTRY / "dossiers_specimen_10_patientes.pdf"
EVAL_DIR = REPO / "evaluation"
GROUND_TRUTH = EVAL_DIR / "ground_truth.json"
