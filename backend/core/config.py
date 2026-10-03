"""Settings and import paths.

The backend glues together two modules that live elsewhere in the repo:
  extraction/            (repo root)      photo -> structured fields
  offlineModule/offline  (offlineModule/)  encrypted store, record states, queue
Adding their folders to sys.path lets us `import extraction` and `import offline`
without packaging them. Run the server from inside backend/:  uvicorn main:app --reload
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
FRONTEND_FILE = REPO_ROOT / "front_end2.HTML"

for p in (REPO_ROOT, REPO_ROOT / "offlineModule"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Where records, photos and the encryption key are kept (created if missing).
DATA_DIR = Path(os.environ.get("DAYONE_DATA_DIR", BACKEND_DIR / "app_data"))

# Which extractor runs on queued photos:
#   local   - offline OCR (default, what we demo and evaluate)
#   claude  - optional cloud backend, needs ANTHROPIC_API_KEY and DAYONE_ALLOW_CLOUD=1
#   fixture - no OCR at all: every photo returns a saved real output from fixtures/.
#             Used automatically if the OCR packages are not installed.
EXTRACTOR = os.environ.get("DAYONE_EXTRACTOR", "local")

# Does AI processing need the internet? The local OCR runs on the device, so by default
# only the cloud backend waits for a connection. Set DAYONE_AI_NEEDS_NETWORK=1 to make the
# demo queue AI processing while offline too.
_flag = os.environ.get("DAYONE_AI_NEEDS_NETWORK")
AI_NEEDS_NETWORK = (_flag == "1") if _flag is not None else (EXTRACTOR == "claude")

# Fixture used for photos in fixture mode when the upload doesn't name one.
DEFAULT_FIXTURE = "cover_photo"
