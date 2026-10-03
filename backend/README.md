# DayOne backend

Wires the three parts together: **extraction** (photo → fields) → **offline store**
(encrypted records + state machine + queue) → **chat** (`front_end2.HTML`).

## Run

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
pip install -r ../requirements-extraction.txt   # the real OCR (optional, see below)
uvicorn main:app --reload
```

- **http://localhost:8000** — the chat, talking to the real backend
- **http://localhost:8000/docs** — every endpoint, try them in the browser
- `python -m pytest -q tests` — 14 tests

If the OCR packages aren't installed, the server still runs: photos get a saved real
output from `fixtures/` instead (the header shows "demo extractor"). The 📄 sample button
always uses fixtures (cover page, then the current-pregnancy page) so the demo is the same
on every laptop. Real photos uploaded with 📎 go through the real OCR.

## Flow

| Chat step | Endpoint | Record state |
|---|---|---|
| photo sent | `POST /records` | CAPTURED → PENDING_AI (or SUSPECTED_DUPLICATE) |
| AI reads it | `POST /queue/process` | → AI_PROCESSED → TO_REVIEW (or PROCESSING_FAILED / MANUAL_REVIEW_REQUIRED) |
| midwife answers | `PATCH /records/{id}/fields` | TO_REVIEW |
| confirm all | `POST /records/{id}/validate` | → VALIDATED |
| choose patient | `GET /patients?code=…`, `POST /records/{id}/link` | → PATIENT_LINKED → SAVED |
| upload | `POST /sync` | → SYNCED (waits while offline) |

Others: `GET /records/{id}` (all fields + history), `POST /records/{id}/manual` (type a page
the AI couldn't read), `POST /records/{id}/retry`, `GET /records/{id}/image` (header
`X-Role: midwife`, restricted), `POST /connectivity` (demo offline switch), `GET /status`.

## Design choices worth knowing for the pitch

- **Local OCR works offline**, so by default only *sync* waits for the network. To show the AI
  queue filling up offline too, start with `DAYONE_AI_NEEDS_NETWORK=1`.
- **Never auto-links a patient.** A code one character off (misread digit) is shown as a
  candidate; the midwife decides. "I'm not sure" leaves the page unlinked.
- **Typed answers are checked**: a date must be a date, Pos/Neg must be Pos/Neg, otherwise
  the chat asks again. `?` means UNKNOWN, `-` means not applicable.
- Settings (env vars): `DAYONE_DATA_DIR`, `DAYONE_EXTRACTOR` (`local` / `claude` / `fixture`),
  `DAYONE_AI_NEEDS_NETWORK`, `DAYONE_KEY`. See `core/config.py`.
