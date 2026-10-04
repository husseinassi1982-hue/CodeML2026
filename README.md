# DayOne — The Offline Midwife

A WhatsApp-style assistant that turns a photo of a page of the pink Moroccan maternal registry into a
structured digital record: each field with a **value, a status and a confidence**, checked by the midwife
in a chat, linked to the woman's earlier visits by her registry code. The phone works offline: photos are
kept encrypted on it and sent for processing when the network returns. The paper registry stays the reference: the digital layer sits on top of it.

> CodeML 2026 — DayOne challenge ("A midwife, a phone and an AI: pregnancy follow-up goes digital").
> All data in this repository is the organisers' synthetic data or our own synthetic data.

## What it does

1. **Capture (on the phone, works offline).** The midwife photographs a registry page in the chat. The
   photo is saved **encrypted on the phone** and waits there (`CAPTURED → PENDING_AI`) while there is no
   network; she can keep photographing pages. The chat page itself is cached on the phone, so it opens
   with no network too.
2. **Upload and extraction.** When the phone is back online it sends the waiting photos (oldest first) to
   the processing server, and deletes each one from the phone only once the server confirms it has it.
   The server reads the page: page type → every field → handwriting → value + status + confidence
   (`extraction/`). Identifiers (name, ID number, phone, address, husband's name) are never read or stored.
3. **Review.** The chat shows what was read and asks only about uncertain fields:
   **Confirm / Edit / Retake photo**, follow-up questions for illegible fields, full manual entry when the
   AI is unavailable. The agent says when it is unsure.
4. **Patient linking.** By the code the midwife writes on the booklet: exact and near matches (one
   character off) are proposed — *Patient 1 / Patient 2 / None, create new / I'm not sure* — the
   midwife decides; a patient is never created automatically when a match is plausible.
5. **Sync.** Saved records are uploaded when connectivity returns (`SAVED → SYNCED`); failures are retried,
   nothing is lost.

## Architecture

```
 Phone — the only input device               Processing server (health system's own machine;
 (installable web app, works offline)         in the demo, the team's GPU laptop — nobody types on it)

 front_end2.HTML + pwa/sw.js                  backend/  FastAPI: records, review, linking, sync
  ├─ page cached → opens with no network        │
  ├─ outbox: photos AES-GCM encrypted in        ├─ offlineModule/  encrypted record store (Fernet, SQLite),
  │  IndexedDB, key generated on the phone      │                  record state machine, AI queue
  │  (not exportable)                           └─ extraction/     photo -> fields (value/status/confidence)
  └─ online again → uploads oldest first ──▶          ├─ CPU: PP-OCRv6 (ONNX) + field-aware decoding  (default)
     ◀── fields, questions ── review in chat           └─ GPU: fine-tuned Qwen3-VL-8B reader            (optional)
```

No photo leaves the phone except to the processing server, and the phone keeps its copy until that server
has stored it (a photo sent twice is flagged `SUSPECTED_DUPLICATE`, never processed twice silently). The
header's **📶 Online** switch makes the phone behave as if it had no network, for the demo; a real loss of
network does the same automatically.

| Folder | Owner | What |
|---|---|---|
| `extraction/` | AI / OCR | Page type, field location, handwriting reading, statuses, checks, identifier redaction ([README](extraction/README.md)) |
| `offlineModule/offline/` | Offline | Encrypted local store, record states, processing and sync queue |
| `backend/` | API | FastAPI: capture, queue, review answers, manual entry, linking, patients, sync, role-restricted photos |
| `front_end2.HTML`, `pwa/` | Chat | WhatsApp-style chat served by the backend; offline outbox, service worker, app manifest |
| `tools/`, `evaluation/`, `notebooks/` | AI | Evaluation, ground truth, crop editor, model training |

## Quick start

```bash
git clone https://github.com/husseinassi1982-hue/CodeML2026.git && cd CodeML2026
python3 -m venv .venv
.venv/bin/pip install -r requirements-extraction.txt -r backend/requirements.txt
.venv/bin/python -m extraction --download-models     # once, online: optional second OCR model
cd backend && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` (from a phone on the same Wi-Fi, see **Phone setup** below).
API documentation: `http://localhost:8000/docs`. Tests: `cd backend && ../.venv/bin/python -m pytest`,
`cd offlineModule && ../.venv/bin/python -m pytest`.

Without the OCR packages the API still runs and returns saved extractor outputs (`fixtures/`).

### With the fine-tuned vision model (GPU)

Needs an NVIDIA GPU with ~7 GB free (e.g. an RTX 4060 laptop, or Colab T4). Native Windows 11, PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"          # must print True
.venv\Scripts\pip install -r requirements-extraction.txt -r backend\requirements.txt -r requirements-vlm.txt
```

1. Download [`vlm_ocr_lora_v2.zip`](https://github.com/husseinassi1982-hue/CodeML2026/releases/tag/vlm-ocr-v2)
   (192 MB) into the repo folder; leave it zipped.
2. Load it once (downloads the 4-bit Qwen3-VL-8B base, ~6 GB, from Hugging Face; no data is sent):
   `$env:DAYONE_VLM="1"; .venv\Scripts\python -c "from extraction import vlm; vlm.load_or_raise(); print('VLM ready')"`
3. Measure it on the real photos before relying on it (CPU baseline: 85/127):
   `.venv\Scripts\python -m tools.evaluate_real --backend vlm`
4. Run the app with it: `cd backend; $env:DAYONE_EXTRACTOR="vlm"; ..\.venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8080`
   and allow the port through the Windows firewall (Private networks).

On Linux the same steps work with `.venv/bin/...`; `pip install unsloth` instead of `requirements-vlm.txt`
is also supported. Without a GPU, pages wait as `PENDING_AI` until a machine with the model processes the queue.

### Phone setup

Browsers keep encrypted data and offline pages only for **secure** addresses (`https://…`, or
`http://localhost`). Over plain `http://<server-IP>:8000` the chat still works, but it warns that it cannot
keep photos on the phone while offline. Two ways to give the phone a secure address:

* **Demo (Android Chrome):** open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add
  `http://<server-IP>:8000`, tap *Relaunch*.
* **Deployment:** serve HTTPS with a certificate the phone trusts (e.g. made with
  [mkcert](https://github.com/FiloSottile/mkcert), its root certificate installed on the phone):
  `uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile key.pem --ssl-certfile cert.pem`.

Open the page once while online, then *Add to Home screen*: from then on it opens without network.
To demo: tap **📶 Online** (→ ✈️ Offline), photograph pages (they show as "📱 N on phone"), tap again: the
phone sends them and the review starts. Closing the page or turning off the server meanwhile loses nothing.

## Record lifecycle

| Brief | Ours | Meaning |
|---|---|---|
| CAPTURED | `CAPTURED` | photo saved, encrypted, on the phone (its capture time is kept when it is uploaded later) |
| PENDING_AI | `PENDING_AI` | waiting for AI processing: on the phone while offline, then in the server's queue |
| AI_PROCESSED | `AI_PROCESSED` | fields extracted |
| NEEDS_REVIEW | `TO_REVIEW` | waiting for the midwife to check |
| VALIDATED | `VALIDATED` | midwife confirmed |
| PATIENT_MATCHED | `PATIENT_LINKED` | attached to a patient profile |
| REGISTERED | `SAVED` | stored in final form on the device |
| SYNCED | `SYNCED` | uploaded (terminal) |
| failures | `PROCESSING_FAILED`, `SYNC_FAILED`, `SUSPECTED_DUPLICATE`, `MANUAL_REVIEW_REQUIRED` | each has a way out (retry, manual entry) |

Every change goes through one transition table (`offlineModule/offline/states.py`): an illegal jump is
rejected, every transition is logged. The same photo captured twice is flagged `SUSPECTED_DUPLICATE`.

## How the brief's constraints are met

* **No real patient data to a third party.** Photos go only from the phone to the health system's own
  processing server (CPU OCR, or the GPU vision model on the team's machine). The optional cloud backend (`extraction/backends/claude.py`) refuses to start unless
  `DAYONE_ALLOW_CLOUD=1` and masks identifiers first; it is not used by default. The vision model was
  trained in Colab on synthetic data only.
* **Offline layer without internet.** On the phone: the app opens, photos are captured and kept encrypted,
  and the queue survives closing the app, all with no network; they are sent when it returns, and nothing
  is deleted from the phone before the server confirms it. Reading the page and reviewing it happen once
  the phone is back online.
* **Identifiers never stored.** Identifier fields are skipped by the extractor and painted out of every
  crop; only the midwife's random patient code is used for linking. Internal record IDs are random UUIDs.
* **Original image kept, restricted.** Stored byte-for-byte (encrypted, on the phone and then on the
  server) with record ID, capture date,
  midwife ID and status; only the `midwife` and `admin` roles can view it.
* **Originals and ground truth unmodified.** Checked against the organisers' manifest checksums.

## Results

Extraction, local CPU backend (details and method: [extraction/README.md](extraction/README.md),
[evaluation/RESULTS.md](evaluation/RESULTS.md)):

| Test | Field accuracy | Silent errors (accepted but wrong) | Sent to review |
|---|---|---|---|
| Specimen pages, held-out patients 8-10, clean | 98.9% (written values 99.6%) | 1 / 1692 | 1.7% |
| Same, simulated phone photos (medium) | 81.9% (written values 91.9%) | 15 (1.9%) | 24% |
| **Real booklet photos (5 pages, 127 hand-transcribed fields)** | **67%** | **1** | 52% |
| Same, vision model v2 reading the handwriting (`backend="vlm"`, GPU) | 75% | 3 | 54% |

Fine-tuned vision model (exact transcription of single field crops):

| Held-out crops | CPU OCR | Vision model |
|---|---|---|
| Specimen pages (300) | 64% | 95% |
| Synthetic crops in the midwife's formats | 58% | 94% |
| Real booklet photos (93) | 46% | 50.5% (first model; v2 measured on whole pages above) |

The synthetic tests are easy; real cursive is the hard part. The design choice is that the system
**asks rather than guesses**: on real photos about half the fields go to the midwife, almost none are
accepted wrongly.

## Known limitations

* Real handwriting: about two thirds of fields right on the 5 real photos (which we also used while
  developing); new booklets will do somewhat worse. Uncertain fields go to review.
* Arabic handwriting is not read.
* The real booklet's delivery and postpartum pages were not in the data: not supported yet (the
  specimen layout of those pages is).
* The vision model needs a GPU. On the 5 real photos it reads more fields right than the CPU OCR
  (75% vs 67%, mostly on the dense visit table) but makes more confident mistakes (3 vs 1) and is
  ~9x slower; a stricter confidence bar or using it as a second reader (`DAYONE_VLM=1`) is safer.
* WhatsApp is simulated (web chat); the Business API sandbox is not connected.
* The phone's key is a non-exportable browser key; a native app would keep it in the OS keystore. The
  server's key is in a local file for the demo.
* Review needs the network (the reading happens on the server); a phone with no network can capture
  but not review.

## Team

Person 1 — extraction and AI · Person 2 — offline storage and queue · Person 3 — backend API ·
Person 4 — chat front end.
