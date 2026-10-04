# Extraction pipeline (Person 1)

Photo of one page of the paper maternal registry → structured fields, each with a
**status** and a **confidence**. Runs fully offline on a CPU laptop; nothing leaves the
machine. An optional Claude backend is available when an API key is configured.

```python
from extraction import extract, ExtractionError

result = extract("photo.jpg")          # path, bytes, PIL image or numpy array
result.page_type                       # "current_pregnancy"
result.record_number                   # midwife's patient code (cover page) → patient linking
result.fields["visits.m9.blood_pressure"].status     # Status.KNOWN
result.needs_review                    # keys to ask the midwife about, least confident first
result.to_ui_fields()                  # {key: {label, value, conf, status}} for the chat mock-up
result.model_dump_json(indent=2)       # what the backend stores / sends
```

Command line:

```bash
python -m extraction "data/Paper Registry/dossiers_specimen_10_patientes-03.png" --review
```

## Integration (Persons 2, 3, 4)

Suggested `backend/services/ai_service.py` (runs when a queued capture is processed):

```python
from extraction import ExtractionError, empty_form, extract
from extraction.redact import redact_photo

def process_capture(image_bytes: bytes) -> tuple[str, dict | None]:
    """-> (next record state, extraction JSON)."""
    try:
        result = extract(image_bytes)                      # local, offline, ~3 s on a laptop CPU
    except ExtractionError as e:
        if e.code == "unreadable_image":
            return "PROCESSING_FAILED", None               # ask the midwife to retake the photo
        return "PENDING_AI", None                          # keep queued, retry later
    if result.page_type == "unknown":
        return "MANUAL_REVIEW_REQUIRED", empty_form("cover").model_dump(mode="json")
    state = "NEEDS_REVIEW" if result.needs_review else "AI_PROCESSED"
    return state, result.model_dump(mode="json")
```

* `result.record_number` is the midwife's patient code from the cover page (always returned
  as `NEEDS_REVIEW` so the midwife confirms it before linking).
* `redact_photo(image_bytes)` returns a copy with identifiers blacked out: store it next to
  the restricted original.
* `empty_form(page_type)` gives every field as `NOT_PROVIDED`: the chat can walk through it
  for full manual entry when the AI is unavailable.
* Sample outputs for building the UI without running OCR: `fixtures/` (`python -m tools.make_fixtures`).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-extraction.txt
ln -s ../dayone-participants/data data      # or: export DAYONE_DATA=/path/to/data
```

The base OCR models (~30 MB) ship inside the `rapidocr` wheel. Real booklet photos also use a
larger recogniser as a second opinion (PP-OCRv6 medium, ~73 MB), downloaded once while online:

```bash
.venv/bin/python -m extraction --download-models
```

Without it (offline first run) extraction still works with the small model alone: no
agreement between two readers, so more fields go to review (real photos: 54% right, 0 silent
errors, 62% to review, vs 55% / 2% / 55% with it). No photo or text is ever sent anywhere.

## How it works

```
photo ─► quality check (blur, darkness)          ─► "retake" hint
      ─► OCR of the whole page (RapidOCR / PaddleOCR models, ONNX, CPU)
      ─► page type: which of the 8 registry pages, from its printed labels
      ─► alignment: printed labels ↔ template → homography (shift, rotation, perspective),
         refined on the measured ink of every label (≈0.3 pt error on clean pages)
      ─► for every field of that page (catalog.py):
           text/cell  rectified crop → remove printed lines (and the form's printed ink, if a
                      printed_<page>.png mask from tools/build_templates.py is present)
                      → ink? → OCR → parse → status
           checkbox   rectified crop → locate the printed square → ink inside → ticked?
      ─► cross-field checks (due date = LMP + 280 d, GA vs visit date, parity ≤ gravidity…)
      ─► PageExtraction (schema.py)
```

* `catalog.py` – the field schema: 8 page types, ~550 fields (keys, FR/EN labels, data
  type, plausible range, identifier flag). Table fields are `section.column.row`, e.g.
  `visits.m9.hiv`, `previous_delivery.2.newborn_weight_g`.
* `template.py` + `data/templates.json` – where every label/field/checkbox is on each page,
  built from the specimen PDF by `tools/build_templates.py`.
* `normalize.py` – text → typed value (ISO dates, `{systolic, diastolic}`, weeks, grams…),
  OCR digit fixes (O→0, l→1, "g"→"9"), snapping to a clinical/geographic lexicon (`lexicon.py`).
* `status.py` – the status/confidence rules (below). `validate.py` – cross-field checks.
* `redact.py` – masks identifiers on a copy of the photo (for sharing / sending anywhere).
* `backends/local.py` (default) and `backends/claude.py` (optional, needs `ANTHROPIC_API_KEY`).

**Cloud use and the brief.** The brief forbids sending real patient data to a third-party
service. The local backend never uses the network and is what we evaluate and demo. The
Claude backend refuses to start unless `DAYONE_ALLOW_CLOUD=1` is set, which is only
acceptable for the organisers' synthetic data; when it can, it also masks identifiers on the
copy it sends. In a real deployment the same interface would point to a model hosted by
the health system itself.

## Real booklet photos (the pink "Fiche de surveillance")

The organisers' 5 real photos (`1-1.jpg` … `1-5.jpg`) show the actual booklet as a midwife
uses it: a different page layout from the specimen (the visit table spread over two facing
pages, the right page without row labels), photos at an angle with a curved spine, and real
cursive handwriting in French shorthand. `booklet.py` handles these pages without templates:

* deskews the photo from its printed table lines, finds the printed labels by OCR, and builds
  each table cell from row labels × column headers, snapped to the grid lines (a border hidden
  under handwriting falls back to the midpoint between headers);
* the unlabelled right page places its rows relative to the four shaded section bands, using
  the row positions measured on the left page (`tools/booklet_rows.py`);
* reads whole handwritten words even when they spill over the thin rows, removes printed lines,
  normalises the pink/shadowed paper to grey (+8 points of character accuracy);
* reads every field with two recognisers (PP-OCRv6 small and medium) plus the OCR's reading
  of the whole printed line; **agreement raises confidence, disagreement sends the field to
  review**;
* understands the midwife's shorthand: French-style "1" (Λ), "16SA+3j", "11/7" (cmHg),
  "NF" (non fait → `NOT_APPLICABLE`), "+", "0", "nég", "Reçu", "RAS";
* checkboxes as printed squares next to their label; options circled by hand (blood group,
  rhesus) by reading the printed text inside the ring; "RAS" written diagonally across a
  section is reported on every row it crosses, always for review.

Accuracy on these 5 photos is in the results below: honest but limited by how well a small
local OCR model reads fast cursive. A stronger recogniser (a vision-language model running on
a GPU inside the health system) plugs into the same interface.

## Fine-tuned vision model (GPU)

A vision-language model (Qwen3-VL-8B, 4-bit, + a LoRA adapter) fine-tuned to read one registry field
at a time: `notebooks/finetune_vlm_ocr.ipynb` (Colab T4, ~1 h), data from `tools/export_vlm_dataset.py`
(specimen pages, patients 1-7) and `tools/synth_midwife.py` (synthetic crops in the real midwife's formats:
cmHg "11/7", "16SA+3j", "NF", "Reçu", "nég"…). Trained on synthetic data only.

```python
extract("photo.jpg", backend="vlm")       # the trained model reads every handwritten value
```

* **The model only reads.** The CPU pipeline still finds the page type, every field, identifiers (painted
  out before the model sees anything) and checkboxes; each field's area plus a margin is given to the
  model with the field's label and expected format, exactly as in training. Confidence = probability of
  the least certain generated token; below the bar (0.80 scans / 0.88 photos) the field goes to review.
* **Needs** a CUDA GPU with ~7 GB free (Colab/Kaggle T4, RTX 4060 laptop), `pip install unsloth`, and the
  adapter `vlm_ocr_lora_v2.zip` (GitHub release *vlm-ocr-v2* of this repo) in the repo folder or at
  `DAYONE_VLM_ADAPTER`. Everything runs on that machine: nothing is sent anywhere.
* **Without them** `backend="vlm"` raises `backend_unavailable`: in the app (`DAYONE_EXTRACTOR=vlm`) the
  page waits as PENDING_AI until a machine with the model processes the queue.
* `DAYONE_VLM=1` with the default backend uses the model as a *second* reader instead: agreement with the
  CPU OCR -> KNOWN, disagreement -> review.

Measured (exact transcription of single field crops):

| Held-out crops | CPU OCR | Model, run 1 | Model, run 2 (+ midwife formats) |
|---|---|---|---|
| Specimen pages, patients 8-10 (300) | 64% | 92% | 95% |
| Synthetic crops in the midwife's formats (150-600) | 48-58% | 76% | 94% |
| **Real booklet photos (93)** | 46% | **50.5%** | not measured yet (needs a GPU) |

The synthetic tests are easy (same generators as training); the real-photo line is the one that counts.
On whole specimen pages the CPU pipeline itself reads 99% of written values (it reads tight ink crops,
not the margin crops of this table).

## Statuses

| Status | Local backend rule |
|---|---|
| `NOT_PROVIDED` | no ink in the field (after removing printed lines) |
| `NOT_APPLICABLE` | a handwritten dash "—" |
| `UNKNOWN` | "?", "inconnu", "NSP" written |
| `ILLEGIBLE` | ink present but OCR cannot read it (score < 0.35) |
| `NEEDS_REVIEW` | read, but confidence below the bar (0.80 for scans, 0.88 for phone photos), or unparseable, out of plausible range, not a known value (region, province, education), an unrecognised short word, failed a cross-field check, or several boxes ticked in a single-choice group. The patient code is always `NEEDS_REVIEW`: a misread digit would link the visit to another woman |
| `KNOWN` | read with confidence above the bar and passes all checks |

Confidence = OCR score × alignment quality × lexicon-snap similarity (× 0.9 if the photo
is blurry/dark), capped at 0.5–0.6 when a check fails. For checkboxes it grows with the
distance of the ink ratio from the decision threshold. The two bars were chosen on the
development patients only: on simulated phone photos 0.88 halves the silent errors of 0.80,
while on clean scans a higher bar only adds review work (`status.py`).

**Identifiers** (name, CIN, phone, address, husband's name, name in page headers) are never
read: their regions are skipped, they are absent from the output, and `redact.py` blacks
them out on image copies.

## Evaluation

Ground truth comes for free: the specimen PDF stores every handwritten value in a
handwriting font and every tick as a coloured stroke, so `tools/build_ground_truth.py`
recovers the exact value of 5,640 fields on 80 pages (identifiers excluded) into
`evaluation/ground_truth.json`.

```bash
python -m tools.evaluate --split dev                 # patients 1-7 (used while building)
python -m tools.evaluate --split test                # patients 8-10 (held out)
python -m tools.evaluate --split test --degrade medium --show-errors   # simulated phone photos
```

Held-out test patients (8–10, never used while building), local backend, CPU only:

| Condition | Field accuracy | Written values right | Silent errors (KNOWN but wrong) | Sent to review | s/page |
|---|---|---|---|---|---|
| Clean scans | 98.2% | 98.7% | 1 / 1692 (0.1%) | 2.7% | ~2.6 |
| Simulated photo, mild | 96.4% | 98.3% | 1 (0.1%) | 5.7% | ~2.5 |
| Simulated photo, medium | 83.0% | 89.5% | 19 (2.4%) | 23.6% | ~2.6 |
| Simulated photo, hard | 45.5% | 51.6% | 0 (0%) | 32.1% | ~1.7 |

Real booklet photos (5 pages, 127 fields transcribed by hand from the images,
`evaluation/real_photos_truth.json`, `python -m tools.evaluate_real`): **55% of fields right,
2% silent errors, 55% sent to review**. Before booklet support all 5 pages were rejected as
unknown. The 2 silent errors are on near-empty fields (a stray "0" and "1").

Checkboxes: 100% on clean and medium photos. Page type: 100% except two unreadable "hard"
photos, which are refused rather than guessed. As the photo gets worse, accuracy drops but
the agent says so: the silent-error rate stays low and the review rate rises. Full
breakdown per page type and confidence calibration: `evaluation/RESULTS.md`.

## Known limitations

* **Dataset quirk:** one handwriting font in the specimen (NanumPen) has no glyph for
  accented letters or "—". The PDF records them as U+FFFD and the image shows a gap
  ("coll ge"). Scoring treats the gap as a wildcard and accepts "blank" for an invisible dash.
* The local backend knows the 8 specimen layouts and 5 pages of the real booklet (cover,
  identification, obstetric history, pregnancy left/right). Postpartum and delivery pages of
  the real booklet were not in the data, so they are not supported yet; an unrecognised page
  returns `page_type="unknown"` → retake, manual entry, or another backend.
* Real cursive is read by a small local OCR model: about half the transcribed fields come out
  right, and most of the rest are flagged for review rather than guessed.
* Unticked boxes are reported as `false`/KNOWN: on paper "not ticked" and "not filled in"
  look the same.
* OCR is trained mostly on print; unusual handwriting lowers confidence (and so raises the
  review rate) rather than producing silent errors — that is the intended trade-off.
* Arabic handwriting is not supported by the local OCR models.
