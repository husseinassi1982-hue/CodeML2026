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

The OCR models (~15 MB) ship inside the `rapidocr` wheel; no download at runtime.

## How it works

```
photo ─► quality check (blur, darkness)          ─► "retake" hint
      ─► OCR of the whole page (RapidOCR / PaddleOCR models, ONNX, CPU)
      ─► page type: which of the 8 registry pages, from its printed labels
      ─► alignment: printed labels ↔ template → homography (shift, rotation, perspective),
         refined on the measured ink of every label (≈0.3 pt error on clean pages)
      ─► for every field of that page (catalog.py):
           text/cell  rectified crop → subtract the form's printed ink (template mask)
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
| Simulated photo, medium | 85.6% | 88.2% | 19 (2.3%) | 19.3% | ~2.4 |
| Simulated photo, hard | 45.5% | 51.6% | 0 (0%) | 32.1% | ~1.7 |

Checkboxes: 100% on clean and medium photos. Page type: 100% except two unreadable "hard"
photos, which are refused rather than guessed. As the photo gets worse, accuracy drops but
the agent says so: the silent-error rate stays low and the review rate rises. Full
breakdown per page type and confidence calibration: `evaluation/RESULTS.md`.

## Known limitations

* **Dataset quirk:** one handwriting font in the specimen (NanumPen) has no glyph for
  accented letters or "—". The PDF records them as U+FFFD and the image shows a gap
  ("coll ge"). Scoring treats the gap as a wildcard and accepts "blank" for an invisible dash.
* The local backend only knows the 8 specimen page layouts. A page it cannot align returns
  `page_type="unknown"` → retake, manual entry, or the Claude backend.
* Unticked boxes are reported as `false`/KNOWN: on paper "not ticked" and "not filled in"
  look the same.
* OCR is trained mostly on print; unusual handwriting lowers confidence (and so raises the
  review rate) rather than producing silent errors — that is the intended trade-off.
* Arabic handwriting is not supported by the local OCR models.
