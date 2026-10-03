# Extractor sample outputs

Real outputs of `extraction.extract()` on fictional patient 1's pages, for building and
testing the chat UI and backend without running OCR. Regenerate with
`python -m tools.make_fixtures`.

| File | What |
|---|---|
| `<page_type>.json` | full `PageExtraction` (schema in `extraction/schema.py`) |
| `<page_type>.ui.json` | same, as `{key: {label, value, conf, status}}` for the WhatsApp mock-up |
| `cover_photo.json`, `current_pregnancy_photo.json` | the page through a simulated phone photo: more fields in `needs_review` |
| `manual_<page_type>.json` | empty form (all `NOT_PROVIDED`) for full manual entry when the AI is unavailable |

Things the chat should use: `needs_review` (ask these, in order), each field's
`question_fr` / `question_en`, `image_quality.issues` (offer "Retake photo"),
`record_number` (patient code, always to be confirmed), and `page_type == "unknown"`
(page not recognised: retake or manual entry).
