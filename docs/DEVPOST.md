# DayOne — The Offline Midwife

**Tagline:** Photograph a page of the paper maternal registry; get a checked, structured record that
follows the woman from visit to visit — offline, with the AI asking instead of guessing.

## Inspiration

In Moroccan health centres, midwives record pregnancy, delivery and postpartum care by hand in a pink
paper booklet. Once written, that information stays on paper: it does not follow the woman to her next
visit and never reaches the health system. Retyping it is slow, and connectivity is unreliable. We wanted
a digital layer that changes nothing in the midwife's practice: she keeps writing in the booklet, takes a
photo, and the record builds itself — with her in control of every uncertain value.

## What it does

* **Capture anywhere.** The midwife photographs a registry page in a WhatsApp-style chat on her phone.
  With no network the app still opens, and photos are kept encrypted on the phone; when the network
  returns they are sent for reading automatically, and deleted from the phone only once the server has them.
* **Reads the page.** The page type is recognised (cover, identification, obstetric history, current
  pregnancy, delivery, postpartum mother/newborn) and every field is extracted with a **value, a status
  and a confidence**. Statuses are explicit — KNOWN, UNKNOWN, NOT_PROVIDED, ILLEGIBLE, NOT_APPLICABLE,
  NEEDS_REVIEW — so a blank, a dash and an unreadable word are never confused.
* **Asks instead of guessing.** The chat only asks about uncertain fields: Confirm / Edit / Retake photo,
  follow-up questions, full manual entry when the AI is unavailable.
* **Links visits.** By the code the midwife writes on the booklet: exact and near matches are proposed
  (one misread digit would link the wrong woman), the midwife chooses; no patient is created
  automatically when a match is plausible.
* **Never loses a record.** Every record follows one state machine (captured → pending AI → processed →
  to review → validated → linked → saved → synced, plus failure states that always have a way out);
  sync waits for the network.
* **Privacy by design.** Names, ID numbers, phone numbers and addresses are never read or stored —
  they are painted out before any model sees the page. Photos are viewable only by the midwife and admin
  roles.

## How we built it

* **Extraction (Python, offline).** PP-OCRv6 recognisers on ONNX Runtime (CPU). For the organisers'
  specimen layout: alignment to templates built from the specimen PDF, ink detection, checkbox scoring.
  For the real pink booklet: a template-free reader that finds printed labels and table lines in tilted,
  curved phone photos, builds each cell, removes printed lines and print (by ink colour), and joins
  writing that runs diagonally across rows.
* **Field-aware reading.** Instead of the OCR's greedy guess, we score every *plausible* value of a field
  (a blood pressure in cmHg, a real date, a gestational age "16SA+3j", a clinical word) with the
  recogniser's own character probabilities, with weak priors from the organisers' synthetic data. Same
  model, better question: on the real photos this took us from 74 to 85 correct fields out of 127.
* **A fine-tuned vision model.** We fine-tuned Qwen3-VL-8B (QLoRA, Colab T4) to read one field at a time,
  first on field crops of the specimen pages (3,200 examples drawn from 10,000), then continued on a
  mix with 6,000 synthetic crops we generated in the real midwife's formats (cmHg pressures, "NF", "Reçu", French "1"). It plugs into the same pipeline as a GPU reader
  (`backend="vlm"`), running on the team's own laptop — no data goes to a cloud.
* **Offline-first phone app:** the chat is an installable web app; a service worker keeps it available
  offline, and photos wait in an encrypted outbox (AES-GCM, WebCrypto, key generated on the phone and
  not exportable) until they can be uploaded.
* **Processing server:** FastAPI backend with an encrypted record store and state machine (SQLite +
  Fernet). The phone is the only input device.
* **Honest evaluation.** Ground truth for 5,640 specimen fields extracted from the PDF itself; held-out
  patients; simulated phone photos (blur, shadow, perspective, JPEG); 127 fields of the real photos
  transcribed by hand; a crop editor to label crops and export a learning set.

## Results

| Test | Correct | Accepted but wrong |
|---|---|---|
| Held-out specimen pages, clean (1,692 fields) | 98.9% | 0.1% |
| Same, simulated phone photos | 82% (92% of written values) | 1.9% |
| Real booklet photos (127 fields) | 67% | 1 field |
| Vision model, held-out specimen crops | 95% | — |
| Vision model, real booklet crops (first version) | 50.5% exact (CPU OCR: 46%) | — |

On real cursive about half the fields go to the midwife for review — and almost none are accepted wrongly.
That trade-off is deliberate: a wrong blood pressure silently entered is worse than a question.

## Challenges we ran into

* **Real handwriting is not the specimen.** The organisers' pages use handwriting *fonts*; the real
  booklet has fast French cursive, a "1" that OCR reads as "Λ", cmHg pressures, words spilling over
  thin rows and "RAS" written diagonally across whole sections.
* **Small CPU.** Our development laptop had 6 GB of RAM and no GPU: memory leaks in the OCR runtime,
  models that did not fit, and GPU quota running out in Colab before the last benchmark.
* **Privacy versus training.** The brief forbids sending real patient data to a third party, so we
  trained only on synthetic data and built a generator for the midwife's formats instead of uploading
  her handwriting.

## Accomplishments that we're proud of

* An extractor that says when it is unsure: 1 silent error in 127 fields on real photos.
* Reading the real booklet without any template of it.
* The same OCR model reading much better simply by being asked "which plausible value is this?".
* A fine-tuned vision model that learned the registry (81% → 95% on held-out pages) from synthetic data only.

## What we learned

Start from the schema, not the OCR: knowing what each field can contain was worth more than a bigger
model. And treating "missing", "illegible" and "not applicable" as first-class states makes the
uncertainty visible instead of hiding it.

## What's next

* Benchmark the fine-tuned model on the real booklets on a GPU laptop; train it on real handwriting
  inside the health system, where the data may stay.
* Arabic handwriting; the real booklet's delivery and postpartum pages.
* A real WhatsApp Business sandbox; a native app with the key in the phone's secure keystore.

## Built with

Python · PP-OCRv6 (RapidOCR, ONNX Runtime) · OpenCV · Qwen3-VL · Unsloth / QLoRA · PyTorch · FastAPI ·
SQLite · cryptography (Fernet) · HTML/JavaScript · Service Worker · IndexedDB · WebCrypto · Google Colab
