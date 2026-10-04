"""Optional second reader: the fine-tuned vision-language model (Qwen3-VL-8B + LoRA adapter).

The CPU pipeline still does everything else (page type, fields, identifiers, statuses); this reads
each field's crop again with the trained model and merges the two readings:

* both agree                      -> KNOWN, high confidence
* they disagree                   -> NEEDS_REVIEW (the model's reading pre-filled when the CPU was unsure;
                                     the CPU's kept, with the alternative noted, when it was sure)
* the model sees EMPTY, the CPU read something -> NEEDS_REVIEW

Needs a CUDA GPU (~6-7 GB: Colab/Kaggle T4, RTX 4060 laptop) and the adapter (`vlm_ocr_lora_v2.zip`,
GitHub release of this repo, or notebooks/finetune_vlm_ocr.ipynb). Enabled with DAYONE_VLM=1; the
adapter folder or zip is DAYONE_VLM_ADAPTER (default: ./vlm_ocr_lora_v2[.zip] or models/...). Without a
GPU or the adapter the extractor runs exactly as before (CPU only). Runs locally: no data leaves the
machine the model runs on.

The prompt and crop margins are the ones the model was trained with (tools/export_vlm_dataset.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

EMPTY = "EMPTY"
HINTS = {
    "bp": "a blood pressure, written like 11/7 (cmHg) or 110/70 (mmHg)",
    "date": "a date, written like 13/05/25 or 13/05/2025",
    "date_or_year": "a date or a year",
    "weeks": "a gestational age in weeks, written like 31 SA or 16SA+3j",
    "kg": "a weight in kg, written like 62 or 62 kg (NF = not done)",
    "int": "a whole number",
    "float": "a number, written like 11,1 or 11.1",
    "cm": "a length in cm",
    "yesno": "yes/no, written like Oui, Non, +, 0",
    "posneg": "a test result, written like Neg, nég, +, -, 0",
    "code": "the patient's record number (digits)",
    "str": "a short French clinical note (e.g. RAS, colorées, Normal, Fermé)",
    "enum": "one of a few written options",
}
MARGIN_Y = 0.6  # crop margins around the field's area, in printed-line heights (as in training)
MARGIN_X = 0.3
LINE_PT = 10.0  # printed line height on the specimen layout, in PDF points
SCALE = 3.0  # px per point for specimen crops
MAX_SIDE = 768
REPO = Path(__file__).resolve().parent.parent


def prompt(label_fr: str, label_en: str, dtype: str) -> str:
    """What the model is asked for one field crop (identical to training)."""
    hint = HINTS.get(dtype, "a short handwritten value")
    return (f"This image is one field of a Moroccan maternal-health registry (pink booklet), "
            f"filled in by hand by a midwife, in French. Field: «{label_fr}» ({label_en}). "
            f"It usually contains {hint}. Neighbouring rows may show at the edges: read only the "
            f"value of this field. Transcribe exactly what is handwritten, keeping the midwife's "
            f"abbreviations. If nothing is handwritten, answer {EMPTY}. If a dash is written, answer —. "
            f"Answer with the value only.")


# --- model ------------------------------------------------------------------------------------------

_MODEL = None
_FAILED: str | None = None


def wanted() -> bool:
    return os.environ.get("DAYONE_VLM", "0") == "1"


def _adapter_dir() -> Path | None:
    env = os.environ.get("DAYONE_VLM_ADAPTER")
    cands = [Path(env)] if env else []
    cands += [REPO / "vlm_ocr_lora_v2", REPO / "vlm_ocr_lora_v2.zip", REPO / "models" / "vlm_ocr_lora_v2",
              REPO / "models" / "vlm_ocr_lora_v2.zip", Path("vlm_ocr_lora_v2"), Path("vlm_ocr_lora_v2.zip")]
    for c in cands:
        if c.is_dir() and (c / "adapter_model.safetensors").exists():
            return c
        if c.suffix == ".zip" and c.exists():
            with zipfile.ZipFile(c) as z:
                top = z.namelist()[0].split("/")[0]
                z.extractall(c.parent)
            d = c.parent / top
            if (d / "adapter_model.safetensors").exists():
                return d
    return None


def available() -> bool:
    """Requested, GPU present, adapter found, packages installed (loads the model once)."""
    return wanted() and _load() is not None


def _load():
    global _MODEL, _FAILED
    if _MODEL is not None or _FAILED is not None:
        return _MODEL
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU")
        adapter = _adapter_dir()
        if adapter is None:
            raise RuntimeError("adapter vlm_ocr_lora_v2 not found (set DAYONE_VLM_ADAPTER)")
        info = json.loads((adapter / "training_info.json").read_text()) if (adapter / "training_info.json").exists() else {}
        base = info.get("base_model", "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
        rank = int(info.get("lora_rank", 16))
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from unsloth import FastVisionModel

        # the same way the adapter was trained and re-loaded in the notebook
        model, tok = FastVisionModel.from_pretrained(base, load_in_4bit=True)
        model = FastVisionModel.get_peft_model(model, finetune_vision_layers=True, finetune_language_layers=True,
                                               finetune_attention_modules=True, finetune_mlp_modules=True,
                                               r=rank, lora_alpha=rank, lora_dropout=0, bias="none", random_state=3407)
        set_peft_model_state_dict(model, load_file(str(adapter / "adapter_model.safetensors")))
        FastVisionModel.for_inference(model)
        _MODEL = (model, tok, int(info.get("max_side", MAX_SIDE)))
        log.info("vision model loaded: %s + %s", base, adapter)
    except Exception as e:  # no GPU, no adapter, packages missing: CPU pipeline only
        _FAILED = str(e)
        log.warning("vision model not used (%s): CPU OCR only", e)
    return _MODEL


def load_or_raise():
    """For the VLM-only backend: the model, or an error explaining why it is not available."""
    global _FAILED
    _FAILED = None if _MODEL is None else _FAILED
    if _load() is None:
        raise RuntimeError(f"vision model not available: {_FAILED}")
    return _MODEL


def ask_scored(img_rgb: np.ndarray, text: str) -> tuple[str, float]:
    """(answer, confidence): confidence = probability of the least certain generated token."""
    import math

    import torch
    from PIL import Image

    model, tok, max_side = _MODEL
    im = Image.fromarray(img_rgb)
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = tok.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = tok(im, chat, add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, use_cache=True, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0][inputs["input_ids"].shape[1]:]
    logps = [torch.log_softmax(sc[0].float(), -1)[t].item() for sc, t in zip(out.scores, seq)]
    answer = tok.decode(seq, skip_special_tokens=True).strip()
    return answer, (math.exp(min(logps)) if logps else 0.0)


def decide(f, answer: str, conf: float, threshold: float, field_result):
    """FieldResult for one field read by the vision model only."""
    from .normalize import parse
    from .schema import Status

    a = (answer or "").strip()
    if not a or a.upper() == EMPTY:
        return field_result(f, Status.NOT_PROVIDED, conf)
    if a in ("—", "-", "–"):
        return field_result(f, Status.NOT_APPLICABLE, conf, display="—", raw=a)
    p = parse(a, f)
    if p.marker == "dash":
        return field_result(f, Status.NOT_APPLICABLE, conf, display="—", raw=a)
    if p.marker == "not_done":
        return field_result(f, Status.NOT_APPLICABLE, conf, display=p.display, raw=a)
    if p.marker == "unknown":
        return field_result(f, Status.UNKNOWN, conf, display="?", raw=a)
    if not p.ok:
        return field_result(f, Status.NEEDS_REVIEW, min(conf, 0.45), raw=a,
                            issues=list(p.issues) or ["could not interpret the reading"])
    if p.issues:
        return field_result(f, Status.NEEDS_REVIEW, min(conf, 0.5), value=p.value, display=p.display, raw=a,
                            issues=list(p.issues))
    if conf < threshold:
        return field_result(f, Status.NEEDS_REVIEW, conf, value=p.value, display=p.display, raw=a,
                            issues=["low reading confidence"])
    return field_result(f, Status.KNOWN, conf, value=p.value, display=p.display, raw=a)


def read_only(fields: dict, crops: dict[str, np.ndarray], page_type: str, threshold: float, field_result) -> None:
    """VLM-only mode: every field with ink (per the CPU's ink detection) is read by the model."""
    from .catalog import get_field
    from .schema import Status

    for k, crop in crops.items():
        g = fields.get(k)
        if g is not None and g.status == Status.NOT_PROVIDED and g.confidence >= 0.9:
            continue  # clearly blank paper: nothing to read (faint or doubtful ink is still read)
        try:
            f = get_field(page_type, k)
        except KeyError:
            continue
        if f.pii:
            continue
        answer, conf = ask_scored(crop, prompt(f.label_fr, f.label_en, f.dtype))
        fields[k] = decide(f, answer, conf, threshold, field_result)


def ask(img_rgb: np.ndarray, text: str) -> str:
    import torch
    from PIL import Image

    model, tok, max_side = _MODEL
    im = Image.fromarray(img_rgb)
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = tok.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = tok(im, chat, add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, use_cache=True, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# --- merging the two readings ---------------------------------------------------------------------

def _same(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    norm = lambda t: re.sub(r"[\s.]", "", str(t)).lower()
    return norm(a) == norm(b)


def merge(fields: dict, answers: dict[str, str], threshold: float, page_type: str) -> int:
    """Combine the CPU result (FieldResult per key) with the model's readings. Returns #fields changed."""
    from .catalog import get_field
    from .normalize import parse
    from .schema import Status

    changed = 0
    for key, ans in answers.items():
        g = fields.get(key)
        if g is None:
            continue
        upd = None
        if ans.strip().upper() == EMPTY:
            if g.status in (Status.KNOWN, Status.NEEDS_REVIEW) and g.value not in (None, ""):
                upd = dict(status=Status.NEEDS_REVIEW, confidence=min(g.confidence, 0.5),
                           issues=g.issues + ["the vision model sees no handwriting here"])
        else:
            try:
                f = get_field(page_type, key)
            except KeyError:
                continue
            p = parse(ans, f)
            agree_marker = p.marker in ("dash", "not_done") and g.status == Status.NOT_APPLICABLE
            if agree_marker:
                upd = dict(confidence=max(g.confidence, 0.9))
            elif not p.ok:
                if g.status == Status.KNOWN:
                    upd = dict(status=Status.NEEDS_REVIEW, confidence=min(g.confidence, 0.6),
                               issues=g.issues + [f"the vision model reads «{ans}»"])
            elif _same(p.value, g.value):
                if g.status in (Status.KNOWN, Status.NEEDS_REVIEW) and not [i for i in g.issues if "range" in i or "check" in i]:
                    upd = dict(status=Status.KNOWN, confidence=max(g.confidence, min(0.97, threshold + 0.07)),
                               issues=[])
            elif g.status == Status.KNOWN:
                upd = dict(status=Status.NEEDS_REVIEW, confidence=min(g.confidence, 0.6),
                           issues=g.issues + [f"the vision model reads «{ans}»"])
            else:  # the CPU was unsure or saw nothing: the model's reading, to confirm
                upd = dict(status=Status.NEEDS_REVIEW, value=p.value, display=p.display, raw=ans,
                           confidence=min(0.75, threshold - 0.05),
                           issues=["read by the vision model: please confirm"])
        if upd:
            fields[key] = g.model_copy(update=upd)
            changed += 1
    return changed


# --- field crops (as in training) -------------------------------------------------------------------

def specimen_crops(img: np.ndarray, H, page_type: str, keys: list[str]) -> dict[str, np.ndarray]:
    """Specimen layout: each field's template area plus margins, identifiers painted out."""
    import cv2

    from .imageproc import warp_region
    from .template import load_templates

    tpl = load_templates()[page_type]
    masked = img.copy()
    for g in tpl.fields.values():
        if g.get("pii"):
            x0, y0, x1, y1 = g["region"]
            q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64).reshape(-1, 1, 2)
            cv2.fillPoly(masked, [cv2.perspectiveTransform(q, H).reshape(-1, 2).astype(np.int32)], (0, 0, 0))
    out = {}
    for k in keys:
        g = tpl.fields.get(k)
        if g is None:
            continue
        x0, y0, x1, y1 = g["region"]
        rect = (x0 - MARGIN_X * LINE_PT, y0 - MARGIN_Y * LINE_PT, x1 + MARGIN_X * LINE_PT, y1 + MARGIN_Y * LINE_PT)
        out[k] = cv2.cvtColor(warp_region(masked, H, rect, scale=SCALE), cv2.COLOR_BGR2RGB)
    return out


def booklet_crops(page, loc, keys: list[str]) -> dict[str, np.ndarray]:
    """Real booklet: each located field's area plus margins, identifiers painted out."""
    import cv2

    masked = page.img.copy()
    for x0, y0, x1, y1 in loc.pii:
        masked[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = (0, 0, 0)
    h, w = masked.shape[:2]
    regions = {r.key: r for r in loc.regions}
    out = {}
    for k in keys:
        r = regions.get(k)
        if r is None or r.rect[2] <= r.rect[0]:
            continue
        x0, y0, x1, y1 = r.rect
        mx, my = int(MARGIN_X * page.th), int(MARGIN_Y * page.th)
        c = masked[max(0, y0 - my):min(h, y1 + my), max(0, x0 - mx):min(w, x1 + mx)]
        if c.size:
            out[k] = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    return out


def read_and_merge(fields: dict, crops: dict[str, np.ndarray], page_type: str, threshold: float) -> int:
    """Ask the model about every field with a crop whose CPU result is not a confident empty."""
    from .catalog import get_field
    from .schema import Status

    answers = {}
    for k, crop in crops.items():
        g = fields.get(k)
        if g is None or (g.status == Status.NOT_PROVIDED and g.confidence >= 0.9):
            continue  # clearly blank paper: no need to ask
        try:
            f = get_field(page_type, k)
        except KeyError:
            continue
        if f.pii:
            continue
        answers[k] = ask(crop, prompt(f.label_fr, f.label_en, f.dtype))
    return merge(fields, answers, threshold, page_type)
