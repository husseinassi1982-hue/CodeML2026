"""Claude vision backend: higher accuracy on hand-held photos and layouts the local
templates do not know (e.g. the real pink booklet), at the cost of needing a network
connection and an API key.

Privacy: the brief forbids sending real patient data to a third party. This backend is
for the synthetic hackathon data, or for a deployment where the model runs under an
agreement with the health ministry. When the local aligner recognises the page, direct
identifiers (name, CIN, phone, address, husband's name) are blacked out on the copy that
is sent. Set DAYONE_CLOUD_REDACT_ONLY=1 to refuse to send any page that could not be
redacted.

Configuration: DAYONE_ALLOW_CLOUD=1 (explicit opt-in, synthetic data only), ANTHROPIC_API_KEY
(or `ant auth login`), optional DAYONE_CLAUDE_MODEL.
"""

from __future__ import annotations

import base64
import json
import os
import time

import cv2
import numpy as np

from .. import ocr, validate
from ..align import align
from ..catalog import PAGE_TYPES, Field, fields_for
from ..imageproc import image_quality
from ..normalize import parse
from ..redact import redact_identifiers
from ..schema import PageExtraction, Status
from ..status import KNOWN_THRESHOLD
from .common import field_result, finish

MODEL = os.environ.get("DAYONE_CLAUDE_MODEL", "claude-opus-5-5")
VERSION = f"claude:{MODEL}"
MAX_SIDE = 2400  # keep handwriting legible in the dense visit table
PAGE_STATUSES = ["KNOWN", "UNKNOWN", "NOT_PROVIDED", "ILLEGIBLE", "NOT_APPLICABLE"]

SYSTEM = """You digitise pages of a French maternal-health paper registry (Moroccan "Fiche de \
surveillance de la grossesse et du post-partum"). You transcribe exactly what is written; you never \
guess, infer or correct clinical content.

Rules:
- Transcribe handwriting as written (keep units such as "SA", "cm", "g/dL"). Do not convert.
- Status per field: KNOWN (you can read it), NOT_PROVIDED (left blank), ILLEGIBLE (something is \
written but you cannot read it), NOT_APPLICABLE (a dash "—" is written), UNKNOWN (the writer \
wrote "?", "inconnu" or similar).
- confidence: your probability (0 to 1) that the transcription is exactly right. Be honest: a \
midwife reviews everything below 0.8, so low confidence is useful, not a failure.
- Checkboxes: text "checked" or "unchecked". Option groups: text is the ticked option's value \
from the list, or null with NOT_PROVIDED if none is ticked.
- Never transcribe names of people, national ID (CIN) numbers, phone numbers or addresses; \
those fields are not in the list and must not appear anywhere in your answer."""


class ClaudeBackend:
    name = "claude"

    def __init__(self):
        if os.environ.get("DAYONE_ALLOW_CLOUD") != "1":
            raise RuntimeError(
                "cloud backend disabled: the brief forbids sending real patient data to a third-party "
                "service. Set DAYONE_ALLOW_CLOUD=1 only for the organisers' synthetic data.")
        import anthropic

        self.client = anthropic.Anthropic()

    # --- public ------------------------------------------------------------------

    def extract(self, img: np.ndarray, page_type_hint: str | None = None) -> PageExtraction:
        t0 = time.time()
        quality = image_quality(img)
        warnings: list[str] = []

        # Local alignment (offline, ~2 s) tells us the page type and lets us mask identifiers.
        al = align(ocr.read_page(img), img)
        ocr.trim_memory()
        send = img
        if al is not None:
            send = redact_identifiers(img, al.page_type, al.H)
        elif os.environ.get("DAYONE_CLOUD_REDACT_ONLY") == "1":
            raise RuntimeError("page not recognised locally, so identifiers cannot be masked; not sending it")
        else:
            warnings.append("identifiers could not be masked before sending (page not recognised locally)")

        page_type = page_type_hint or (al.page_type if al is not None else self._classify(send))
        if page_type not in PAGE_TYPES:
            return finish("unknown", 0.0, {}, quality, self.name, VERSION, int((time.time() - t0) * 1000),
                          warnings + ["page not recognised as a registry page"])

        items = self._extract_fields(send, page_type)
        fields = {}
        for f in fields_for(page_type):
            fields[f.key] = _to_result(f, items.get(f.key))
        validate.check(page_type, fields)
        page_conf = al.confidence if (al is not None and al.page_type == page_type) else 0.8
        return finish(page_type, page_conf, fields, quality, self.name, VERSION, int((time.time() - t0) * 1000),
                      warnings + [f"image quality: {i}" for i in quality.issues])

    # --- calls ------------------------------------------------------------------

    def _call(self, img: np.ndarray, prompt: str, schema: dict, max_tokens: int) -> dict:
        h, w = img.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        data = base64.standard_b64encode(buf.tobytes()).decode()
        with self.client.beta.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if the request is declined, the API re-runs it on a suitable model
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": prompt},
            ]}],
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError("the model declined to process this page")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("model output was cut off (max_tokens)")
        text = next(b.text for b in msg.content if b.type == "text")
        return json.loads(text)

    def _classify(self, img: np.ndarray) -> str:
        schema = {"type": "object", "additionalProperties": False, "required": ["page_type"],
                  "properties": {"page_type": {"type": "string", "enum": list(PAGE_TYPES) + ["unknown"]}}}
        listing = "\n".join(f"- {k}: {v}" for k, v in PAGE_TYPES.items())
        out = self._call(img, f"Which registry page is this?\n{listing}\n- unknown: anything else", schema, 2000)
        return out["page_type"]

    def _extract_fields(self, img: np.ndarray, page_type: str) -> dict[str, dict]:
        fs = fields_for(page_type)
        schema = build_schema(fs)
        prompt = (f"This is the page « {PAGE_TYPES[page_type]} ». Return one item per field below, in order.\n"
                  "key | printed label on the form | kind | expected content\n" + field_listing(fs))
        out = self._call(img, prompt, schema, max_tokens=64000)
        return {it["key"]: it for it in out.get("fields", [])}


# --- helpers (no network; unit-testable) ---------------------------------------------


def field_listing(fs: list[Field]) -> str:
    rows = []
    for f in fs:
        if f.kind == "radio":
            what = "one of: " + ", ".join(f"{o.value} (« {o.anchor} »)" for o in f.options)
        elif f.kind == "check":
            what = "checkbox"
        else:
            what = f.dtype
        rows.append(f"{f.key} | {f.label_fr} | {f.kind} | {what}")
    return "\n".join(rows)


def build_schema(fs: list[Field]) -> dict:
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["key", "status", "text", "confidence"],
        "properties": {
            "key": {"type": "string", "enum": [f.key for f in fs]},
            "status": {"type": "string", "enum": PAGE_STATUSES},
            "text": {"type": ["string", "null"]},
            "confidence": {"type": "number"},
        },
    }
    return {"type": "object", "additionalProperties": False, "required": ["fields"],
            "properties": {"fields": {"type": "array", "items": item}}}


def _to_result(f: Field, it: dict | None):
    if it is None:
        return field_result(f, Status.NEEDS_REVIEW, 0.0, issues=["not returned by the model"])
    status = Status(it["status"]) if it["status"] in PAGE_STATUSES else Status.NEEDS_REVIEW
    conf = max(0.0, min(1.0, float(it.get("confidence") or 0.0)))
    text = it.get("text")

    if f.kind == "check":
        if status != Status.KNOWN or text not in ("checked", "unchecked"):
            return field_result(f, Status.NEEDS_REVIEW, min(conf, 0.5), issues=["unclear whether the box is ticked"])
        ticked = text == "checked"
        st = Status.KNOWN if conf >= KNOWN_THRESHOLD else Status.NEEDS_REVIEW
        return field_result(f, st, conf, value=ticked, display="☒" if ticked else "☐")

    if f.kind == "radio":
        values = [o.value for o in f.options]
        if status == Status.NOT_PROVIDED or text is None:
            return field_result(f, Status.NOT_PROVIDED, conf)
        if text not in values:
            return field_result(f, Status.NEEDS_REVIEW, min(conf, 0.5), raw=text, issues=["unexpected option"])
        st = Status.KNOWN if conf >= KNOWN_THRESHOLD else Status.NEEDS_REVIEW
        return field_result(f, st, conf, value=text, display=text)

    if status in (Status.NOT_PROVIDED, Status.NOT_APPLICABLE, Status.UNKNOWN, Status.ILLEGIBLE):
        disp = {"NOT_APPLICABLE": "—", "UNKNOWN": "?"}.get(status.value)
        return field_result(f, status, conf, display=disp, raw=text)
    p = parse(text, f)
    if not p.ok or p.issues:
        return field_result(f, Status.NEEDS_REVIEW, min(conf, 0.5), value=p.value, display=p.display, raw=text,
                            issues=p.issues or ["could not interpret the value"])
    st = Status.KNOWN if conf >= KNOWN_THRESHOLD else Status.NEEDS_REVIEW
    return field_result(f, st, conf, value=p.value, display=p.display, raw=text,
                        issues=[] if st == Status.KNOWN else ["model unsure"])
