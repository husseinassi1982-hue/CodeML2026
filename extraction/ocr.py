"""Local OCR engine (RapidOCR = PaddleOCR models on ONNX Runtime, CPU only, no network).

Two uses:
  * ``read_page``: detect + recognise every text line on the page, used to recognise the
    page type and align it with its template;
  * ``read_crops``: recognise a batch of tight, rectified field crops (one value each).
"""

from __future__ import annotations

import ctypes
import gc
import logging
from dataclasses import dataclass

import re

import numpy as np

_ENGINE = None
LATIN = re.compile(r"[^\u0000-\u024F\u2010-\u2027\u00B0\u2103]")


THREADS = 4


def _lean_session(model_path: str):
    """ONNX Runtime session that does not grow on a small laptop: RapidOCR's own sessions
    (all cores, one malloc arena each) grew ~50 MB per page and got OOM-killed on 6 GB."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.intra_op_num_threads = THREADS
    return ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])


def trim_memory() -> None:
    """Free the page's big arrays and hand the memory back to the OS.

    RapidOCR leaves reference cycles around its outputs; without an explicit collection
    Python 3.14 kept ~50 MB per page alive and a 30-page batch was OOM-killed."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def engine():
    global _ENGINE
    if _ENGINE is None:
        try:
            ctypes.CDLL("libc.so.6").mallopt(-8, 2)  # M_ARENA_MAX = 2
        except OSError:
            pass
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        from rapidocr import RapidOCR

        eng = RapidOCR()
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        for part in ("text_det", "text_cls", "text_rec"):
            wrapper = getattr(eng, part).session
            wrapper.session = _lean_session(wrapper.session._model_path)
        _ENGINE = eng
    return _ENGINE


@dataclass
class OcrLine:
    text: str
    score: float
    quad: np.ndarray  # 4x2 image pixels: top-left, top-right, bottom-right, bottom-left

    @property
    def left_mid(self):
        return (self.quad[0] + self.quad[3]) / 2

    @property
    def right_mid(self):
        return (self.quad[1] + self.quad[2]) / 2


def read_page(img: np.ndarray) -> list[OcrLine]:
    r = engine()(img, text_score=0.2)
    if r.boxes is None:
        return []
    return [OcrLine(t, float(s), np.asarray(b, dtype=float)) for b, t, s in zip(r.boxes, r.txts, r.scores)]


_MEDIUM = None
_MEDIUM_ERROR: Exception | None = None


def medium_engine():
    """Larger recogniser (PP-OCRv6 medium, ~73 MB), used as a second opinion on real handwriting.

    Not shipped in the rapidocr wheel: RapidOCR downloads it on first use. Offline before that
    download (`python -m extraction --download-models`), this returns None and callers fall
    back to the bundled small model, so extraction never needs the internet."""
    global _MEDIUM, _MEDIUM_ERROR
    if _MEDIUM is None and _MEDIUM_ERROR is None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import ModelType, OCRVersion

        engine()  # same allocator settings
        try:
            eng = RapidOCR(params={"Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.MEDIUM})
        except Exception as e:  # no network and not downloaded yet
            _MEDIUM_ERROR = e
            logging.getLogger(__name__).warning("medium OCR model unavailable, using the small one only: %s", e)
            return None
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        for part in ("text_det", "text_cls", "text_rec"):
            wrapper = getattr(eng, part).session
            wrapper.session = _lean_session(wrapper.session._model_path)
        _MEDIUM = eng
    return _MEDIUM


def medium_available() -> bool:
    return medium_engine() is not None


def read_crops(crops: list[np.ndarray], batch: int = 32, model: str = "small") -> list[tuple[str, float]]:
    from rapidocr.ch_ppocr_rec.typings import TextRecInput

    eng = engine() if model == "small" else (medium_engine() or engine())
    out: list[tuple[str, float]] = []
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch]
        r = eng.text_rec(TextRecInput(img=chunk))
        for t, sc in zip(r.txts, r.scores):
            latin = LATIN.sub("", t)  # the model also knows Chinese; the registry is Latin script
            out.append((latin.strip(), float(sc) if latin.strip() == t.strip() else float(sc) * 0.5))
    return out


def rec_probs(crops: list[np.ndarray], model: str = "medium") -> tuple[list[np.ndarray], list[str]]:
    """Per-timestep character probabilities of each crop (CTC output, before greedy decoding),
    and the recogniser's character list (index 0 = CTC blank). One crop at a time: no padding
    from other crops, so every column belongs to this crop."""
    eng = engine() if model == "small" else (medium_engine() or engine())
    rec = eng.text_rec
    _, h, w = rec.rec_image_shape[:3]
    out = []
    for crop in crops:
        ratio = max(w / h, crop.shape[1] / crop.shape[0])
        x = rec.resize_norm_img(crop, ratio)[np.newaxis].astype(np.float32)
        p = rec.session(x)[0]  # (T, C)
        used = min(p.shape[0], int(np.ceil(p.shape[0] * min(1.0, (crop.shape[1] / crop.shape[0]) / ratio))) + 2)
        out.append(np.asarray(p[:used], dtype=np.float32))
    return out, list(rec.postprocess_op.character)
