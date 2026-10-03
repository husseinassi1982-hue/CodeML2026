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


def read_crops(crops: list[np.ndarray], batch: int = 32) -> list[tuple[str, float]]:
    from rapidocr.ch_ppocr_rec.typings import TextRecInput

    out: list[tuple[str, float]] = []
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch]
        r = engine().text_rec(TextRecInput(img=chunk))
        for t, sc in zip(r.txts, r.scores):
            latin = LATIN.sub("", t)  # the model also knows Chinese; the registry is Latin script
            out.append((latin.strip(), float(sc) if latin.strip() == t.strip() else float(sc) * 0.5))
    return out
