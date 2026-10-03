"""Mask direct identifiers (name, CIN, phone, address, husband's name) on a page photo.

Use it to store a shareable copy of each capture next to the restricted original, and
before any image leaves the device. The input image is never modified.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import ocr
from .align import align
from .geometry import apply, rect_corners
from .imageproc import load_image
from .template import load_templates


def redact_identifiers(img: np.ndarray, page_type: str, H: np.ndarray, pad: float = 2.0) -> np.ndarray:
    out = img.copy()
    for geo in load_templates()[page_type].fields.values():
        if geo.get("pii") and "region" in geo:
            x0, y0, x1, y1 = geo["region"]
            poly = apply(H, rect_corners((x0 - pad, y0 - pad, x1 + pad, y1 + pad))).astype(np.int32)
            cv2.fillPoly(out, [poly], (0, 0, 0))
    return out


def redact_photo(image) -> tuple[np.ndarray | None, str | None]:
    """Photo (path/bytes/array) -> (masked copy, page type). (None, None) if the page is not
    recognised: then the caller must not share the image."""
    img = load_image(image)
    al = align(ocr.read_page(img), img)
    ocr.trim_memory()
    if al is None:
        return None, None
    return redact_identifiers(img, al.page_type, al.H), al.page_type
