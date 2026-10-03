"""Image utilities: loading, quality check, rectified crops, ink and checkbox detection."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .schema import ImageQuality

CROP_SCALE = 4.0  # pixels per PDF point in rectified crops (~290 dpi)


class ImageError(ValueError):
    pass


def load_image(src) -> np.ndarray:
    """Path / bytes / PIL image / numpy array -> BGR uint8 array, EXIF rotation applied."""
    if isinstance(src, np.ndarray):
        img = src
    else:
        try:
            if isinstance(src, (str, Path)):
                pil = Image.open(src)
            elif isinstance(src, (bytes, bytearray)):
                pil = Image.open(io.BytesIO(src))
            elif isinstance(src, Image.Image):
                pil = src
            else:
                raise ImageError(f"unsupported image input: {type(src)}")
            pil = ImageOps.exif_transpose(pil).convert("RGB")
        except (OSError, Image.UnidentifiedImageError) as e:
            raise ImageError(f"cannot open image: {e}") from e
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[0] < 200 or img.shape[1] < 200:
        raise ImageError("image too small")
    return img


def image_quality(img: np.ndarray) -> ImageQuality:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 1000 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    bright = float(small.mean() / 255)
    issues = []
    if sharp < 60:
        issues.append("blurry")
    if bright < 0.25:
        issues.append("too dark")
    if bright > 0.97:
        issues.append("overexposed")
    if min(img.shape[:2]) < 800:
        issues.append("low resolution")
    return ImageQuality(ok=not issues, sharpness=round(sharp, 1), brightness=round(bright, 3), issues=issues)


def warp_region(img: np.ndarray, H: np.ndarray, rect, scale: float = CROP_SCALE) -> np.ndarray:
    """Rectified crop of a template-coordinate rectangle (points) from the photo.

    H maps template points -> image pixels; the crop has `scale` pixels per point."""
    x0, y0, x1, y1 = rect
    w, h = max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))
    T = np.array([[1 / scale, 0, x0], [0, 1 / scale, y0], [0, 0, 1]], dtype=float)
    M = H @ T  # crop pixel -> image pixel
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def _ink(gray: np.ndarray) -> np.ndarray:
    """Binary mask of dark strokes, robust to uneven lighting (divide by local background)."""
    bg = cv2.dilate(gray, np.ones((15, 15), np.uint8))  # wider than any pen stroke
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    norm = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1)
    return (norm < 0.72).astype(np.uint8)


def _remove_lines(mask: np.ndarray, h_len: int, v_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove long horizontal/vertical straight lines (underlines, table grid, box borders)."""
    lines = np.zeros_like(mask)
    if h_len > 0 and mask.shape[1] > h_len:
        lines |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, h_len), np.uint8))
    if v_len > 0 and mask.shape[0] > v_len:
        lines |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((v_len, 1), np.uint8))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return mask & (1 - lines), lines


@dataclass
class Ink:
    present: bool
    fraction: float
    bbox: tuple[int, int, int, int] | None  # x0, y0, x1, y1 in crop pixels
    is_dash: bool
    clean: np.ndarray  # crop with printed lines painted out, for OCR


def analyze_text_crop(crop: np.ndarray, scale: float = CROP_SCALE, printed: np.ndarray | None = None) -> Ink:
    """printed: the template's printed ink for the same region (same size as crop), if known."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _ink(gray)
    if printed is not None and printed.shape == mask.shape:
        # widen by ~1 pt to absorb residual misalignment and blur, then drop it
        grown = cv2.dilate(printed, np.ones((int(2 * scale) | 1, int(2 * scale) | 1), np.uint8))
        mask = mask & (1 - grown)
    mask, lines = _remove_lines(mask, h_len=int(10 * scale), v_len=int(0.75 * gray.shape[0]))
    if printed is not None and printed.shape == mask.shape:
        lines = lines | grown
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape
    edge = int(0.8 * scale)

    def is_remnant(i):  # leftover piece of a grid line / underline hugging the crop border
        x, y, w, h = (stats[i, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        near_tb = y <= edge or y + h >= H - edge
        near_lr = x <= edge or x + w >= W - edge
        return (near_tb and h <= 2.0 * scale and w >= 1.5 * h) or (near_lr and w <= 2.0 * scale and h >= 1.5 * w)

    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(6, int(0.5 * scale * scale))
            and not is_remnant(i)]
    clean = crop.copy()
    bgval = np.median(gray[mask == 0]) if (mask == 0).any() else 255
    clean[lines.astype(bool)] = bgval
    if not keep:
        return Ink(False, 0.0, None, False, clean)
    xs0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    ys0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    xs1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    ys1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    area = sum(stats[i, cv2.CC_STAT_AREA] for i in keep)
    frac = area / mask.size
    h, w = ys1 - ys0, xs1 - xs0
    # a handwritten "—": one or two thin horizontal strokes, little ink
    mid = (ys0 + ys1) / 2
    is_dash = (len(keep) <= 2 and h <= 2.6 * scale and w >= 2.5 * scale and w > 2.5 * h
               and area >= 1.0 * scale * scale and 0.2 * H < mid < 0.8 * H)
    present = is_dash or area >= 3 * scale * scale  # at least ~3 pt² of ink
    return Ink(present, float(frac), (xs0, ys0, xs1, ys1), bool(is_dash), clean)


def text_crop_for_ocr(ink: Ink, scale: float = CROP_SCALE) -> np.ndarray:
    x0, y0, x1, y1 = ink.bbox
    pad = int(2 * scale)
    h, w = ink.clean.shape[:2]
    c = ink.clean[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
    return cv2.copyMakeBorder(c, 6, 6, 10, 10, cv2.BORDER_REPLICATE)


def checkbox_score(img: np.ndarray, H: np.ndarray, box, scale: float = CROP_SCALE) -> float:
    """Fraction of the box interior covered by ink (tick, cross, hatching, filling).

    The printed square is located precisely (hollow-square template match within ±2 pt of
    where alignment puts it), then ink is measured strictly inside it. No line removal:
    some midwives hatch boxes with horizontal strokes."""
    m = 3.0
    rect = (box[0] - m, box[1] - m, box[2] + m, box[3] + m)
    crop = warp_region(img, H, rect, scale)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _ink(gray).astype(np.float32)
    bw, bh = int(round((box[2] - box[0]) * scale)), int(round((box[3] - box[1]) * scale))
    t = max(2, int(0.8 * scale))
    square = np.zeros((bh + t, bw + t), np.float32)
    cv2.rectangle(square, (t // 2, t // 2), (bw + t // 2 - 1, bh + t // 2 - 1), 1.0, t)
    if mask.shape[0] < square.shape[0] or mask.shape[1] < square.shape[1]:
        return 0.0
    res = cv2.matchTemplate(mask, square, cv2.TM_CCORR)
    expect = int(round(m * scale - t / 2))
    r = int(2 * scale)
    y0, x0 = max(0, expect - r), max(0, expect - r)
    win = res[y0:expect + r + 1, x0:expect + r + 1]
    if win.size and win.max() > 0.5 * square.sum():
        dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
        oy, ox = y0 + dy + t // 2, x0 + dx + t // 2
    else:  # square not found (faded print): trust the alignment
        oy = ox = int(round(m * scale))
    inset = int(round(1.3 * scale))
    roi = mask[oy + inset:oy + bh - inset, ox + inset:ox + bw - inset]
    return float(roi.mean()) if roi.size else 0.0


def ink_bbox(crop: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight bbox (crop pixels) of the ink in a small crop, ignoring lines that cross it."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mask = _ink(gray)
    mask, _ = _remove_lines(mask, h_len=int(gray.shape[1] * 0.9), v_len=int(gray.shape[0] * 0.9))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if not keep:
        return None
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    return x0, y0, x1, y1
