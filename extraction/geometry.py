"""Small geometry helpers shared by the template, ground-truth and OCR alignment code."""

from __future__ import annotations

import numpy as np


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2x3 affine mapping src points -> dst points."""
    A = np.c_[src, np.ones(len(src))]
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return M.T  # 2x3


def to_3x3(M: np.ndarray) -> np.ndarray:
    if M.shape == (3, 3):
        return M
    return np.vstack([M, [0, 0, 1]])


def apply(M: np.ndarray, pts) -> np.ndarray:
    """Apply a 2x3 affine or 3x3 homography to an (N, 2) array of points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    H = to_3x3(M)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def invert(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(to_3x3(M))


def rect_corners(r) -> np.ndarray:
    x0, y0, x1, y1 = r
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def map_rect(M: np.ndarray, r) -> list[float]:
    """Axis-aligned bounding box of a rectangle after transformation."""
    p = apply(M, rect_corners(r))
    return [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]


def contains(r, x, y) -> bool:
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]


def intersects(a, b, pad=0.0) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])
