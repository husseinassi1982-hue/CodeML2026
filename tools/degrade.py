"""Simulate field photos of a registry page: perspective, rotation, blur, shadow, low light,
sensor noise, JPEG compression. Deterministic for a given seed.

    python -m tools.degrade IN.png OUT.jpg --level medium --seed 3
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

LEVELS = {
    #          rot°  persp  blurσ  shadow  dark  noise  jpegQ  scale
    "mild":   (2.0, 0.015, 0.8, 0.25, 0.15, 3, 75, 0.9),
    "medium": (4.0, 0.03, 1.3, 0.45, 0.30, 6, 55, 0.75),
    "hard":   (7.0, 0.05, 2.0, 0.60, 0.45, 10, 35, 0.6),
}


def degrade(img: np.ndarray, level: str = "medium", seed: int = 0) -> np.ndarray:
    rot, persp, blur, shadow, dark, noise, q, scale = LEVELS[level]
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    # photographed on a table: page inside a darker border, tilted, slight perspective
    border = int(0.06 * max(h, w))
    canvas = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=(70, 60, 55))
    H, W = canvas.shape[:2]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    jitter = rng.uniform(-persp, persp, size=(4, 2)) * [W, H]
    M = cv2.getPerspectiveTransform(src, np.float32(src + jitter))
    R = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-rot, rot), 1.0)
    M = np.vstack([R, [0, 0, 1]]) @ M
    out = cv2.warpPerspective(canvas, M, (W, H), borderValue=(70, 60, 55))

    # uneven lighting: a linear shadow across the page plus overall darkening
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    light = (1 - shadow * grad) * (1 - dark * rng.uniform(0.5, 1.0))
    out = np.clip(out.astype(np.float32) * light[..., None], 0, 255)

    out = cv2.GaussianBlur(out, (0, 0), blur * rng.uniform(0.7, 1.0))
    out = out + rng.normal(0, noise, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    s = scale * rng.uniform(0.9, 1.0)
    out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--level", default="medium", choices=list(LEVELS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cv2.imwrite(a.dst, degrade(cv2.imread(a.src), a.level, a.seed))


if __name__ == "__main__":
    main()
