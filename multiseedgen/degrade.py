from __future__ import annotations
"""Camera-matched photometric degradation (noise/JPEG/blur/vignette)."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .models import Config




# --------------------------------------------------------------------------- #
# Camera-matched degradation helpers                                          #
# --------------------------------------------------------------------------- #
def _jpeg_quality(cfg: Config, rng: np.random.Generator) -> int:
    """Per-scene JPEG save quality. Draws from rng ONLY when a range is set, so the
    default (min==max==95) consumes no draw and preserves determinism/draw order."""
    lo, hi = int(cfg.jpeg_quality_min), int(cfg.jpeg_quality_max)
    if hi <= lo:
        return lo
    return int(rng.integers(lo, hi + 1))



def _motion_blur(img: np.ndarray, length: int, rng) -> np.ndarray:
    length = max(1, int(length))
    if length < 2:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    ang = float(rng.uniform(0, 180))
    m = cv2.getRotationMatrix2D((length / 2, length / 2), ang, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    s = k.sum()
    if s > 1e-6:
        k = k / s
    return cv2.filter2D(img, -1, k)



def _apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    fall = np.clip(1.0 - strength * (r ** 2), 0.0, 1.0)[..., None]
    return img * fall



def apply_degradation(canvas_f32: np.ndarray, cfg: Config,
                      rng: np.random.Generator) -> np.ndarray:
    """Camera-matched PHOTOMETRIC degradation on the final composite (per scene).
    These never move boxes, so labels stay correct. JPEG recompression is handled
    at save time via _jpeg_quality; geometric perspective (which WOULD move boxes)
    is intentionally not applied here."""
    img = canvas_f32
    if cfg.vignette > 0:
        img = _apply_vignette(img, cfg.vignette)
    if cfg.blur_max_sigma > 0:
        s = float(rng.uniform(0, cfg.blur_max_sigma))
        if s > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), s)
    if cfg.motion_blur_max > 0:
        img = _motion_blur(img, int(rng.integers(1, cfg.motion_blur_max + 1)), rng)
    if cfg.poisson_noise:
        scaled = np.clip(img, 0, None)
        img = rng.poisson(scaled).astype(np.float32)
    if cfg.noise_sigma > 0:
        img = img + rng.normal(0, cfg.noise_sigma, img.shape)
    return img
