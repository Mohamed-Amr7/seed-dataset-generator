"""Per-seed transforms: resize, flip, rotate, shear, HSV jitter."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import Config, Seed


def _resize_to_long_side(rgba: np.ndarray, long_px: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = long_px / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(rgba, (nw, nh), interpolation=interp)


def _rotate(rgba: np.ndarray, angle: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _shear(rgba: np.ndarray, shear_deg: float) -> np.ndarray:
    """Horizontal shear of a BGRA cut-out, expanding the canvas so nothing is clipped
    (transparent border). Applied before placement so the label box is recomputed from
    the warped alpha and stays correct."""
    h, w = rgba.shape[:2]
    k = float(np.tan(np.deg2rad(shear_deg)))
    nw = max(1, int(np.ceil(w + abs(k) * (h - 1))))
    tx = -k * (h - 1) if k < 0 else 0.0
    m = np.array([[1.0, k, tx], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(rgba, m, (nw, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _perspective(rgba: np.ndarray, jit: np.ndarray) -> np.ndarray:
    """Random perspective warp of a BGRA cut-out, expanding the canvas so the warped
    quad is never clipped (transparent border). ``jit`` is a (4, 2) array of fractional
    corner offsets (dx as a frac of w, dy as a frac of h) for the TL, TR, BR, BL corners.
    Applied before placement so the label box is recomputed from the warped alpha and
    stays correct (mirrors :func:`_shear`)."""
    h, w = rgba.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + jit * np.array([w, h], dtype=np.float32)
    dst -= dst.min(axis=0, keepdims=True)  # shift the warped quad into the +,+ quadrant
    # Output size grows with the corner jitter; Config caps `perspective` at 0.3 (worst
    # case ~1.6x the input), so this safety clamp never fires for a validated run - it only
    # stops a direct call with an out-of-range jit from allocating a giant buffer (OOM).
    cap = 4 * max(h, w)
    nw = min(cap, max(1, int(np.ceil(float(dst[:, 0].max())))))
    nh = min(cap, max(1, int(np.ceil(float(dst[:, 1].max())))))
    m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _jitter(rgba: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    rgb = rgba[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.jitter_hue, cfg.jitter_hue)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - cfg.jitter_sat, 1 + cfg.jitter_sat), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(1 - cfg.jitter_val, 1 + cfg.jitter_val), 0, 255)
    out = rgba.copy()
    out[..., :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def transform_seed(
    seed: Seed, cfg: Config, canvas: int, rng: np.random.Generator, record: bool = False
):
    """Scale/flip/rotate/jitter a seed cut-out. Draw order is unchanged whether or
    not ``record`` is set (it only collects the sampled params), so determinism is
    preserved. With record=True returns (rgba, params); else just rgba."""
    rgba = seed.rgba
    oh, ow = rgba.shape[:2]  # original cut-out dims (provenance)
    scale_frac = rng.uniform(cfg.scale_min, cfg.scale_max)
    long_px = max(4, int(scale_frac * canvas))
    rgba = _resize_to_long_side(rgba, long_px)
    flip_h = cfg.flip and rng.random() < 0.5
    if flip_h:
        rgba = rgba[:, ::-1].copy()
    flip_v = cfg.flip and rng.random() < 0.5
    if flip_v:
        rgba = rgba[::-1, :].copy()
    rot = 0.0
    if cfg.max_rotation > 0:
        rot = float(rng.uniform(-cfg.max_rotation, cfg.max_rotation))
        rgba = _rotate(rgba, rot)
    shear = 0.0
    if cfg.aug_shear > 0:  # draw only when enabled (keeps
        shear = float(rng.uniform(-cfg.aug_shear, cfg.aug_shear))  # legacy streams stable)
        rgba = _shear(rgba, shear)
    persp = None
    if cfg.perspective > 0:  # one 8-value draw, gated like shear so OFF consumes none
        persp = rng.uniform(-cfg.perspective, cfg.perspective, size=(4, 2)).astype(np.float32)
        rgba = _perspective(rgba, persp)
    if cfg.jitter_hue or cfg.jitter_sat or cfg.jitter_val:
        rgba = _jitter(rgba, cfg, rng)
    if record:
        rec = {
            "scale": round(float(scale_frac), 4),
            "flip_h": bool(flip_h),
            "flip_v": bool(flip_v),
            "rotation": round(rot, 2),
            "shear": round(shear, 2),
            "orig_w": int(ow),
            "orig_h": int(oh),
        }
        if persp is not None:  # recorded only when applied, so OFF leaves provenance bytes stable
            rec["perspective"] = round(float(np.abs(persp).max()), 4)
        return rgba, rec
    return rgba
