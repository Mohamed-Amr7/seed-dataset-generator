"""Background generation: solid / gradient / noise / real-texture crop / seedbed."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import Config, Seed
from .blend import _blend_region
from .transforms import _resize_to_long_side


def _bg_solid(size: int, rng) -> np.ndarray:
    color = rng.integers(30, 226, size=3)
    return np.full((size, size, 3), color, dtype=np.uint8)


def _bg_gradient(size: int, rng) -> np.ndarray:
    c0 = rng.integers(20, 236, size=3).astype(np.float32)
    c1 = rng.integers(20, 236, size=3).astype(np.float32)
    line = np.linspace(0, 1, size, dtype=np.float32)
    if rng.random() < 0.5:  # vertical gradient
        t = np.repeat(line[:, None], size, axis=1)
    else:  # horizontal gradient
        t = np.repeat(line[None, :], size, axis=0)
    t = t[..., None]  # (H, W, 1)
    grad = c0[None, None, :] * (1 - t) + c1[None, None, :] * t
    return np.clip(grad, 0, 255).astype(np.uint8)


def _bg_noise(size: int, rng) -> np.ndarray:
    base = rng.integers(40, 216, size=3).astype(np.float32)
    noise = rng.normal(0, 18, size=(size, size, 3)).astype(np.float32)
    img = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.5))


def _bg_texture(size: int, textures: list[np.ndarray], rng) -> np.ndarray:
    """Random square crop from a real photo, with random zoom + flip, resized to
    the canvas. A handful of real photos thus yield many distinct backgrounds."""
    img = textures[rng.integers(0, len(textures))]
    h, w = img.shape[:2]
    short = min(h, w)
    # crop between 40% and 100% of the short side -> scale variety ("zoom")
    crop = int(max(16, min(short, rng.uniform(0.4, 1.0) * short)))
    y = int(rng.integers(0, h - crop + 1))
    x = int(rng.integers(0, w - crop + 1))
    patch = img[y : y + crop, x : x + crop]
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    if rng.random() < 0.5:
        patch = patch[::-1, :]
    interp = cv2.INTER_AREA if crop > size else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interp)


def _bg_seedbed(size: int, seeds: list[Seed], rng) -> np.ndarray:
    """A blurred clutter background made of out-of-focus seeds: forces the
    detector to separate in-focus seeds from seed-coloured background."""
    canvas = _bg_solid(size, rng).astype(np.float32)
    for _ in range(rng.integers(8, 20)):
        s = seeds[rng.integers(0, len(seeds))].rgba
        big = _resize_to_long_side(s, int(rng.uniform(0.25, 0.6) * size))
        a = cv2.GaussianBlur(big[..., 3].astype(np.float32) / 255.0, (0, 0), 6.0)[..., None]
        rgb = cv2.GaussianBlur(big[..., :3], (0, 0), 6.0).astype(np.float32)
        ph, pw = big.shape[:2]
        x = int(rng.integers(-pw // 2, size - pw // 2))
        y = int(rng.integers(-ph // 2, size - ph // 2))
        _blend_region(canvas, rgb, a, x, y)
    return cv2.GaussianBlur(canvas, (0, 0), rng.uniform(2.0, 5.0)).astype(np.uint8)


def make_background(
    cfg: Config, seeds: list[Seed], textures: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    # When real background photos are supplied, use them most of the time and
    # keep a little synthetic variety (domain randomization aids generalization
    # even with real backgrounds). Real backgrounds are strongly preferred.
    if textures and rng.random() < 0.7:
        return _bg_texture(cfg.img_size, textures, rng).astype(np.float32)

    kinds = [k for k in cfg.backgrounds if k != "texture"] or ["solid"]
    kind = kinds[rng.integers(0, len(kinds))]
    if kind == "solid":
        bg = _bg_solid(cfg.img_size, rng)
    elif kind == "gradient":
        bg = _bg_gradient(cfg.img_size, rng)
    elif kind == "noise":
        bg = _bg_noise(cfg.img_size, rng)
    elif kind == "seedbed":
        bg = _bg_seedbed(cfg.img_size, seeds, rng)
    elif kind == "texture" and textures:
        bg = _bg_texture(cfg.img_size, textures, rng)
    else:
        bg = _bg_solid(cfg.img_size, rng)
    return bg.astype(np.float32)
