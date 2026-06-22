#!/usr/bin/env python3
"""Generate synthetic SINGLE-seed source images that exercise both segmentation
regimes the brief calls out, plus the failure case the skip-safeguard must catch.

Regimes (one subfolder each so they double as --class-mode subfolders classes):
  typeA  - dark elongated ribbed seed on a near-white VIGNETTED tray + faint hair
           line. (caraway/cumin-like; brightness contrast, colour contrast)
  typeB  - pale tan seed on a light tray WITH a strong dark CAST SHADOW touching it
           and a burned-in phone WATERMARK ("REDMI 13C 5G" + date). Hardest case.
  pale   - small pale seed on a MEDIUM-GRAY background (wheat-like; tiny footprint,
           opposite-ish polarity).
  faint  - a near-invisible seed only ~3 grey levels from the background. The
           >70%-frame / low-confidence safeguard must SKIP these, never emit a patch.

Deterministic given --seed. Usage:
  python tests/make_synthetic_seeds.py --out /tmp/synth_seeds --per-class 6
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np


def _vignette(img: np.ndarray, strength: float = 0.45) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    fall = np.clip(1.0 - strength * (r ** 2), 0.3, 1.0)[..., None]
    return np.clip(img.astype(np.float32) * fall, 0, 255).astype(np.uint8)


def _ribbed_seed(canvas, cx, cy, ax, ay, base_bgr, rng, ribs=7):
    """Draw a filled ellipse with darker ribs/texture (caraway-like)."""
    h, w = canvas.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    tex = np.zeros_like(canvas, dtype=np.float32)
    tex[:] = base_bgr
    # ribs: darker stripes along the long axis
    for i in range(ribs):
        off = int((i - ribs / 2) * (2 * ay / ribs))
        cv2.ellipse(tex, (cx, cy + off), (ax, max(1, ay // (ribs + 2))), 0, 0, 360,
                    tuple(float(c) * 0.55 for c in base_bgr), -1)
    tex += rng.normal(0, 10, tex.shape).astype(np.float32)
    m = (mask > 0)[..., None]
    return np.where(m, np.clip(tex, 0, 255).astype(np.uint8), canvas), mask


def make_typeA(rng) -> np.ndarray:
    h, w = 480, 640
    bg = np.full((h, w, 3), (245, 248, 247), np.uint8)
    bg = bg + rng.normal(0, 3, bg.shape).astype(np.float32)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    cx, cy = w // 2 + int(rng.integers(-30, 30)), h // 2 + int(rng.integers(-20, 20))
    ax, ay = int(rng.integers(150, 190)), int(rng.integers(80, 110))
    base = (55, 75, 95)  # dark brown (BGR)
    img, _ = _ribbed_seed(bg, cx, cy, ax, ay, base, rng)
    img = _vignette(img, 0.5)
    # faint hair / scratch line on the tray
    p0 = (int(rng.integers(0, w)), int(rng.integers(0, 40)))
    p1 = (int(rng.integers(0, w)), int(rng.integers(0, 60)))
    cv2.line(img, p0, p1, (180, 185, 185), 1, cv2.LINE_AA)
    return img


def make_typeB(rng) -> np.ndarray:
    h, w = 640, 640
    bg = np.full((h, w, 3), (225, 228, 226), np.uint8)
    bg = np.clip(bg + rng.normal(0, 4, bg.shape), 0, 255).astype(np.uint8)
    cx, cy = w // 2 + int(rng.integers(-40, 40)), h // 2 + int(rng.integers(-40, 40))
    ax, ay = int(rng.integers(150, 200)), int(rng.integers(190, 240))
    ang = float(rng.integers(-30, 30))
    # strong dark cast shadow, offset up-left, TOUCHING the seed
    shadow = bg.copy().astype(np.float32)
    sh_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(sh_mask, (cx - 35, cy - 30), (ax, ay), ang, 0, 360, 255, -1)
    sm = cv2.GaussianBlur((sh_mask > 0).astype(np.float32), (0, 0), 9)[..., None]
    shadow = shadow * (1 - 0.55 * sm)  # darken
    img = shadow.astype(np.uint8)
    # pale tan seed on top
    seed_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(seed_mask, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)
    tan = np.zeros_like(img, dtype=np.float32)
    tan[:] = (120, 165, 195)  # pale tan (BGR)
    tan += rng.normal(0, 8, tan.shape).astype(np.float32)
    # a darker seam line across the seed
    cv2.ellipse(tan, (cx, cy), (ax // 6, ay), ang, 0, 360, (70, 95, 120), -1)
    m = (seed_mask > 0)[..., None]
    img = np.where(m, np.clip(tan, 0, 255).astype(np.uint8), img)
    # faint hairs
    for _ in range(3):
        p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p1 = (p0[0] + int(rng.integers(-60, 60)), p0[1] + int(rng.integers(-60, 60)))
        cv2.line(img, p0, p1, (150, 155, 155), 1, cv2.LINE_AA)
    # burned-in phone watermark (dark text -> far from light bg, must not be a seed)
    cv2.putText(img, "REDMI 13C 5G", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(img, "13/08/2025 22:13", (w - 230, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
    return img


def make_pale(rng) -> np.ndarray:
    h, w = 256, 256
    bg = np.full((h, w, 3), (110, 112, 115), np.uint8)  # medium gray
    bg = np.clip(bg + rng.normal(0, 4, bg.shape), 0, 255).astype(np.uint8)
    cx, cy = w // 2 + int(rng.integers(-30, 30)), h // 2 + int(rng.integers(-30, 30))
    ax, ay = int(rng.integers(14, 22)), int(rng.integers(24, 34))  # tiny footprint
    ang = float(rng.integers(-40, 40))
    seed_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(seed_mask, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)
    pale = np.zeros_like(bg, dtype=np.float32)
    pale[:] = (205, 225, 235)
    pale += rng.normal(0, 6, pale.shape).astype(np.float32)
    m = (seed_mask > 0)[..., None]
    return np.where(m, np.clip(pale, 0, 255).astype(np.uint8), bg)


def make_faint(rng) -> np.ndarray:
    h, w = 480, 640
    base = int(rng.integers(120, 200))
    bg = np.full((h, w, 3), (base, base, base), np.uint8)
    bg = np.clip(bg + rng.normal(0, 2, bg.shape), 0, 255).astype(np.uint8)
    cx, cy = w // 2, h // 2
    seed_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(seed_mask, (cx, cy), (140, 90), 0, 0, 360, 255, -1)
    # only ~3 grey levels different -> essentially invisible
    seed = np.clip(bg.astype(np.float32) + 3.0, 0, 255).astype(np.uint8)
    m = (seed_mask > 0)[..., None]
    return np.where(m, seed, bg)


MAKERS = {"typeA": make_typeA, "typeB": make_typeB,
          "pale": make_pale, "faint": make_faint}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for name, fn in MAKERS.items():
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.per_class):
            img = fn(rng)
            cv2.imwrite(str(d / f"{name}_{i:03d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    total = a.per_class * len(MAKERS)
    print(f"wrote {total} synthetic seeds to {a.out} "
          f"({len(MAKERS)} classes: {list(MAKERS)})")


if __name__ == "__main__":
    main()
