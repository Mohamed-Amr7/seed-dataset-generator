"""Shared fixtures for the isolated unit tests.

These tests exercise single functions in memory with synthetic data - no subprocess,
no GPU/onnxruntime, no network, and no real image corpus - so the whole ``tests/unit``
suite stays fast and deterministic. The integration/contract tests that need a real
generation (determinism, golden, schema/config snapshots, web) live one level up in
``tests/``.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from multiseedgen import Config
from multiseedgen.models import Seed


@pytest.fixture
def make_cfg(tmp_path):
    """Factory for a minimal, valid :class:`Config`.

    ``sources`` / ``out`` point at (unused) tmp paths - the model validates ranges and
    derivations but never touches the filesystem, so the dirs need not exist. Keyword
    overrides let each test flip exactly the field under test and nothing else.
    """

    def _make(**overrides) -> Config:
        base = dict(
            sources=tmp_path / "src",
            out=tmp_path / "out",
            img_size=64,
            num_images=2,
            min_seeds=1,
            max_seeds=3,
        )
        base.update(overrides)
        return Config(**base)

    return _make


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed-seed PRNG so every draw-order-sensitive test is reproducible."""
    return np.random.default_rng(1234)


@pytest.fixture
def seed_cutout() -> Seed:
    """A clean BGRA cut-out: an opaque red disk on a transparent border.

    Used by the compositing / QA tests that need a ready-made seed patch (the alpha
    channel is the silhouette the blender and box-finder rely on)."""
    rgba = np.zeros((40, 40, 4), np.uint8)
    cv2.circle(rgba, (20, 20), 14, (40, 40, 200, 255), -1)  # B,G,R,A
    return Seed(rgba=rgba, class_id=0, source="synthetic/seed_000.png", method="otsu")


@pytest.fixture
def seed_on_uniform_bg() -> np.ndarray:
    """A BGR source for the classical segmenters: a dark square centred on a uniform
    light border. Every border-colour / Otsu / GrabCut path can separate it cleanly,
    and it occupies ~22% of the frame (well under the 70% oversize safeguard)."""
    img = np.full((64, 64, 3), 200, np.uint8)  # uniform light background
    img[17:47, 17:47] = (30, 30, 30)  # dark, clearly separable object
    return img
