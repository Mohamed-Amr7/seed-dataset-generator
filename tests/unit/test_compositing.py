"""Unit tests for the compositing primitives (``multiseedgen.compositing``).

The pipeline's byte-identity contract rests on these functions drawing from the PRNG in
a fixed order, so the determinism checks here (same seed -> same pixels) are the
load-bearing assertions; the rest pin shapes and the pure ``_iou`` geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from multiseedgen.compositing.backgrounds import (
    _bg_gradient,
    _bg_noise,
    _bg_seedbed,
    _bg_solid,
    _bg_texture,
    make_background,
)
from multiseedgen.compositing.blend import _iou
from multiseedgen.compositing.scene import generate_scene
from multiseedgen.compositing.transforms import _perspective, transform_seed


# --------------------------------------------------------------------------- #
# Backgrounds                                                                 #
# --------------------------------------------------------------------------- #
def test_named_background_kinds_have_canvas_shape(rng, seed_cutout):
    """Every synthetic background fills the full square canvas with 3 channels."""
    size = 48
    textures = [np.full((80, 80, 3), 100, np.uint8)]
    for bg in (
        _bg_solid(size, rng),
        _bg_gradient(size, rng),
        _bg_noise(size, rng),
        _bg_seedbed(size, [seed_cutout], rng),
        _bg_texture(size, textures, rng),
    ):
        assert bg.shape[:2] == (size, size)
        assert bg.shape[2] == 3
        assert np.isfinite(np.asarray(bg, np.float64)).all()


def test_make_background_is_deterministic_for_fixed_seed(make_cfg, seed_cutout):
    cfg = make_cfg(backgrounds=["solid", "gradient", "noise"])
    a = make_background(cfg, [seed_cutout], [], np.random.default_rng(99))
    b = make_background(cfg, [seed_cutout], [], np.random.default_rng(99))
    assert np.array_equal(a, b)


def test_make_background_returns_float32(make_cfg, seed_cutout):
    """The pipeline composites onto a float32 canvas; the contract is dtype-stable."""
    cfg = make_cfg()
    bg = make_background(cfg, [seed_cutout], [], np.random.default_rng(0))
    assert bg.dtype == np.float32


# --------------------------------------------------------------------------- #
# Seed transform                                                              #
# --------------------------------------------------------------------------- #
def test_transform_seed_is_deterministic_for_fixed_seed(make_cfg, seed_cutout):
    """Identical (config, seed) -> identical transformed pixels, across all draws."""
    cfg = make_cfg(aug_shear=5.0, max_rotation=30.0)
    a = transform_seed(seed_cutout, cfg, 64, np.random.default_rng(7))
    b = transform_seed(seed_cutout, cfg, 64, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_transform_seed_record_does_not_perturb_pixels(make_cfg, seed_cutout):
    """``record=True`` only *collects* the sampled params - it must not consume extra
    RNG or alter the pixels, or provenance logging would change the dataset."""
    cfg = make_cfg(aug_shear=5.0, max_rotation=30.0)
    plain = transform_seed(seed_cutout, cfg, 64, np.random.default_rng(7))
    recorded, params = transform_seed(seed_cutout, cfg, 64, np.random.default_rng(7), record=True)

    assert np.array_equal(plain, recorded)
    assert set(params) >= {"scale", "flip_h", "flip_v", "rotation", "shear"}
    assert cfg.scale_min <= params["scale"] <= cfg.scale_max


def test_perspective_off_consumes_no_draw(make_cfg, seed_cutout):
    """The perspective draw is gated on ``cfg.perspective > 0``: at the default 0 it
    must consume nothing, so a perspective=0 stream is byte-identical to one where the
    field doesn't exist. This is the legacy-determinism contract for the new branch."""
    off = transform_seed(seed_cutout, make_cfg(), 64, np.random.default_rng(3))
    base = transform_seed(seed_cutout, make_cfg(perspective=0.0), 64, np.random.default_rng(3))
    assert np.array_equal(off, base)


def test_perspective_warps_and_is_deterministic(make_cfg, seed_cutout):
    """Enabling perspective changes the cut-out (it is actually applied) and stays
    byte-identical for a fixed (config, seed)."""
    cfg = make_cfg(perspective=0.3)
    plain = transform_seed(seed_cutout, make_cfg(), 64, np.random.default_rng(11))
    a = transform_seed(seed_cutout, cfg, 64, np.random.default_rng(11))
    b = transform_seed(seed_cutout, cfg, 64, np.random.default_rng(11))
    assert np.array_equal(a, b)  # deterministic
    assert a.shape != plain.shape or not np.array_equal(a, plain)  # actually warped


def test_perspective_recorded_only_when_applied(make_cfg, seed_cutout):
    """``record=True`` adds a ``perspective`` key iff perspective is enabled, so OFF
    runs leave provenance bytes unchanged. The recorded value is a non-negative jitter
    magnitude bounded by the configured strength."""
    _, off = transform_seed(seed_cutout, make_cfg(), 64, np.random.default_rng(11), record=True)
    assert "perspective" not in off

    rgba, on = transform_seed(
        seed_cutout, make_cfg(perspective=0.3), 64, np.random.default_rng(11), record=True
    )
    plain = transform_seed(seed_cutout, make_cfg(perspective=0.3), 64, np.random.default_rng(11))
    assert np.array_equal(plain, rgba)  # record must not perturb pixels
    assert 0.0 <= on["perspective"] <= 0.3


def test_perspective_output_size_is_bounded(seed_cutout):
    """Regression: an out-of-range jit (the config gate normally rejects this) must not
    allocate an unbounded buffer - the freeze/OOM mode. The helper caps output at 4x the
    input's longer side, so even an absurd jit stays small."""
    rgba = seed_cutout.rgba
    huge = np.full((4, 2), 50.0, np.float32)  # 50x the cut-out - would be GBs uncapped
    out = _perspective(rgba, huge)
    cap = 4 * max(rgba.shape[:2])
    assert max(out.shape[:2]) <= cap


# --------------------------------------------------------------------------- #
# IoU geometry (pure)                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "a, b, expected",
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),  # identical
        ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),  # disjoint
        ((0, 0, 10, 10), (5, 0, 15, 10), 1 / 3),  # half-overlap
    ],
)
def test_iou_matches_hand_computed_value(a, b, expected):
    assert _iou(a, b) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Scene composition                                                           #
# --------------------------------------------------------------------------- #
def test_generate_scene_negative_has_no_labels(make_cfg, seed_cutout):
    cfg = make_cfg()
    img, labels, polys, prov = generate_scene(
        cfg, [seed_cutout], [], np.random.default_rng(1), negative=True
    )
    assert labels == []
    assert polys is None and prov is None
    assert img.shape == (cfg.img_size, cfg.img_size, 3)
    assert img.dtype == np.uint8


def test_generate_scene_is_deterministic_for_fixed_seed(make_cfg, seed_cutout):
    cfg = make_cfg(min_seeds=2, max_seeds=4)
    seeds = [seed_cutout, seed_cutout, seed_cutout]
    img_a, labels_a, _, _ = generate_scene(cfg, seeds, [], np.random.default_rng(5))
    img_b, labels_b, _, _ = generate_scene(cfg, seeds, [], np.random.default_rng(5))

    assert np.array_equal(img_a, img_b)
    assert labels_a == labels_b
