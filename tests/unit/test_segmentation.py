"""Unit tests for the classical (CPU-only) segmentation paths.

Only the deterministic, dependency-free methods are covered here - ``otsu`` / ``grabcut``
/ ``threshold`` / ``alpha``. The learned backends (``rembg`` / ``sam``) need onnxruntime /
torch + a checkpoint and a live GPU session, so they are out of scope for the in-memory
unit suite (they are smoke-tested separately and gated behind the ``gpu`` marker).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from multiseedgen.segmentation.loader import load_seeds, write_seg_report
from multiseedgen.segmentation.masks import _mask_otsu
from multiseedgen.segmentation.segment import extract_background, segment_seed


def test_scan_limit_caps_the_sources_segmented(make_cfg, tmp_path):
    """scan_limit must reduce the actual segmentation work (and thus the report counts),
    not just the rendered preview: only N sources are processed, so scanned == N and
    kept + skipped == N."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):  # six separable dark-square-on-light sources (otsu keeps them)
        img = np.full((48, 48, 3), 200, np.uint8)
        img[14:34, 14:34] = 30
        cv2.imwrite(str(src / f"s{i}.png"), img)

    cfg = make_cfg(
        sources=src, out=tmp_path / "out", segment="otsu", jobs=1, use_cache=False, scan_limit=3
    )
    _seeds, _names, _bgs, results = load_seeds(cfg)
    assert len(results) == 3  # only the sampled subset was segmented (not all 6)

    summary = write_seg_report(cfg.out, results, cfg)
    assert summary["scanned"] == 3
    assert summary["kept"] + summary["skipped"] == 3


@pytest.mark.parametrize("method", ["otsu", "grabcut", "threshold"])
def test_classical_methods_return_cropped_bgra_patch(make_cfg, seed_on_uniform_bg, method):
    """Each forced classical method cuts the object out as a tight BGRA patch.

    Forced (non-``auto``) methods skip the confidence gate, so a clean high-contrast
    object is always retained."""
    cfg = make_cfg(segment=method, rembg_fallback=False)

    patch, report = segment_seed(seed_on_uniform_bg, cfg)

    assert patch is not None, f"{method} unexpectedly skipped the source"
    assert patch.ndim == 3 and patch.shape[2] == 4  # tight BGRA cut-out
    assert report["method"] == method
    assert report["skipped"] is False


def test_explicit_method_confidence_gate_is_opt_in(make_cfg, seed_on_uniform_bg):
    """For a non-``auto`` method the confidence gate is opt-in. At the default ``0`` it
    keeps every mask; raising ``seg_conf_thresh`` above the mask's score skips the source
    as ``low_confidence`` WITHOUT routing it to rembg (the fallback stays auto-only)."""
    # gate off (default for explicit methods) -> the mask is kept, with a real score
    off = make_cfg(segment="otsu", rembg_fallback=True, seg_conf_thresh=0.0)
    patch, rep = segment_seed(seed_on_uniform_bg, off)
    assert patch is not None and rep["skipped"] is False
    conf = rep["confidence"]
    assert conf > 0.0

    # opt-in: a threshold above the score skips the SAME source as low_confidence, and
    # leaves the method unchanged (rembg_fallback=True must not fire outside auto).
    on = make_cfg(segment="otsu", rembg_fallback=True, seg_conf_thresh=conf + 0.05)
    patch2, rep2 = segment_seed(seed_on_uniform_bg, on)
    assert patch2 is None
    assert rep2["skipped"] is True
    assert rep2["skip_reason"] == "low_confidence"
    assert rep2["method"] == "otsu"  # not swapped to rembg


def test_alpha_method_uses_existing_alpha_channel(make_cfg, seed_cutout):
    """A source that already carries a meaningful alpha channel is cropped to it with
    full confidence - no thresholding involved."""
    cfg = make_cfg(segment="alpha")

    patch, report = segment_seed(seed_cutout.rgba, cfg)

    assert patch is not None and patch.shape[2] == 4
    assert report["method"] == "alpha"
    assert report["confidence"] == 1.0


def test_segment_seed_skips_a_blank_source(make_cfg):
    """A uniform frame has no separable object -> the source is skipped, not faked."""
    cfg = make_cfg(segment="otsu", rembg_fallback=False)
    blank = np.full((48, 48, 3), 128, np.uint8)

    patch, report = segment_seed(blank, cfg)

    assert patch is None
    assert report["skipped"] is True


def test_mask_otsu_is_binary_and_non_degenerate(seed_on_uniform_bg):
    """The raw Otsu mask is a 0/255 binary image that separates *something* from the
    background (neither all-foreground nor all-background)."""
    mask = _mask_otsu(seed_on_uniform_bg)

    assert mask.shape == seed_on_uniform_bg.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})
    assert 0.0 < (mask > 0).mean() < 1.0


def test_extract_background_inpaints_to_same_size_bgr(make_cfg, seed_on_uniform_bg):
    """Recovering the bare background returns a same-shape BGR canvas with the object
    inpainted away (used by ``--bg-from-sources``)."""
    cfg = make_cfg(segment="otsu")

    bg = extract_background(seed_on_uniform_bg, cfg)

    assert bg is not None
    assert bg.shape == seed_on_uniform_bg.shape  # same HxWx3
    assert bg.dtype == np.uint8


def test_extract_background_is_none_for_transparent_source(make_cfg, seed_cutout):
    """A cut-out with a transparent border has no surface behind it to recover."""
    cfg = make_cfg(segment="otsu")
    assert extract_background(seed_cutout.rgba, cfg) is None


def _write_separable_seed(path):
    """A dark square on a light field - otsu keeps it as a tight cut-out."""
    img = np.full((48, 48, 3), 200, np.uint8)
    img[14:34, 14:34] = 30
    cv2.imwrite(str(path), img)


def test_single_mode_warns_on_multiple_subfolders(make_cfg, tmp_path, caplog):
    """A multi-subfolder layout left in the default single mode collapses every seed into
    one class - warn so the user knows to reach for --class-mode subfolders."""
    src = tmp_path / "src"
    for sub in ("wheat", "soybean"):
        (src / sub).mkdir(parents=True)
        _write_separable_seed(src / sub / "s0.png")

    cfg = make_cfg(sources=src, out=tmp_path / "out", segment="otsu", jobs=1, use_cache=False)
    with caplog.at_level(logging.WARNING, logger="multiseedgen.segmentation.loader"):
        _seeds, class_names, _bgs, _results = load_seeds(cfg)

    assert class_names == ["seed"]  # subfolders ignored in single mode
    assert any("class_mode=single" in r.message for r in caplog.records)


def test_subfolders_mode_records_class_in_seg_report(make_cfg, tmp_path):
    """class_mode=subfolders carries each subfolder name through to the per-seed report,
    so write_seg_report can tally kept cut-outs per class."""
    src = tmp_path / "src"
    for sub in ("wheat", "soybean"):
        (src / sub).mkdir(parents=True)
        _write_separable_seed(src / sub / "s0.png")

    cfg = make_cfg(
        sources=src,
        out=tmp_path / "out",
        segment="otsu",
        jobs=1,
        use_cache=False,
        class_mode="subfolders",
    )
    _seeds, class_names, _bgs, results = load_seeds(cfg)
    assert set(class_names) == {"wheat", "soybean"}
    # every kept seed's report carries its class
    assert all(
        r.report.get("class_name") in class_names for r in results if not r.report.get("skipped")
    )

    summary = write_seg_report(cfg.out, results, cfg)
    assert summary["class_counts"] == {"wheat": 1, "soybean": 1}
