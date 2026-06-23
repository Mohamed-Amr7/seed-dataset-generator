#!/usr/bin/env python3
"""Determinism / byte-identity regression test for MultiSeedGen.

The pipeline's core contract is that, for a fixed ``(config, seed)``, the generated
dataset is **byte-identical** regardless of how the work is parallelised:

  * scene RNG is a per-(split, scene-index) ``np.random.SeedSequence`` stream, so a
    scene's pixels never depend on worker count or scheduling
    (``multiseedgen/pipeline.py::_scene_seeds``);
  * cut-outs are content-hash + ``SEG_VERSION`` + seg-param keyed and cached, so
    segmentation is reproducible;
  * the ``loky`` and ``threading`` backends reconstruct the SAME seed/texture pool in
    the same order (``_stage_assets_mmap`` / ``_load_assets``).

The recent ``fix/bounded-memory-generation`` work (memmap-shared loky assets, the
``--max-seed-px`` cap) relies on this invariant but nothing guarded it automatically.
This test does, reproducing the manual check used while that fix was developed:
generate a tiny fixed set of synthetic seeds, run the generator several ways, and
SHA-1 every output artifact that is supposed to be deterministic.

It runs fully on CPU and needs no real image data: segmentation is forced to
``otsu`` with ``rembg_fallback=False`` so no GPU / onnxruntime / rembg path (which is
deliberately *not* bit-reproducible run-to-run) is ever touched.

Run with ``pytest tests/`` or standalone: ``python tests/test_determinism.py``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest

# Skip cleanly (rather than error) on a host missing the runtime deps.
pytest.importorskip("cv2")
pytest.importorskip("numpy")
msg = pytest.importorskip("multiseedgen")

_MAKE_SEEDS = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_seeds.py"

# Output artifacts whose bytes MUST be identical for a fixed (config, seed). Excluded
# on purpose: data.yaml (embeds the absolute --out path, which differs per run dir),
# run_config.{yaml,json} (embeds a wall-clock timestamp + argv), and seg_report.txt
# (embeds run-varying cache-hit counts). The task's contract is exactly this set.
_DETERMINISTIC_GLOBS = (
    "train/images/*",
    "val/images/*",
    "train/labels/*",
    "val/labels/*",
    "annotations_*.json",
    "multilabel.csv",
    "provenance_*.jsonl",
)

# Fixed run parameters. Small + fast (~1-2s/run after the cache is warm). num_images,
# otsu, and rembg_fallback=False follow the task brief; class-mode=subfolders gives 4
# classes so multilabel.csv / COCO categories are non-trivial.
SEED = 1234
NUM_IMAGES = 20
IMG_SIZE = 320
PER_CLASS = 3


def _hash_tree(out: Path) -> dict[str, str]:
    """Map each deterministic artifact (by path RELATIVE to ``out``) to its SHA-1.

    Hashing by relative path means two runs in different output directories are
    directly comparable: same relative layout -> same hashes iff bytes match."""
    digests: dict[str, str] = {}
    for f in sorted(out.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(out).as_posix()
        if any(fnmatch.fnmatch(rel, g) for g in _DETERMINISTIC_GLOBS):
            digests[rel] = hashlib.sha1(f.read_bytes()).hexdigest()
    return digests


def _run(out: Path, cache_dir: Path, seeds_dir: Path, **overrides) -> dict[str, str]:
    """Generate one dataset into ``out`` and return its artifact-hash map.

    All runs share ``cache_dir`` (segmentation is deterministic and its cache key does
    NOT depend on backend/workers/max_seed_px), so the cut-outs are computed once.
    ``overrides`` varies only the factor under test."""
    kw = dict(
        sources=seeds_dir,
        out=out,
        num_images=NUM_IMAGES,
        img_size=IMG_SIZE,
        seed=SEED,
        segment="otsu",  # forced classical path: deterministic, CPU-only
        rembg_fallback=False,  # never touch the GPU/onnxruntime (non-deterministic)
        cache_dir=cache_dir,
        class_mode="subfolders",
        fmt="both",  # -> YOLO label .txt AND annotations_*.json
        provenance=True,  # -> provenance_*.jsonl
        multilabel_csv=True,  # -> multilabel.csv
        jobs=1,  # serial segmentation (the cache makes it a one-off)
    )
    kw.update(overrides)
    msg.run(msg.Config(**kw))
    return _hash_tree(out)


# --------------------------------------------------------------------------- #
# Fixtures: synthetic seeds + a shared cache + one reference run              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def seeds_dir(tmp_path_factory) -> Path:
    """A small fixed set of synthetic single-seed sources, generated via the existing
    tests/make_synthetic_seeds.py helper (deterministic given --seed)."""
    d = tmp_path_factory.mktemp("synth_seeds")
    subprocess.run(
        [
            sys.executable,
            str(_MAKE_SEEDS),
            "--out",
            str(d),
            "--per-class",
            str(PER_CLASS),
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert list(d.rglob("*.jpg")), "make_synthetic_seeds produced no images"
    return d


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory) -> Path:
    """One segmentation cache shared by every run in this module."""
    return tmp_path_factory.mktemp("seg_cache")


@pytest.fixture(scope="module")
def reference(tmp_path_factory, seeds_dir, cache_dir) -> dict[str, str]:
    """The canonical run: serial threading, 1 worker (the documented reference path).
    Every other configuration must reproduce this byte-for-byte."""
    out = tmp_path_factory.mktemp("ref")
    ref = _run(out, cache_dir, seeds_dir, parallel_backend="threading", workers=1)
    assert ref, "reference run produced no deterministic artifacts"
    # Sanity: the artifact set we promised to guard is actually present.
    assert any(k.startswith("train/images/") for k in ref)
    assert any(k.startswith("train/labels/") for k in ref)
    assert "annotations_train.json" in ref and "annotations_val.json" in ref
    assert "multilabel.csv" in ref
    assert "provenance_train.jsonl" in ref and "provenance_val.jsonl" in ref
    return ref


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "backend,workers",
    list(product(["loky", "threading"], [1, 3])),
    ids=lambda v: str(v),
)
def test_output_independent_of_backend_and_workers(
    backend, workers, tmp_path, seeds_dir, cache_dir, reference
):
    """Core invariant: output must be byte-identical across parallel backends and
    worker counts. (The threading/1-worker case re-runs the reference config, which
    also proves plain run-to-run reproducibility.)"""
    got = _run(
        tmp_path / f"{backend}_{workers}",
        cache_dir,
        seeds_dir,
        parallel_backend=backend,
        workers=workers,
    )

    assert got.keys() == reference.keys(), (
        "different set of output files for "
        f"backend={backend} workers={workers}: "
        f"only-in-reference={sorted(set(reference) - set(got))[:5]} "
        f"only-in-run={sorted(set(got) - set(reference))[:5]}"
    )
    mismatched = [k for k in reference if reference[k] != got[k]]
    assert not mismatched, (
        f"non-deterministic output for backend={backend} workers={workers}; "
        f"{len(mismatched)} file(s) differ, e.g. {mismatched[:5]}"
    )


def test_max_seed_px_above_largest_is_noop(tmp_path, seeds_dir, cache_dir, reference):
    """``--max-seed-px`` above every seed's long side resamples nothing, so output is
    byte-identical to the un-capped reference (the cap never enters the seg cache key,
    it only downscales oversized patches in RAM)."""
    got = _run(
        tmp_path / "max_px_above",
        cache_dir,
        seeds_dir,
        parallel_backend="threading",
        workers=1,
        max_seed_px=4096,
    )
    mismatched = [k for k in reference if reference[k] != got[k]]
    assert got.keys() == reference.keys() and not mismatched, (
        f"--max-seed-px above the largest seed should be a no-op; "
        f"{len(mismatched)} file(s) differ, e.g. {mismatched[:5]}"
    )


def test_max_seed_px_below_largest_changes_output(tmp_path, seeds_dir, cache_dir, reference):
    """A tiny ``--max-seed-px`` downscales the stored patches, so the rendered pixels
    must differ from the reference (this is the opt-in memory/quality tradeoff)."""
    got = _run(
        tmp_path / "max_px_below",
        cache_dir,
        seeds_dir,
        parallel_backend="threading",
        workers=1,
        max_seed_px=32,
    )
    assert got, "capped run produced no artifacts"
    ref_imgs = {k: v for k, v in reference.items() if "/images/" in k}
    got_imgs = {k: v for k, v in got.items() if "/images/" in k}
    assert ref_imgs and ref_imgs.keys() == got_imgs.keys()
    assert ref_imgs != got_imgs, (
        "--max-seed-px below the largest seed should change the rendered pixels, "
        "but every image hashed identically to the un-capped run"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
