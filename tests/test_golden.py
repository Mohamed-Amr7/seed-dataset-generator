#!/usr/bin/env python3
"""Golden byte-manifest regression test for MultiSeedGen.

Where ``tests/test_determinism.py`` proves the output is *backend/worker-invariant*
(it recomputes its own reference every run), THIS test locks the output **across code
changes**: it stores a committed SHA-1 manifest of every deterministic artifact for a
fixed ``(config, seed)`` and fails if a refactor changes a single byte.

It is the primary safety net for the backend rework. Two profiles are pinned:

  * ``otsu_basic`` - the minimal CPU/otsu config (mirrors test_determinism.py).
  * ``otsu_rich``  - turns on the RNG-*gated* branches the basic profile never
    exercises (per-seed shear, gamma/blur/motion-blur/vignette degradation, per-scene
    shadow direction, a jpeg-quality range, overlap limiting, COCO polygons), so the
    "draw only when the feature is enabled" determinism contract is locked too.

Golden manifests live in ``tests/golden/<profile>.json``. Regenerate them DELIBERATELY
(e.g. after a reviewed dependency bump that legitimately changes pixels) with::

    UPDATE_GOLDEN=1 pytest tests/test_golden.py

The manifests are environment-sensitive (numpy SeedSequence + cv2 reductions), so a
diff after a numpy/opencv change is expected and must be reviewed, not blindly updated.

Runs fully on CPU, no GPU/onnxruntime/rembg path (segment=otsu, rembg_fallback=False).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")
msg = pytest.importorskip("multiseedgen")

_MAKE_SEEDS = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_seeds.py"
_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Artifacts whose bytes MUST be identical for a fixed (config, seed). Excludes
# data.yaml (embeds the absolute --out path), run_config.* (timestamp + argv), and
# seg_report.txt (run-varying cache-hit counts) - same contract as test_determinism.py.
_DETERMINISTIC_GLOBS = (
    "train/images/*",
    "val/images/*",
    "train/labels/*",
    "val/labels/*",
    "annotations_*.json",
    "multilabel.csv",
    "provenance_*.jsonl",
)

SEED = 1234
NUM_IMAGES = 20
IMG_SIZE = 320
PER_CLASS = 3

# Shared by every profile: the deterministic CPU/otsu base. Profiles layer on top.
_BASE = dict(
    num_images=NUM_IMAGES,
    img_size=IMG_SIZE,
    seed=SEED,
    segment="otsu",  # forced classical path: deterministic, CPU-only
    rembg_fallback=False,  # never touch the GPU/onnxruntime (non-deterministic)
    class_mode="subfolders",
    fmt="both",  # -> YOLO label .txt AND annotations_*.json
    provenance=True,  # -> provenance_*.jsonl
    multilabel_csv=True,  # -> multilabel.csv
    jobs=1,  # serial segmentation (the cache makes it a one-off)
    parallel_backend="threading",
    workers=1,
)

# Each profile's deltas over _BASE. otsu_rich flips on the RNG-gated branches that the
# basic profile leaves at their disabled defaults.
_PROFILES = {
    "otsu_basic": {},
    "otsu_rich": dict(
        masks=True,  # -> COCO segmentation polygons
        aug_shear=6.0,  # per-seed shear draw (gated on aug_shear > 0)
        max_rotation=180.0,
        degrade=True,  # master switch for the photometric draws below
        noise_sigma=3.0,
        aug_gamma=0.12,  # gated on aug_gamma > 0
        blur_max_sigma=0.6,  # gated on blur_max_sigma > 0
        motion_blur_max=3,  # gated on motion_blur_max > 0
        vignette=0.25,  # gated on vignette > 0
        jpeg_quality_min=80,  # quality range (min < max) -> per-scene draw
        jpeg_quality_max=95,
        shadow_dir_mode="scene",  # per-scene light direction draw
        overlap_mode="limit",  # placement rejection path
        max_iou=0.5,
        neg_frac=0.10,  # background-only scenes (empty labels)
        class_balanced=True,  # class-first sampling path
    ),
}


def _hash_tree(out: Path) -> dict[str, str]:
    """Map each deterministic artifact (path RELATIVE to ``out``) to its SHA-1."""
    digests: dict[str, str] = {}
    for f in sorted(out.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(out).as_posix()
        if any(fnmatch.fnmatch(rel, g) for g in _DETERMINISTIC_GLOBS):
            digests[rel] = hashlib.sha1(f.read_bytes()).hexdigest()
    return digests


def _run_profile(out: Path, cache_dir: Path, seeds_dir: Path, profile: str) -> dict[str, str]:
    kw = dict(_BASE)
    kw.update(_PROFILES[profile])
    kw.update(sources=seeds_dir, out=out, cache_dir=cache_dir)
    msg.run(msg.Config(**kw))
    return _hash_tree(out)


@pytest.fixture(scope="module")
def seeds_dir(tmp_path_factory) -> Path:
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
    return tmp_path_factory.mktemp("seg_cache")


@pytest.mark.golden
@pytest.mark.parametrize("profile", sorted(_PROFILES))
def test_output_matches_golden(profile, tmp_path, seeds_dir, cache_dir):
    """The generated dataset must match the committed golden manifest byte-for-byte."""
    got = _run_profile(tmp_path / profile, cache_dir, seeds_dir, profile)
    assert got, f"profile {profile!r} produced no deterministic artifacts"

    golden_path = _GOLDEN_DIR / f"{profile}.json"
    if os.environ.get("UPDATE_GOLDEN") or not golden_path.exists():
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"wrote golden manifest {golden_path.name} ({len(got)} artifacts)")

    golden = json.loads(golden_path.read_text())
    assert got.keys() == golden.keys(), (
        f"{profile}: artifact set changed; "
        f"only-in-golden={sorted(set(golden) - set(got))[:5]} "
        f"only-in-run={sorted(set(got) - set(golden))[:5]}"
    )
    mismatched = [k for k in golden if golden[k] != got[k]]
    assert not mismatched, (
        f"{profile}: {len(mismatched)} artifact(s) differ from the golden manifest, "
        f"e.g. {mismatched[:5]}. If this is an intentional/dependency-driven change, "
        f"review the diff and regenerate with UPDATE_GOLDEN=1."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
