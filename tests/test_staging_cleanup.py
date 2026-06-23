"""B1 regression: the loky staging dir + manifest must be removed on ANY exit.

``pipeline.run()`` stages seeds/textures to ``<out>/.assets_mmap`` (+ a manifest) for
the loky backend so workers can memory-map one shared copy. The cleanup used to run only
on the happy path, so a crash / Ctrl-C / exception mid-generation orphaned the multi-GB
staging dir until a *later* loky run self-healed it. The cleanup is now in a ``finally``;
this test forces the loky branch and injects a mid-generation failure, then asserts both
the staging dir and the manifest are gone.

Runs fully on CPU with tiny synthetic seeds (forced ``otsu``, ``rembg_fallback=False``),
so no GPU / onnxruntime path is touched.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
msg = pytest.importorskip("multiseedgen")

from multiseedgen import pipeline  # noqa: E402

_MAKE_SEEDS = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_seeds.py"


@pytest.fixture(scope="module")
def seeds_dir(tmp_path_factory) -> Path:
    """A few synthetic single-seed sources via the existing generator helper."""
    d = tmp_path_factory.mktemp("synth_seeds_b1")
    subprocess.run(
        [sys.executable, str(_MAKE_SEEDS), "--out", str(d), "--per-class", "2", "--seed", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert list(d.rglob("*.jpg")), "make_synthetic_seeds produced no images"
    return d


def _cfg(out: Path, seeds_dir: Path):
    # workers=1 keeps rendering serial/in-process so a monkeypatched crash propagates to
    # the test; we force the loky backend separately so assets are actually staged.
    return msg.Config(
        sources=seeds_dir,
        out=out,
        num_images=6,
        img_size=128,
        seed=1,
        segment="otsu",
        rembg_fallback=False,
        jobs=1,
        workers=1,
    )


def _staging_paths(out: Path) -> tuple[Path, Path]:
    return out / ".assets_mmap", out / ".assets_manifest.json"


def test_staging_cleaned_up_when_generation_crashes(tmp_path, seeds_dir, monkeypatch):
    out = tmp_path / "ds_crash"
    monkeypatch.setattr(pipeline, "_resolve_backend", lambda *a, **k: "loky")  # force staging

    def boom(*_a, **_k):
        raise RuntimeError("injected mid-generation failure")

    monkeypatch.setattr(pipeline, "generate_scene", boom)

    with pytest.raises(RuntimeError, match="injected mid-generation failure"):
        pipeline.run(_cfg(out, seeds_dir))

    stage_dir, manifest = _staging_paths(out)
    assert not stage_dir.exists(), "staging dir orphaned after a mid-generation crash"
    assert not manifest.exists(), "staging manifest orphaned after a mid-generation crash"


def test_staging_cleaned_up_on_success(tmp_path, seeds_dir, monkeypatch):
    out = tmp_path / "ds_ok"
    monkeypatch.setattr(pipeline, "_resolve_backend", lambda *a, **k: "loky")  # force staging

    msg.run(_cfg(out, seeds_dir))

    stage_dir, manifest = _staging_paths(out)
    assert not stage_dir.exists()
    assert not manifest.exists()
    assert list((out / "train" / "images").glob("*")), "no images were generated"
