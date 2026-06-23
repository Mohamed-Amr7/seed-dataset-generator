"""Unit tests for the rembg session helpers in ``segmentation/session.py``.

Covers the device-adaptivity additions:
  * the VRAM-scaled GPU job count (``_free_vram_gb`` / ``_auto_gpu_seg_jobs``), and
  * the mid-run CUDA OOM -> CPU fallback in ``_mask_rembg``.

Everything runs on CPU with no real GPU / onnxruntime / rembg: ``nvidia-smi`` and the
``rembg`` module are faked via monkeypatch, so the suite stays fast and hermetic.
"""

from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest

from multiseedgen.segmentation import session


# --------------------------------------------------------------------------- #
# VRAM probe + auto job count (A1)                                            #
# --------------------------------------------------------------------------- #
def _fake_smi(monkeypatch, free_mib: str | None) -> None:
    """Make ``session._free_vram_gb`` observe a given nvidia-smi result.

    ``free_mib=None`` simulates a host without nvidia-smi on PATH."""
    if free_mib is None:
        monkeypatch.setattr(session.shutil, "which", lambda _name: None)
        return
    monkeypatch.setattr(session.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        session.subprocess, "run", lambda *_a, **_k: types.SimpleNamespace(stdout=f"{free_mib}\n")
    )


def test_free_vram_gb_parses_first_gpu(monkeypatch):
    _fake_smi(monkeypatch, "6144")  # 6144 MiB == 6.0 GiB
    assert session._free_vram_gb() == pytest.approx(6.0)


def test_free_vram_gb_none_without_nvidia_smi(monkeypatch):
    _fake_smi(monkeypatch, None)
    assert session._free_vram_gb() is None


def test_free_vram_gb_none_on_probe_error(monkeypatch):
    monkeypatch.setattr(session.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def boom(*_a, **_k):
        raise subprocess.SubprocessError("nvidia-smi failed")

    monkeypatch.setattr(session.subprocess, "run", boom)
    assert session._free_vram_gb() is None


def test_auto_gpu_seg_jobs_scales_to_vram(monkeypatch):
    monkeypatch.setattr(session, "_AUTO_GPU_SEG_JOBS", None)  # reset per-process memo
    monkeypatch.setattr(session.os, "cpu_count", lambda: 64)  # don't let cores clamp
    _fake_smi(monkeypatch, "6144")  # 6 GiB -> round(6 * 0.7 / 0.9) == round(4.67) == 5
    assert session._auto_gpu_seg_jobs() == 5


def test_auto_gpu_seg_jobs_clamped_by_cpu_count(monkeypatch):
    monkeypatch.setattr(session, "_AUTO_GPU_SEG_JOBS", None)
    monkeypatch.setattr(session.os, "cpu_count", lambda: 2)
    _fake_smi(monkeypatch, "24576")  # 24 GiB -> ~18.7 sessions, clamped to the 2 cores
    assert session._auto_gpu_seg_jobs() == 2


def test_auto_gpu_seg_jobs_fallback_without_probe(monkeypatch):
    monkeypatch.setattr(session, "_AUTO_GPU_SEG_JOBS", None)
    _fake_smi(monkeypatch, None)
    assert session._auto_gpu_seg_jobs() == 4  # documented 6 GB baseline


def test_auto_gpu_seg_jobs_is_at_least_one(monkeypatch):
    monkeypatch.setattr(session, "_AUTO_GPU_SEG_JOBS", None)
    monkeypatch.setattr(session.os, "cpu_count", lambda: 8)
    _fake_smi(monkeypatch, "512")  # 0.5 GiB -> round(0.39) == 0 -> clamped up to 1
    assert session._auto_gpu_seg_jobs() == 1


def test_auto_gpu_seg_jobs_is_memoized(monkeypatch):
    monkeypatch.setattr(session, "_AUTO_GPU_SEG_JOBS", None)
    monkeypatch.setattr(session.os, "cpu_count", lambda: 64)
    _fake_smi(monkeypatch, "6144")
    first = session._auto_gpu_seg_jobs()
    # Even if the probe would now report something different, the memo wins.
    _fake_smi(monkeypatch, "24576")
    assert session._auto_gpu_seg_jobs() == first


# --------------------------------------------------------------------------- #
# CUDA OOM detection (A2)                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: out of memory",
        "Failed to allocate memory in BFC_Arena",
        "cudaErrorMemoryAllocation",
        "CUDA failure 2: out of memory",
    ],
)
def test_is_cuda_oom_true_for_oom_messages(message):
    assert session._is_cuda_oom(RuntimeError(message)) is True


@pytest.mark.parametrize("message", ["invalid image shape", "broken pipe", "unsupported model"])
def test_is_cuda_oom_false_for_unrelated_errors(message):
    assert session._is_cuda_oom(ValueError(message)) is False


# --------------------------------------------------------------------------- #
# rembg remove() OOM -> CPU fallback (A2)                                     #
# --------------------------------------------------------------------------- #
def _force_cuda_session(monkeypatch) -> None:
    """Put session.py in the 'currently running on CUDA' state without a real GPU."""
    monkeypatch.setattr(session, "_REMBG_PROVIDER", "cuda")
    monkeypatch.setattr(session, "_REMBG_SESSION", object())  # skip the build branch
    monkeypatch.setattr(
        session, "_ort_available", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


def _install_fake_rembg(monkeypatch, remove_fn) -> None:
    fake = types.ModuleType("rembg")
    fake.new_session = lambda *_a, **_k: object()
    fake.remove = remove_fn
    monkeypatch.setitem(sys.modules, "rembg", fake)


def test_mask_rembg_falls_back_to_cpu_on_cuda_oom(monkeypatch):
    _force_cuda_session(monkeypatch)
    h, w = 8, 8
    calls = {"n": 0}

    def fake_remove(rgb, session=None):  # noqa: A002 - mirrors rembg.remove signature
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA error: out of memory (bfc_arena)")
        out = np.zeros((h, w, 4), np.uint8)
        out[..., 3] = 200
        return out

    _install_fake_rembg(monkeypatch, fake_remove)

    alpha = session._mask_rembg(np.full((h, w, 3), 120, np.uint8))

    assert calls["n"] == 2  # one GPU attempt + exactly one CPU retry
    assert alpha is not None and alpha.shape == (h, w)
    assert int(alpha.max()) == 200
    assert session._REMBG_PROVIDER == "cpu"  # worker permanently dropped to CPU


def test_mask_rembg_propagates_non_oom_error(monkeypatch):
    _force_cuda_session(monkeypatch)
    calls = {"n": 0}

    def fake_remove(rgb, session=None):  # noqa: A002
        calls["n"] += 1
        raise ValueError("corrupt input image")

    _install_fake_rembg(monkeypatch, fake_remove)

    with pytest.raises(ValueError, match="corrupt input image"):
        session._mask_rembg(np.full((8, 8, 3), 120, np.uint8))
    assert calls["n"] == 1  # a non-OOM error is NOT retried on CPU
