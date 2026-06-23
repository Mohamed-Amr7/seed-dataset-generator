#!/usr/bin/env python3
"""Segmentation cut-out cache-key contract.

The cut-out cache key (``SEG_VERSION`` + file content hash + ``_seg_params_signature``)
is a *wire format*: change which fields it includes, the ``round()`` digits, or the
``int()`` casts and every previously-cached cut-out silently misses (or, worse,
collides). This pins the signature string and the resulting key digest for a fixed
``(content_hash, cfg)`` across representative methods so the rework can't perturb it.

``_effective_rembg_provider`` probes the real onnxruntime device (GPU vs CPU), so it is
patched to a constant here to keep the contract environment-independent while still
exercising the rembg-device namespacing branch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("cv2")
msg = pytest.importorskip("multiseedgen")
from multiseedgen import Config  # noqa: E402
from multiseedgen import segmentation as seg

_GOLDEN = Path(__file__).resolve().parent / "golden" / "seg_cache_contract.json"
_CONTENT_HASH = "0123456789abcdef0123456789abcdef"  # fixed stand-in for a file hash


def _cfg(**overrides) -> Config:
    base = dict(sources=Path("s"), out=Path("o"), rembg_fallback=False)
    base.update(overrides)
    return Config(**base)


# (label, cfg kwargs, override-dict) tuples spanning the signature's branches.
_CASES = [
    ("otsu_no_fallback", dict(segment="otsu", rembg_fallback=False), None),
    ("otsu_with_fallback", dict(segment="otsu", rembg_fallback=True), None),
    ("grabcut", dict(segment="grabcut", grabcut_iters=5, morph_kernel=5), None),
    ("threshold", dict(segment="threshold", bg_tolerance=28.0), None),
    ("rembg", dict(segment="rembg"), None),
    (
        "sam",
        dict(segment="sam", sam_checkpoint=Path("models/sam.pth"), sam_model_type="vit_b"),
        None,
    ),
    (
        "otsu_override_grabcut",
        dict(segment="otsu"),
        {"method": "grabcut", "bg_tolerance": 12.0, "strip_shadow": False},
    ),
]


def _build_contract(monkeypatch) -> dict:
    # Patch where _seg_params_signature looks it up: it lives in segmentation.cache,
    # which imported the name from segmentation.session. Patching the __init__ or the
    # session binding would not affect the reference already bound inside cache.
    monkeypatch.setattr(
        "multiseedgen.segmentation.cache._effective_rembg_provider", lambda pref: "cpu"
    )
    out = {}
    for label, kw, override in _CASES:
        cfg = _cfg(**kw)
        sig = seg._seg_params_signature(cfg, override)
        key = seg._seg_cache_key(_CONTENT_HASH, cfg, override)
        key_bg = seg._seg_cache_key(_CONTENT_HASH, cfg, override, suffix="bg")
        out[label] = {"signature": json.loads(sig), "key": key, "key_bg": key_bg}
    out["_seg_version"] = seg.SEG_VERSION
    return out


def test_seg_cache_contract(monkeypatch):
    got = _build_contract(monkeypatch)

    if os.environ.get("UPDATE_GOLDEN") or not _GOLDEN.exists():
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"wrote seg-cache contract {_GOLDEN.name}")

    expected = json.loads(_GOLDEN.read_text())
    assert got == expected, (
        "segmentation cache-key wire format changed. If intentional (and you bumped "
        "SEG_VERSION), regenerate with UPDATE_GOLDEN=1; otherwise this would silently "
        f"invalidate or collide cached cut-outs.\nexpected={expected}\ngot={got}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
