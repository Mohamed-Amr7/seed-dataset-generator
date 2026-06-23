#!/usr/bin/env python3
"""Config-parsing contract for MultiSeedGen.

Locks the *current* behavior of the CLI/config layer (``cliconfig.py`` + the ``Config``
dataclass) so the Pydantic v2 migration can be proven behavior-preserving:

  * a full resolved-``Config`` snapshot per shipped ``configs/*.yaml`` (the strongest
    guard - every one of the 89 fields' resolved value is pinned), committed under
    ``tests/golden/config_<name>.json``;
  * the tricky parsing behaviors a snapshot can't express: the five ``--no-*``
    store_false flags, ``--format``->``fmt``, ``nargs="+"`` lists, the
    ``defaults < file < only-explicit-CLI`` precedence, nested-section aliasing, the
    two implied-derivations, and fail-fast on unknown/conflicting keys.

Regenerate the snapshots deliberately with ``UPDATE_GOLDEN=1 pytest`` (only when a
config change is intended).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest

pytest.importorskip("yaml")
msg = pytest.importorskip("multiseedgen")
from multiseedgen.cliconfig import _load_config_file, parse_args  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_CONFIGS_DIR = _REPO_ROOT / "configs"
_CONFIG_FILES = ["example", "root", "wheat", "test", "soybean_healthy"]

# Errors used by the fail-fast paths. Today they are sys.exit (SystemExit); Phase 2
# introduces ConfigError - accept both so this contract survives that intended change.
try:
    from multiseedgen.errors import ConfigError  # type: ignore

    _EXIT = (SystemExit, ConfigError)
except Exception:  # pragma: no cover - errors.py not added yet
    _EXIT = (SystemExit,)


def _as_jsonable(cfg) -> dict:
    """Resolved Config -> plain JSON dict, working for both the dataclass (today) and
    the Pydantic model (post-migration). Paths -> str; declaration order preserved."""
    if hasattr(cfg, "model_dump"):
        d = cfg.model_dump()
    elif is_dataclass(cfg):
        d = asdict(cfg)
    else:  # pragma: no cover
        d = dict(vars(cfg))
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def _write_yaml(tmp_path: Path, body: str, name: str = "c.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


# --------------------------------------------------------------------------- #
# Full resolved-config snapshots for every shipped profile                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _CONFIG_FILES)
def test_shipped_config_snapshot(name):
    cfg = parse_args(["--config", str(_CONFIGS_DIR / f"{name}.yaml")])
    got = _as_jsonable(cfg)
    # `config` is the absolute path to the --config file itself, which is machine-specific
    # (e.g. /home/you/repo/... vs CI's /app/...). Pin the resolved SETTINGS, not where the
    # repo happens to live, so the snapshot is portable across dev boxes and CI.
    got.pop("config", None)

    snap = _GOLDEN_DIR / f"config_{name}.json"
    if os.environ.get("UPDATE_GOLDEN") or not snap.exists():
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"wrote config snapshot {snap.name}")

    expected = json.loads(snap.read_text())
    assert got.keys() == expected.keys(), (
        f"{name}: field set changed; "
        f"only-in-snapshot={sorted(set(expected) - set(got))[:8]} "
        f"only-in-run={sorted(set(got) - set(expected))[:8]}"
    )
    diffs = {k: (expected[k], got[k]) for k in expected if expected[k] != got[k]}
    assert not diffs, f"{name}: resolved Config drifted from snapshot: {diffs}"


# --------------------------------------------------------------------------- #
# Nested-section aliasing (example.yaml uses the nested layout)               #
# --------------------------------------------------------------------------- #
def test_nested_aliases_resolve():
    cfg = parse_args(["--config", str(_CONFIGS_DIR / "example.yaml")])
    assert cfg.segment == "auto"  # segmentation.method -> segment
    assert cfg.fmt == "both"  # export.format -> fmt
    assert cfg.max_rotation == 180.0  # augmentation.rotation -> max_rotation
    assert cfg.aug_gamma == 0.2  # augmentation.gamma -> aug_gamma
    assert cfg.aug_shear == 4.0  # augmentation.shear -> aug_shear
    assert cfg.grabcut_iters == 5  # segmentation.grabcut.iterations
    assert cfg.seg_conf_thresh == 0.55  # segmentation.conf_thresh
    assert cfg.val_seed_holdout == 0.20  # split.val_seed_holdout
    assert cfg.class_balanced is True  # sampling.class_balanced
    assert cfg.extra_annotations is True  # export.extra_annotations


# --------------------------------------------------------------------------- #
# store_false negated flags + --format alias + nargs lists                    #
# --------------------------------------------------------------------------- #
def test_store_false_flags_default_true(tmp_path):
    cfg = parse_args(["--sources", "s", "--out", "o"])
    assert (cfg.flip, cfg.shadow, cfg.rembg_fallback, cfg.strip_shadow, cfg.use_cache) == (
        True,
        True,
        True,
        True,
        True,
    )


@pytest.mark.parametrize(
    "flag,attr",
    [
        ("--no-flip", "flip"),
        ("--no-shadow", "shadow"),
        ("--no-rembg-fallback", "rembg_fallback"),
        ("--no-strip-shadow", "strip_shadow"),
        ("--no-cache", "use_cache"),
    ],
)
def test_store_false_flags_negate(flag, attr):
    cfg = parse_args(["--sources", "s", "--out", "o", flag])
    assert getattr(cfg, attr) is False


def test_format_alias_and_nargs_lists():
    cfg = parse_args(
        [
            "--sources",
            "s",
            "--out",
            "o",
            "--format",
            "both",
            "--backgrounds",
            "solid",
            "noise",
            "--class-probs",
            "1",
            "2",
            "3",
        ]
    )
    assert cfg.fmt == "both"
    assert cfg.backgrounds == ["solid", "noise"]
    assert cfg.class_probs == [1.0, 2.0, 3.0]
    assert cfg.class_balanced is True  # class_probs implies class_balanced


def test_yolo_seg_implies_masks():
    cfg = parse_args(["--sources", "s", "--out", "o", "--format", "yolo-seg"])
    assert cfg.fmt == "yolo-seg" and cfg.masks is True


# --------------------------------------------------------------------------- #
# defaults < file < only-explicit-CLI precedence                              #
# --------------------------------------------------------------------------- #
def test_file_overrides_defaults(tmp_path):
    f = _write_yaml(
        tmp_path, "config:\n  sources: s\n  out: o\n  num_images: 333\n  min_seeds: 11\n"
    )
    cfg = parse_args(["--config", str(f)])
    assert cfg.num_images == 333 and cfg.min_seeds == 11  # file wins over defaults


def test_cli_overrides_file_only_when_explicit(tmp_path):
    f = _write_yaml(
        tmp_path, "config:\n  sources: s\n  out: o\n  num_images: 333\n  min_seeds: 11\n"
    )
    cfg = parse_args(["--config", str(f), "--num-images", "7"])
    assert cfg.num_images == 7  # explicit CLI wins
    assert cfg.min_seeds == 11  # untouched file value survives (not clobbered)


# --------------------------------------------------------------------------- #
# fail-fast on unknown / conflicting keys                                     #
# --------------------------------------------------------------------------- #
def test_unknown_key_fails(tmp_path):
    f = _write_yaml(tmp_path, "config:\n  sources: s\n  out: o\n  not_a_field: 1\n")
    with pytest.raises(_EXIT):
        parse_args(["--config", str(f)])


def test_flat_nested_conflict_fails(tmp_path):
    f = _write_yaml(
        tmp_path, "sources: s\nout: o\nsegment: otsu\nsegmentation:\n  method: grabcut\n"
    )
    with pytest.raises(_EXIT):
        _load_config_file(f)


def test_missing_sources_out_fails():
    with pytest.raises(_EXIT):
        parse_args([])  # neither --sources nor --out nor --config


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
