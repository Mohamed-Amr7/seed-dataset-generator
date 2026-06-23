"""Unit tests for the Pydantic :class:`Config` validation contract.

The migration to Pydantic v2 is determinism-critical: validators may **raise** but must
never *mutate* a value, because the pipeline gates RNG draws on "is this knob > 0 / set?"
(e.g. ``aug_shear > 0`` draws a shear; coercing ``0`` to ``None`` or vice-versa would
silently shift the random stream and change generated pixels). These tests lock both the
range/ordering checks and the "no silent coercion" guarantee, plus the two sanctioned
post-merge derivations.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_minimal_config_uses_documented_defaults(make_cfg):
    cfg = make_cfg()
    assert cfg.segment == "auto"
    assert cfg.aug_shear == 0.0
    assert cfg.backgrounds == ["solid", "gradient", "noise"]


@pytest.mark.parametrize(
    "overrides, needle",
    [
        (dict(min_seeds=10, max_seeds=5), "min_seeds"),
        (dict(scale_min=0.5, scale_max=0.1), "scale_min"),
        (dict(val_split=1.5), "val_split"),
        (dict(val_split=-0.1), "val_split"),
    ],
)
def test_range_and_ordering_violations_raise(make_cfg, overrides, needle):
    """Out-of-range / mis-ordered numeric fields must fail loudly, not be clamped."""
    with pytest.raises(ValidationError, match=needle):
        make_cfg(**overrides)


def test_unknown_field_is_rejected(make_cfg):
    """``extra='forbid'`` is the backstop against typo'd config keys."""
    with pytest.raises(ValidationError, match="not_a_real_field"):
        make_cfg(not_a_real_field=1)


def test_bare_string_is_wrapped_into_a_list(make_cfg):
    """A scalar for a list field (``backgrounds: solid``) becomes ``["solid"]`` -
    mirrors the argparse ``nargs='+'`` behaviour so YAML and CLI agree."""
    assert make_cfg(backgrounds="solid").backgrounds == ["solid"]


def test_class_probs_implies_class_balanced(make_cfg):
    """Sanctioned derivation: explicit per-class weights turn on balanced sampling."""
    cfg = make_cfg(class_probs=[0.5, 0.5], class_balanced=False)
    assert cfg.class_balanced is True


def test_yolo_seg_format_implies_masks(make_cfg):
    """Sanctioned derivation: YOLO-seg labels require per-seed polygons."""
    cfg = make_cfg(fmt="yolo-seg", masks=False)
    assert cfg.masks is True


@pytest.mark.parametrize("zero_field", ["aug_shear", "neg_frac", "blur_max_sigma"])
def test_zero_gated_knobs_are_not_coerced(make_cfg, zero_field):
    """Determinism guard: a knob left at 0 must stay exactly ``0.0`` (not ``None`` or a
    filled default). The pipeline reads these as ``> 0`` predicates that gate draws."""
    cfg = make_cfg(**{zero_field: 0})
    assert getattr(cfg, zero_field) == 0.0
    assert getattr(cfg, zero_field) is not None


def test_str_paths_are_coerced_to_path(make_cfg):
    """Path fields accept strings (CLI/JSON) and normalise to ``pathlib.Path``."""
    from pathlib import Path

    cfg = make_cfg(sources="some/dir", out="other/dir")
    assert isinstance(cfg.sources, Path)
    assert isinstance(cfg.out, Path)
