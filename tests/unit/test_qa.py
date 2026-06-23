"""Unit tests for the QA / preview renderers (``multiseedgen.qa``).

The pure chart helpers are checked directly; ``write_qa_report`` and ``make_previews``
are driven against a hand-built two-image dataset on disk so their label-readback and
montage paths run without a full pipeline invocation.
"""

from __future__ import annotations

import json
import types

import cv2
import numpy as np
import pytest

from multiseedgen.models import SegResult, _QAStats
from multiseedgen.qa import (
    _bar_chart,
    _checkerboard,
    _hist_chart,
    check_segmentation,
    make_previews,
    write_qa_report,
)


@pytest.fixture
def tiny_dataset(make_cfg):
    """A 2-image YOLO dataset written to ``cfg.out`` (images + box labels), enough for
    the QA readback paths. Returns ``(cfg, class_names)``."""
    cfg = make_cfg(fmt="yolo", img_size=64)
    img_dir = cfg.out / "train" / "images"
    lbl_dir = cfg.out / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(2):
        img = np.full((64, 64, 3), 180, np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:03d}.{cfg.img_ext}"), img)
        # one centred box, normalised YOLO form
        (lbl_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    return cfg, ["seed"]


def _summary(**over) -> dict:
    base = dict(
        num_images=2,
        n_train=2,
        n_val=0,
        total_negatives=0,
        total_boxes=2,
        avg_boxes_per_image=1.0,
        n_classes=1,
        n_seeds=3,
        wall_seconds=0.5,
        workers=1,
    )
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# Pure chart helpers                                                          #
# --------------------------------------------------------------------------- #
def test_bar_chart_returns_uint8_image_of_requested_size():
    img = _bar_chart([1, 2, 3], ["a", "b", "c"], "title", w=200, h=120)
    assert img.shape == (120, 200, 3)
    assert img.dtype == np.uint8


def test_hist_chart_handles_empty_samples():
    """Empty input must not crash - it degrades to an empty bar chart."""
    img = _hist_chart([], "nothing")
    assert img.ndim == 3 and img.shape[2] == 3


def test_checkerboard_is_two_toned():
    board = _checkerboard(48, 48, sq=12)
    assert board.shape == (48, 48, 3)
    assert len({tuple(c) for c in board.reshape(-1, 3)}) == 2  # exactly two colours


# --------------------------------------------------------------------------- #
# _QAStats accumulation                                                       #
# --------------------------------------------------------------------------- #
def test_qastats_accumulates_boxes_and_negatives():
    qa = _QAStats()
    qa.add("train", types.SimpleNamespace(labels=[(0, 0, 0, 10, 20), (1, 0, 0, 5, 5)]))
    qa.add("train", types.SimpleNamespace(labels=[]))  # a negative scene

    assert qa.boxes_per_image == [2, 0]
    assert qa.negatives == 1
    assert qa.class_counts == {0: 1, 1: 1}


# --------------------------------------------------------------------------- #
# Report + previews (read exported labels back off disk)                      #
# --------------------------------------------------------------------------- #
def test_write_qa_report_emits_report_and_charts(tiny_dataset):
    cfg, class_names = tiny_dataset
    qa = _QAStats()
    qa.add("train", types.SimpleNamespace(labels=[(0, 0, 0, 10, 20)]))

    write_qa_report(cfg, class_names, qa, _summary())

    qadir = cfg.out / "qa"
    report = (qadir / "report.md").read_text()
    assert "# Dataset QA report" in report
    assert "images: **2**" in report
    assert (qadir / "class_balance.png").exists()
    assert (qadir / "montage.png").exists()  # built from the 2 on-disk composites


def test_make_previews_writes_one_annotated_image_per_input(tiny_dataset):
    cfg, class_names = tiny_dataset
    make_previews(cfg, class_names, k=2)

    previews = list((cfg.out / "preview").glob(f"*.{cfg.img_ext}"))
    assert len(previews) == 2


def test_check_segmentation_writes_one_png_per_seed(make_cfg, seed_cutout, tmp_path):
    # full-run mode (num_images > 0): kept-only output, original filenames, no manifest
    cfg = make_cfg(out=tmp_path, num_images=2)
    check_segmentation([seed_cutout, seed_cutout], ["seed"], cfg, n=5)

    d = tmp_path / "seed_check"
    checks = list(d.glob("seed_*.png"))
    assert len(checks) == 2  # min(n, len(seeds))
    assert not (d / "manifest.json").exists()  # manifest is tuner-mode only


def test_check_segmentation_tuner_mode_writes_manifest(make_cfg, seed_cutout, tmp_path):
    """Tuner mode (num_images == 0) stamps kept cut-outs and writes a manifest with
    confidence; a skipped SegResult is rendered from its on-disk original."""
    cfg = make_cfg(out=tmp_path, sources=tmp_path / "src", num_images=0)
    cfg.sources.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cfg.sources / "bad.png"), np.full((30, 30, 3), 120, np.uint8))

    kept = seed_cutout
    kept.confidence = 0.91
    skipped = SegResult(
        patch=None,
        report={
            "skipped": True,
            "skip_reason": "low_confidence",
            "confidence": 0.2,
            "method": "auto",
            "source": "bad.png",
        },
        source="bad.png",
        class_id=0,
    )
    check_segmentation([kept], ["seed"], cfg, n=5, seg_results=[skipped])

    manifest = json.loads((tmp_path / "seed_check" / "manifest.json").read_text())
    statuses = sorted(e["status"] for e in manifest)
    assert statuses == ["kept", "skipped"]
    kept_entry = next(e for e in manifest if e["status"] == "kept")
    assert kept_entry["confidence"] == 0.91
    assert (tmp_path / "seed_check" / kept_entry["file"]).exists()
