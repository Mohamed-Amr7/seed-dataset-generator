"""Unit tests for the QA / preview renderers (``multiseedgen.qa``).

The pure chart helpers are checked directly; ``write_qa_report`` and ``make_previews``
are driven against a hand-built two-image dataset on disk so their label-readback and
montage paths run without a full pipeline invocation.
"""

from __future__ import annotations

import types

import cv2
import numpy as np
import pytest

from multiseedgen.models import _QAStats
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
    check_segmentation([seed_cutout, seed_cutout], ["seed"], tmp_path, n=5)

    checks = list((tmp_path / "seed_check").glob("*.png"))
    assert len(checks) == 2  # min(n, len(seeds))
