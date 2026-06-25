"""Unit tests for the annotation writers (``multiseedgen.export``).

Each writer is exercised in isolation against a tiny hand-built record set and the
output is parsed back, so the tests assert the *on-disk contract* (YOLO normalisation,
COCO structure, the multilabel count matrix) rather than re-deriving the implementation.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from multiseedgen import export


def test_write_yolo_label_normalises_box_to_centre_form(tmp_path):
    # Arrange: one box on a 100x100 canvas -> known normalised centre/size.
    path = tmp_path / "000.txt"

    # Act
    export.write_yolo_label(path, [(2, 10, 20, 30, 60)], W=100, H=100)

    # Assert: "<cls> cx cy bw bh" with 6-dp normalisation.
    assert path.read_text() == "2 0.200000 0.400000 0.200000 0.400000\n"


def test_write_yolo_label_is_empty_for_negatives(tmp_path):
    """A background-only scene yields an empty (but present) label file."""
    path = tmp_path / "neg.txt"
    export.write_yolo_label(path, [], W=64, H=64)
    assert path.read_text() == ""


def test_write_yolo_seg_label_falls_back_to_box_when_no_polygon(tmp_path):
    """A seed with no polygon piece is emitted as its 4-corner box polygon, so the
    1-line-per-detection invariant holds even without segmentation masks."""
    path = tmp_path / "seg.txt"
    export.write_yolo_seg_label(path, [(1, 0, 0, 50, 50)], polys=None, W=100, H=100)

    line = path.read_text().strip()
    coords = line.split()
    assert coords[0] == "1"
    assert len(coords) == 1 + 8  # class + 4 (x,y) corners


def test_write_data_yaml_round_trips_classes(tmp_path):
    export.write_data_yaml(tmp_path, ["wheat", "root", "soy"])
    text = (tmp_path / "data.yaml").read_text()
    assert "nc: 3" in text
    assert "names: ['wheat', 'root', 'soy']" in text
    assert "train: train/images" in text


def test_write_multilabel_csv_counts_classes_per_image(tmp_path):
    path = tmp_path / "multilabel.csv"
    rows = [
        ("train/images/a.jpg", "train", [0, 0, 1]),  # 2x class0, 1x class1
        ("val/images/b.jpg", "val", []),  # negative
    ]
    export.write_multilabel_csv(path, rows, ["c0", "c1"])

    with open(path) as f:
        parsed = list(csv.reader(f))
    assert parsed[0] == ["image", "split", "c0", "c1"]
    assert parsed[1] == ["train/images/a.jpg", "train", "2", "1"]
    assert parsed[2] == ["val/images/b.jpg", "val", "0", "0"]


def test_write_coco_emits_valid_structure(tmp_path):
    path = tmp_path / "annotations.json"
    records = [("train/images/000.jpg", 100, 100, [(0, 10, 10, 30, 40)])]

    export.write_coco(path, records, ["seed"])

    data = json.loads(path.read_text())
    assert {"images", "annotations", "categories"} <= data.keys()
    assert data["categories"] == [{"id": 1, "name": "seed", "supercategory": "seed"}]
    ann = data["annotations"][0]
    assert ann["bbox"] == [10, 10, 20, 30]  # x, y, w, h
    assert ann["category_id"] == 1  # COCO is 1-indexed


def test_write_coco_includes_polygon_segmentation_when_present(tmp_path):
    path = tmp_path / "annotations.json"
    poly = [np.array([[10, 10], [30, 10], [30, 30], [10, 30]], np.int32)]
    records = [("img.jpg", 100, 100, [(0, 10, 10, 30, 30)], [poly])]

    export.write_coco(path, records, ["seed"])

    ann = json.loads(path.read_text())["annotations"][0]
    assert "segmentation" in ann
    assert ann["segmentation"][0] == [10, 10, 30, 10, 30, 30, 10, 30]


def test_write_coco_flattens_provenance_into_extra_annotations(tmp_path):
    path = tmp_path / "annotations.json"
    prov = [
        {
            "source": "wheat/seed_0.png",
            "seg_method": "otsu",
            "class_id": 1,
            "transform": {"scale": 0.1},
        }
    ]
    records = [("img.jpg", 100, 100, [(1, 0, 0, 10, 10)], [None], prov)]

    export.write_coco(path, records, ["soybean", "wheat"], extra_annotations=True)

    ann = json.loads(path.read_text())["annotations"][0]
    assert ann["extra_annotations"]["source"] == "wheat/seed_0.png"
    assert ann["extra_annotations"]["seg_method"] == "otsu"
    assert ann["extra_annotations"]["scale"] == pytest.approx(0.1)
    # class provenance: id carried from the placement, name resolved from class_names
    assert ann["extra_annotations"]["class_id"] == 1
    assert ann["extra_annotations"]["class_name"] == "wheat"
