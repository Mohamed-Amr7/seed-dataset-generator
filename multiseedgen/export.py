"""Label writers: YOLO, YOLO-seg, COCO, data.yaml, run manifest."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .constants import __version__
from .models import Config
from .optdeps import _HAVE_YAML, yaml

# COCO categories must carry a `supercategory`; with a flat (non-hierarchical) class
# layout every seed type is grouped under this single supercategory.
SEED_SUPERCATEGORY = "seed"


# --------------------------------------------------------------------------- #
# Exporters                                                                   #
# --------------------------------------------------------------------------- #
def write_yolo_label(path: Path, labels, W: int, H: int) -> None:
    lines = []
    for cls, x0, y0, x1, y1 in labels:
        cx = (x0 + x1) / 2.0 / W
        cy = (y0 + y1) / 2.0 / H
        bw = (x1 - x0) / W
        bh = (y1 - y0) / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _largest_piece(pieces):
    """Return the largest polygon piece (Nx2 int array) or None."""
    if not pieces:
        return None
    return max(pieces, key=lambda p: cv2.contourArea(p.astype(np.int32)))


def write_yolo_seg_label(path: Path, labels, polys, W: int, H: int) -> None:
    """YOLO-seg: one line per seed `class x1 y1 x2 y2 ...` (normalized). To keep a
    strict 1 line == 1 detection invariant we emit the largest visible polygon piece
    per seed; a seed with no polygon falls back to its box as a 4-point polygon.
    Empty file for negatives (matches the box-YOLO convention and the real labels)."""
    polys = polys if polys is not None else [None] * len(labels)
    lines = []
    for (cls, x0, y0, x1, y1), pieces in zip(labels, polys, strict=False):
        poly = _largest_piece(pieces)
        if poly is None or len(poly) < 3:
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
        pts = " ".join(
            f"{min(max(float(px), 0.0), W) / W:.6f} {min(max(float(py), 0.0), H) / H:.6f}"
            for px, py in poly
        )
        lines.append(f"{cls} {pts}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(out: Path, class_names: list[str]) -> None:
    names = ", ".join(f"'{n}'" for n in class_names)
    out.joinpath("data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names}]\n"
    )


def _extra_annotation(pv: dict, class_name: str | None) -> dict:
    """Flatten one provenance entry into a COCO `extra_annotations` payload: the seed
    class (id + name), the source seed file, the segmentation method that cut it out, and
    the per-placement transform (scale/rotation/shear/perspective + original cut-out dims)."""
    t = pv.get("transform") or {}
    return {
        "class_id": pv.get("class_id"),
        "class_name": class_name,
        "source": pv.get("source"),
        "seg_method": pv.get("seg_method"),
        "seed_index": pv.get("seed_index"),
        "scale": t.get("scale"),
        "rotation": t.get("rotation"),
        "shear": t.get("shear"),
        "perspective": t.get("perspective"),
        "orig_w": t.get("orig_w"),
        "orig_h": t.get("orig_h"),
    }


def _coco_ann(
    img_id: int, ann_id: int, label, pieces, pv, extra_annotations: bool, class_names: list[str]
) -> dict:
    """Build one COCO annotation dict (shared by both passes of write_coco)."""
    cls, x0, y0, x1, y1 = label
    ann = {
        "id": ann_id,
        "image_id": img_id,
        "category_id": int(cls) + 1,
        "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "iscrowd": 0,
    }
    seg, area = [], 0
    if pieces:
        for poly in pieces:
            flat = [int(v) for v in np.asarray(poly).reshape(-1)]
            if len(flat) >= 6:
                seg.append(flat)
                area += int(abs(cv2.contourArea(np.asarray(poly, np.int32))))
    if seg:
        ann["segmentation"] = seg
    ann["area"] = area if area > 0 else int((x1 - x0) * (y1 - y0))
    if extra_annotations and pv is not None:
        cls_i = int(cls)
        class_name = class_names[cls_i] if 0 <= cls_i < len(class_names) else None
        ann["extra_annotations"] = _extra_annotation(pv, class_name)
    return ann


def write_coco(
    path: Path, records, class_names: list[str], extra_annotations: bool = False
) -> None:
    """Stream the COCO JSON to disk in two passes over `records` (images, then
    annotations) so neither the full images/annotations lists nor a monolithic
    json.dumps string is ever materialised. Output is byte-identical to a single
    json.dumps({"images": [...], "annotations": [...], "categories": [...]})
    because each element is dumped with the default separators and joined by ", "."""

    def _pieces_prov(rec, labels):
        polys = rec[4] if len(rec) > 4 and rec[4] is not None else [None] * len(labels)
        prov = rec[5] if len(rec) > 5 and rec[5] is not None else [None] * len(labels)
        return polys, prov

    with open(path, "w") as f:
        f.write('{"images": [')
        for img_id, rec in enumerate(records, start=1):
            if img_id > 1:
                f.write(", ")
            f.write(
                json.dumps({"id": img_id, "file_name": rec[0], "width": rec[1], "height": rec[2]})
            )
        f.write('], "annotations": [')
        ann_id = 1
        for img_id, rec in enumerate(records, start=1):
            labels = rec[3]
            polys, prov = _pieces_prov(rec, labels)
            for label, pieces, pv in zip(labels, polys, prov, strict=False):
                if ann_id > 1:
                    f.write(", ")
                f.write(
                    json.dumps(
                        _coco_ann(img_id, ann_id, label, pieces, pv, extra_annotations, class_names)
                    )
                )
                ann_id += 1
        categories = [
            {"id": i + 1, "name": n, "supercategory": SEED_SUPERCATEGORY}
            for i, n in enumerate(class_names)
        ]
        f.write('], "categories": ' + json.dumps(categories) + "}")


def write_multilabel_csv(path: Path, rows, class_names: list[str]) -> None:
    """Per-image multi-label CSV alongside COCO (for classifier training etc.):
    `image,split,<class_0>,...,<class_n>` where each class column is the count of that
    class in the image (0 = absent). `rows` is a list of (image_rel, split, [class_id]).
    """
    import csv

    header = ["image", "split"] + list(class_names)
    n = len(class_names)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for image_rel, split, class_ids in rows:
            counts = [0] * n
            for c in class_ids:
                if 0 <= int(c) < n:
                    counts[int(c)] += 1
            w.writerow([image_rel, split] + counts)


def write_run_manifest(cfg: Config, class_names: list[str], summary: dict) -> None:
    """Write the exact resolved Config + a run summary. The file is itself a valid
    --config input, so a run can be reproduced with `--config run_config.yaml`."""
    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.model_dump().items()}
    manifest = {
        "tool": "seedgen.py",
        "version": __version__,
        "timestamp": datetime.now().astimezone().isoformat(),
        "command": " ".join(sys.argv),
        "config": cfg_dict,
        "summary": summary,
    }
    if _HAVE_YAML:
        (cfg.out / "run_config.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    else:
        (cfg.out / "run_config.json").write_text(json.dumps(manifest, indent=2))
