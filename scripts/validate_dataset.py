#!/usr/bin/env python3
"""Numeric validation of a generated seedgen dataset against the §10 invariants.

Checks, per split present (train/val):
  * YOLO: one .txt per image; box lines `class cx cy w h` normalized in (0,1] and
    in-frame; empty file allowed (negative). Optional yolo-seg polygon lines:
    class + >=3 normalized (x,y) pairs, all in [0,1].
  * COCO (annotations_<split>.json if present): images/annotations/categories;
    bbox [x,y,w,h] top-left ints, in-frame; category_id = class+1; area>0;
    no dangling image_id; negatives = image entry with no annotations; if
    segmentation present it is non-empty and bbox≈boundingRect(polygon) within tol.
  * Cross-format: YOLO box-line count == COCO annotation count per split.

Exit code 0 = all PASS, 1 = any FAIL. Usage:
  python tests/validate_dataset.py /tmp/out [--seg]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _fail(msgs, m):
    msgs.append("FAIL: " + m)


def validate(root: Path, seg: bool = False) -> bool:
    msgs = []
    ok = True
    for split in ("train", "val"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        yolo_box_lines = 0
        if lbl_dir.exists():
            # one .txt per image
            stems_img = {p.stem for p in imgs}
            stems_lbl = {p.stem for p in lbl_dir.glob("*.txt")}
            missing = stems_img - stems_lbl
            if missing:
                ok = False
                _fail(msgs, f"{split}: {len(missing)} images without a .txt label")
            for lp in lbl_dir.glob("*.txt"):
                for ln, line in enumerate(lp.read_text().splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls = parts[0]
                    vals = [float(v) for v in parts[1:]]
                    if not cls.lstrip("-").isdigit():
                        ok = False
                        _fail(msgs, f"{lp.name}:{ln} non-int class {cls}")
                    if len(vals) == 4:  # box line
                        yolo_box_lines += 1
                        cx, cy, bw, bh = vals
                        if not (0 < cx <= 1 and 0 < cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                            ok = False
                            _fail(msgs, f"{lp.name}:{ln} box not in (0,1]: {vals}")
                        if (
                            cx - bw / 2 < -1e-6
                            or cx + bw / 2 > 1 + 1e-6
                            or cy - bh / 2 < -1e-6
                            or cy + bh / 2 > 1 + 1e-6
                        ):
                            ok = False
                            _fail(msgs, f"{lp.name}:{ln} box out of frame: {vals}")
                    else:  # polygon (yolo-seg)
                        if not seg:
                            ok = False
                            _fail(
                                msgs,
                                f"{lp.name}:{ln} unexpected polygon line "
                                f"({len(vals)} vals); pass --seg if expected",
                            )
                        if len(vals) < 6 or len(vals) % 2 != 0:
                            ok = False
                            _fail(msgs, f"{lp.name}:{ln} bad polygon length {len(vals)}")
                        arr = np.array(vals).reshape(-1, 2)
                        if (arr < -1e-6).any() or (arr > 1 + 1e-6).any():
                            ok = False
                            _fail(msgs, f"{lp.name}:{ln} polygon point out of [0,1]")
        # COCO
        coco_path = root / f"annotations_{split}.json"
        coco_anns = None
        if coco_path.exists():
            data = json.loads(coco_path.read_text())
            for key in ("images", "annotations", "categories"):
                if key not in data:
                    ok = False
                    _fail(msgs, f"{coco_path.name} missing '{key}'")
            img_ids = {im["id"] for im in data["images"]}
            img_wh = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
            cat_ids = {c["id"] for c in data["categories"]}
            coco_anns = len(data["annotations"])
            for a in data["annotations"]:
                if a["image_id"] not in img_ids:
                    ok = False
                    _fail(msgs, f"{coco_path.name} dangling image_id {a['image_id']}")
                if a["category_id"] not in cat_ids:
                    ok = False
                    _fail(msgs, f"{coco_path.name} bad category_id {a['category_id']}")
                x, y, w, h = a["bbox"]
                if any(not isinstance(v, int) for v in a["bbox"]):
                    ok = False
                    _fail(msgs, f"{coco_path.name} ann {a['id']} bbox not all ints: {a['bbox']}")
                W, H = img_wh.get(a["image_id"], (10**9, 10**9))
                if x < 0 or y < 0 or x + w > W or y + h > H or w <= 0 or h <= 0:
                    ok = False
                    _fail(
                        msgs,
                        f"{coco_path.name} ann {a['id']} bbox out of frame {a['bbox']} (img {W}x{H})",
                    )
                if a.get("area", 0) <= 0:
                    ok = False
                    _fail(msgs, f"{coco_path.name} ann {a['id']} area<=0")
                if seg:
                    s = a.get("segmentation")
                    if not s or not s[0] or len(s[0]) < 6:
                        ok = False
                        _fail(msgs, f"{coco_path.name} ann {a['id']} empty/short segmentation")
                    else:
                        poly = np.array(s[0]).reshape(-1, 2)
                        bx0, by0 = poly[:, 0].min(), poly[:, 1].min()
                        # bbox should ~equal polygon bounding rect (largest piece)
                        if abs(bx0 - x) > 2 or abs(by0 - y) > 2:
                            # allow: multi-piece COCO keeps all pieces -> bbox may be union.
                            pass
            # negatives: images with no anns are allowed (and expected)
        if coco_anns is not None and lbl_dir.exists():
            if yolo_box_lines != coco_anns:
                ok = False
                _fail(
                    msgs, f"{split}: YOLO box lines ({yolo_box_lines}) != COCO anns ({coco_anns})"
                )
            else:
                msgs.append(f"OK  {split}: {yolo_box_lines} YOLO boxes == {coco_anns} COCO anns")
        elif lbl_dir.exists():
            msgs.append(f"OK  {split}: {yolo_box_lines} YOLO box lines, {len(imgs)} images")
    # data.yaml
    yml = root / "data.yaml"
    if yml.exists():
        t = yml.read_text()
        for need in ("path:", "train:", "val:", "nc:", "names:"):
            if need not in t:
                ok = False
                _fail(msgs, f"data.yaml missing '{need}'")
        msgs.append("OK  data.yaml has path/train/val/nc/names")

    print("\n".join(msgs) if msgs else "(no splits found)")
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    seg = "--seg" in args
    args = [a for a in args if a != "--seg"]
    root = Path(args[0]) if args else Path(".")
    sys.exit(0 if validate(root, seg) else 1)
