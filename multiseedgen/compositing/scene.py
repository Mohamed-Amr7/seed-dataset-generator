"""Scene assembly: sample + place seeds, derive occlusion-aware labels/polys/provenance."""

from __future__ import annotations

import cv2
import numpy as np

from ..degrade import apply_degradation
from ..models import Config, Seed
from .backgrounds import make_background
from .blend import _iou, place_seed
from .transforms import transform_seed


def _mask_to_polygons(bin_mask: np.ndarray, min_area: float = 6.0) -> list:
    """Contour polygons (Nx2 int arrays) of a binary mask, lightly simplified.
    Used for the VISIBLE region of a seed (owner == seed_id), so polygons are
    occlusion-aware and in-frame by construction."""
    cnts, _ = cv2.findContours(
        bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 1.0, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(np.int32))
    return polys


def _build_class_sampler(cfg: Config, seeds: list[Seed]):
    """Return a `sample(rng) -> seed_index` that picks a CLASS first (uniform, or by
    cfg.class_probs weights in class-id order), then a seed uniformly within it. This
    counters source-count imbalance and lets a scene mix every class with a configurable
    probability. Returns None when balancing is off (caller keeps the legacy draw)."""
    if not cfg.class_balanced:
        return None
    by_class: dict[int, list[int]] = {}
    for i, s in enumerate(seeds):
        by_class.setdefault(s.class_id, []).append(i)
    classes = sorted(by_class)
    probs = cfg.class_probs
    if probs:
        w = np.array(
            [max(0.0, float(probs[c])) if c < len(probs) else 0.0 for c in classes],
            dtype=np.float64,
        )
        if w.sum() <= 0:
            w = np.ones(len(classes), dtype=np.float64)
    else:
        w = np.ones(len(classes), dtype=np.float64)
    w = w / w.sum()
    idx_lists = [by_class[c] for c in classes]

    def sample(rng: np.random.Generator) -> int:
        ci = int(rng.choice(len(classes), p=w))
        lst = idx_lists[ci]
        return int(lst[rng.integers(0, len(lst))])

    return sample


def generate_scene(
    cfg: Config,
    seeds: list[Seed],
    textures: list[np.ndarray],
    rng: np.random.Generator,
    negative: bool = False,
):
    """Return (image_uint8, labels, polys, prov).

    labels = [(class_id, x0, y0, x1, y1), ...]; polys is a parallel list (one entry
    per label, each a list of Nx2 int polygon pieces) when cfg.masks else None;
    prov is a parallel list of provenance dicts when cfg.provenance or
    cfg.extra_annotations else None.
    If ``negative`` is True, returns a background-only image with empty labels."""
    canvas = make_background(cfg, seeds, textures, rng)
    H, W = canvas.shape[:2]
    if negative:
        img = canvas
        if cfg.degrade:
            img = apply_degradation(canvas, cfg, rng)
        return np.clip(img, 0, 255).astype(np.uint8), [], None, None
    owner = np.full((H, W), -1, dtype=np.int32)

    # one light direction per scene (all seeds cast consistently) in scene mode
    if cfg.shadow and cfg.shadow_dir_mode == "scene":
        ang = float(rng.uniform(0, 2 * np.pi))
        light = (float(np.cos(ang)), float(np.sin(ang)))
    else:
        light = (1.0, 1.0)

    n = int(rng.integers(cfg.min_seeds, cfg.max_seeds + 1))
    placed: list[dict] = []
    boxes_for_iou: list[tuple[int, int, int, int]] = []

    class_sampler = _build_class_sampler(cfg, seeds)  # None unless class_balanced
    want_record = cfg.provenance or cfg.extra_annotations
    seed_id = 0
    for _ in range(n):
        seed_index = (
            class_sampler(rng) if class_sampler is not None else int(rng.integers(0, len(seeds)))
        )
        chosen = seeds[seed_index]
        if want_record:
            rgba, tparams = transform_seed(chosen, cfg, cfg.img_size, rng, record=True)
        else:
            rgba, tparams = transform_seed(chosen, cfg, cfg.img_size, rng), None
        ph, pw = rgba.shape[:2]

        # try a few positions (so overlap-limit mode has room to work)
        result = None
        for _try in range(8):
            x = int(rng.integers(int(-0.5 * pw), int(W - 0.5 * pw) + 1))
            y = int(rng.integers(int(-0.5 * ph), int(H - 0.5 * ph) + 1))

            if cfg.overlap_mode == "limit" and boxes_for_iou:
                provisional = (max(0, x), max(0, y), min(W, x + pw), min(H, y + ph))
                if any(_iou(provisional, b) > cfg.max_iou for b in boxes_for_iou):
                    continue

            result = place_seed(canvas, owner, rgba, x, y, seed_id, cfg, light)
            if result is not None:
                break

        if result is None:
            continue
        placed.append(
            {
                "seed_id": seed_id,
                "class_id": chosen.class_id,
                "seed_index": seed_index,
                "tparams": tparams,
                "source": chosen.source,
                "seg_method": chosen.method,
                **result,
            }
        )
        boxes_for_iou.append(result["box"])
        seed_id += 1

    # ---- occlusion filter: how much of each seed survived later pastes ----
    labels: list[tuple[int, int, int, int, int]] = []
    polys: list | None = [] if cfg.masks else None
    prov: list | None = [] if want_record else None
    for p in placed:
        visible = int((owner == p["seed_id"]).sum())
        if p["denom"] == 0:
            continue
        if visible / p["denom"] < cfg.min_visible_occluded:
            continue  # buried below threshold -> not labelled
        x0, y0, x1, y1 = p["box"]
        if min(x1 - x0, y1 - y0) < cfg.min_box_px:
            continue
        labels.append((p["class_id"], x0, y0, x1, y1))
        if polys is not None:
            pieces = _mask_to_polygons((owner == p["seed_id"]).astype(np.uint8))
            polys.append(pieces)
        if prov is not None:
            prov.append(
                {
                    "seed_index": p["seed_index"],
                    "class_id": p["class_id"],
                    "box": [x0, y0, x1, y1],
                    "transform": p["tparams"],
                    "source": p["source"],
                    "seg_method": p["seg_method"],
                }
            )

    img = apply_degradation(canvas, cfg, rng) if cfg.degrade else canvas
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, labels, polys, prov
