from __future__ import annotations
"""Optional Segment Anything Model (SAM) segmentation backend.

Import-guarded: ``torch`` + ``segment_anything`` are NOT hard dependencies of
MultiSeedGen. They are imported lazily on first use, so the package builds and every
other segmentation method keeps working with nothing installed. To enable SAM:

    pip install segment-anything torch torchvision
    # then download a checkpoint and pass --sam-checkpoint / segmentation.sam.checkpoint

Checkpoint sizes (see AUGMENTATION.md): vit_b ~358 MB (fits a 6 GB GPU / runs on CPU),
vit_l ~1.2 GB, vit_h ~2.4 GB. The single-seed crops this tool segments are small, so
``vit_b`` is the recommended default.

The class exposes the three prompt modes the checklist asks for; for the pipeline's
single-seed crops ``segment_automatic`` (pick the central seed mask) is the default,
``segment_with_box`` (near-full-frame rect, like grabcut) is a fast deterministic
alternative, and ``segment_with_points`` takes explicit fg/bg clicks.
"""

from typing import List, Optional, Tuple
import numpy as np


def sam_available() -> bool:
    """True iff torch + segment_anything import cleanly (deps installed)."""
    try:
        import torch  # noqa: F401
        import segment_anything  # noqa: F401
        return True
    except Exception:
        return False


def _bbox_from_mask(mask: np.ndarray) -> List[int]:
    """[x, y, w, h] of the mask's tight extent (COCO order); zeros if empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


class SAMSegmenter:
    """Thin wrapper over segment_anything's SamPredictor + SamAutomaticMaskGenerator.

    All three methods return ``(mask uint8 HxW in {0,255}, bbox [x,y,w,h])``."""

    def __init__(self, checkpoint, model_type: str = "vit_b",
                 device: Optional[str] = None, points_per_side: int = 32,
                 pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.95):
        import torch
        from segment_anything import (sam_model_registry, SamPredictor,
                                       SamAutomaticMaskGenerator)
        if model_type not in sam_model_registry:
            raise ValueError(f"unknown SAM model_type {model_type!r}; "
                             f"expected one of {list(sam_model_registry)}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_type = model_type
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        self._sam = sam
        self._predictor = SamPredictor(sam)
        self._auto = SamAutomaticMaskGenerator(
            sam, points_per_side=int(points_per_side),
            pred_iou_thresh=float(pred_iou_thresh),
            stability_score_thresh=float(stability_score_thresh))

    def segment_with_points(self, image: np.ndarray, points, labels):
        """points = list of (x, y); labels = list of 1 (fg) / 0 (bg). RGB image."""
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True)
        m = (masks[int(np.argmax(scores))].astype(np.uint8)) * 255
        return m, _bbox_from_mask(m)

    def segment_with_box(self, image: np.ndarray, box):
        """box = [x, y, w, h] (COCO). RGB image."""
        x, y, w, h = box
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            box=np.asarray([x, y, x + w, y + h], dtype=np.float32),
            multimask_output=True)
        m = (masks[int(np.argmax(scores))].astype(np.uint8)) * 255
        return m, _bbox_from_mask(m)

    def segment_automatic(self, image: np.ndarray):
        """Run the automatic mask generator and return the single mask most likely
        to be the central seed: large but not frame-filling, central, low border
        touch, high predicted IoU. RGB image."""
        anns = self._auto.generate(image)
        H, W = image.shape[:2]
        total = float(H * W) or 1.0
        cy, cx = H / 2.0, W / 2.0
        diag = float(np.hypot(cy, cx)) or 1.0
        best, best_score = None, -1.0
        for a in anns:
            seg = a["segmentation"]
            frac = float(seg.sum()) / total
            if frac < 5e-4 or frac > 0.9:           # speck or whole-frame background
                continue
            bt = (int(seg[0, :].sum()) + int(seg[-1, :].sum())
                  + int(seg[:, 0].sum()) + int(seg[:, -1].sum())) / float(2 * (H + W))
            ys, xs = np.where(seg)
            centrality = 1.0 - float(np.hypot(ys.mean() - cy, xs.mean() - cx)) / diag
            score = (float(a.get("predicted_iou", 1.0))
                     * (1.0 - min(1.0, bt * 3.0))
                     * (0.5 + 0.5 * max(0.0, centrality))
                     * min(1.0, frac * 4.0))
            if score > best_score:
                best_score, best = score, seg
        if best is None:                            # nothing plausible -> largest blob
            if not anns:
                return np.zeros((H, W), np.uint8), [0, 0, 0, 0]
            best = max(anns, key=lambda a: a["area"])["segmentation"]
        m = (best.astype(np.uint8)) * 255
        return m, _bbox_from_mask(m)
