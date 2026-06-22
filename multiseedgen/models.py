from __future__ import annotations
"""Dataclasses: Config, Seed, SegResult, QA stats."""

from dataclasses import dataclass, field, asdict, fields as dataclass_fields, MISSING
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np




# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    sources: Path
    out: Path
    num_images: int = 1000
    img_size: int = 640

    # how many seeds per scene
    min_seeds: int = 6
    max_seeds: int = 25

    # seed long-side as a fraction of the canvas (controls apparent size)
    scale_min: float = 0.06
    scale_max: float = 0.18

    # per-seed augmentation
    max_rotation: float = 180.0   # degrees, applied as uniform(-r, +r)
    aug_shear: float = 0.0        # max per-seed shear in degrees (0 = off); label-safe
    flip: bool = True
    jitter_hue: float = 5.0       # degrees (OpenCV hue is 0..180)
    jitter_sat: float = 0.15      # +/- fraction
    jitter_val: float = 0.15      # +/- fraction
    edge_feather: float = 0.5     # gaussian sigma for the RGB blend edge only (px)

    # drop shadow
    shadow: bool = True
    shadow_opacity: float = 0.25
    shadow_blur: float = 9.0
    shadow_offset: int = 4

    # placement / overlap
    overlap_mode: str = "allow"   # "allow" | "limit"
    max_iou: float = 0.60         # only used when overlap_mode == "limit"
    min_visibility: float = 0.40  # fraction of seed that must land on-canvas
    min_visible_occluded: float = 0.30  # fraction left unoccluded to keep label
    min_box_px: int = 8           # drop boxes whose shorter side is smaller

    # backgrounds
    backgrounds: List[str] = field(
        default_factory=lambda: ["solid", "gradient", "noise"]
    )
    textures: Optional[Path] = None
    # fraction of images that are background-only (no seeds, empty labels);
    # these suppress false positives on background texture
    neg_frac: float = 0.0
    # dump N segmented cut-outs on a checkerboard to verify segmentation
    check_seeds: int = 0
    # build backgrounds by inpainting the seed out of the source images
    # (recovers your real, domain-matched surface, e.g. the empty tray)
    bg_from_sources: bool = False

    # segmentation
    segment: str = "auto"         # auto|alpha|threshold|otsu|grabcut|rembg|sam
    bg_tolerance: float = 28.0    # color distance for border-thresholding
    max_sources: int = 0          # 0 = use all
    # OPT-IN memory cap: downscale any stored seed patch whose long side exceeds this
    # many px (0 = off, full resolution). Bounds the in-RAM seed pool when sources are
    # huge. Output is byte-identical only when no seed exceeds the cap (a no-op);
    # otherwise it resamples seeds and changes rendered pixels -- a memory/quality
    # tradeoff (a seed is re-scaled to the render size from the cap, not the original).
    # Keep it well above ceil(scale_max * img_size) (~116px with defaults, the largest
    # a seed is ever rendered) for visually near-lossless results.
    max_seed_px: int = 0
    # P0 robustness: confidence-gated learned fallback + caching + overrides
    seg_conf_thresh: float = 0.55     # below this, route a source to rembg / skip
    rembg_fallback: bool = True       # auto-route low-confidence sources to rembg
    rembg_provider: str = "auto"      # auto|cuda|cpu - onnxruntime EP for rembg
    strip_shadow: bool = True         # drop dark low-sat cast-shadow lobes (auto)
    shadow_sat_max: float = 0.35      # max saturation treated as "shadow" pixel
    cache_dir: Optional[Path] = None  # default: <out>/.seg_cache
    use_cache: bool = True
    refresh_cache: bool = False
    segment_map: Optional[Path] = None    # per-source method override manifest
    # per-method segmentation parameters (config: segmentation.<method>.<param>)
    morph_kernel: int = 5             # ellipse kernel px for mask cleanup (classical)
    grabcut_iters: int = 5            # grabcut: GrabCut iterations
    grabcut_rect_margin: float = 0.06  # grabcut: init-rect inset as frac of min(h,w)
    grabcut_min_object_area: float = 0.0  # grabcut: drop masks below this frame frac
    otsu_blur_kernel: int = 0         # otsu: pre-threshold gaussian blur kernel (0=off)
    otsu_threshold_adjustment: int = 0  # otsu: +/- offset added to the Otsu threshold
    rembg_model: Optional[str] = None  # u2net: rembg model name (None = rembg default)
    sam_checkpoint: Optional[Path] = None  # enables the (optional) SAM path
    sam_model_type: str = "vit_b"     # sam: vit_h | vit_l | vit_b (vit_b fits 6GB GPU)
    sam_prompt_type: str = "automatic"  # sam: automatic | box | points
    sam_points_per_side: int = 32     # sam(automatic): grid density
    sam_pred_iou_thresh: float = 0.88   # sam(automatic): predicted-IoU filter
    sam_stability_score_thresh: float = 0.95  # sam(automatic): stability filter
    jobs: int = -1                    # segmentation workers; -1 = all cores
    gpu_seg_jobs: int = 4             # max concurrent GPU rembg sessions (VRAM cap;
                                      # ~3.7GB peak on a 6GB card, verified)

    # parallel scene generation
    workers: int = 0              # 0 = auto (min(cpu, 8)); 1 = serial reference
    parallel_backend: str = "auto"  # auto | loky | threading; auto falls back to
    #                                  threading (shared memory) when N loky process
    #                                  copies of the assets would exceed free RAM

    # realism
    blend: str = "alpha"          # "alpha" | "seamless"
    shadow_dir_mode: str = "scene"  # "fixed" | "scene" (per-scene light direction)
    ghost_blend: float = 1.0      # strength of low-freq tray model in bg extraction
    degrade: bool = False         # master switch for camera-matched degradation
    noise_sigma: float = 0.0
    poisson_noise: bool = False
    jpeg_quality_min: int = 95
    jpeg_quality_max: int = 95
    aug_gamma: float = 0.0        # max random gamma jitter on the composite (0 = off)
    blur_max_sigma: float = 0.0
    motion_blur_max: int = 0
    vignette: float = 0.0
    perspective: float = 0.0      # OFF by default; applied pre-label if >0

    # output
    class_mode: str = "single"    # "single" | "subfolders"
    val_split: float = 0.15       # fraction of generated IMAGES that go to val
    # seed-level hold-out (prevents data leakage: a source seed used for val is
    # never used for train). 0 = legacy behaviour (both splits share all seeds).
    val_seed_holdout: float = 0.0     # fraction of SOURCE seeds reserved for val
    val_seeds: Optional[List[str]] = None  # globs of source files reserved for val
    # class-balanced scene sampling (config: sampling.*)
    class_balanced: bool = False  # sample a class first (uniform), then a seed in it
    class_probs: Optional[List[float]] = None  # per-class weights (by class id order)
    # export extras (config: export.*)
    extra_annotations: bool = False   # add seed source/dims/scale/method to COCO anns
    multilabel_csv: bool = False      # write a per-image multi-label CSV alongside COCO
    fmt: str = "yolo"             # "yolo" | "coco" | "both" | "yolo-seg"
    masks: bool = False           # compute/export per-seed visible polygons
    img_ext: str = "jpg"
    visualize_count: int = 0
    provenance: bool = False
    qa_report: bool = False
    qa_montage: int = 12
    config: Optional[Path] = None
    seed: int = 0



# --------------------------------------------------------------------------- #
# Source loading                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Seed:
    rgba: np.ndarray   # BGRA uint8
    class_id: int
    source: str = ""        # source-file rel path (provenance / extra_annotations)
    method: str = ""        # segmentation method that produced this cut-out
    split: str = "both"     # "train" | "val" | "both" (seed-level hold-out)



@dataclass
class SegResult:
    patch: Optional[np.ndarray]
    report: dict
    source: str
    class_id: int
    background: Optional[np.ndarray] = None
    cache_hit: bool = False



# --------------------------------------------------------------------------- #
# Dataset QA accumulation (charts/report written by write_qa_report; see M6)  #
# --------------------------------------------------------------------------- #
class _QAStats:
    def __init__(self):
        self.boxes_per_image: List[int] = []
        self.sizes: List[float] = []
        self.aspects: List[float] = []
        self.class_counts: Dict[int, int] = {}
        self.occlusion: List[float] = []
        self.negatives = 0

    def add(self, split: str, result) -> None:
        labels = result[4]
        self.boxes_per_image.append(len(labels))
        if not labels:
            self.negatives += 1
        for (cls, x0, y0, x1, y1) in labels:
            w, h = (x1 - x0), (y1 - y0)
            self.sizes.append(float(max(w, h)))
            self.aspects.append(float(w) / float(max(h, 1)))
            self.class_counts[int(cls)] = self.class_counts.get(int(cls), 0) + 1
