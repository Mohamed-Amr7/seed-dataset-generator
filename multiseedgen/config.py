"""Validated configuration model - the single source of truth for a generation run.

A Pydantic v2 model carrying every setting. Deliberately **flat** (each field is a
top-level attribute) so the pipeline / segmentation / compositing code keeps reading
``cfg.<field>`` unchanged, and deliberately **mutable** (segmentation rewrites
``cfg.rembg_provider`` to a concrete device in the parent before loky workers are
forked, which keeps each worker's seg-cache key consistent with the masks produced).

The nested-section YAML layout and the argparse CLI are handled in ``cliconfig.py``,
which flattens both onto these field names before constructing the model.

Validation rules (see the determinism contract in the rework plan):
  * validators only **raise** - they never coerce a number/empty value, because
    silently turning e.g. ``0`` into ``None`` would flip the "draw only when enabled"
    predicates that gate the RNG and change byte-identical output;
  * the only two sanctioned post-validation derivations mirror the old
    ``_coerce_config``: ``class_probs`` implies ``class_balanced``, and
    ``fmt == "yolo-seg"`` implies ``masks``;
  * ``Literal`` types encode the former argparse ``choices`` (and double as the web
    form's choices), so an invalid method/format is rejected with a clear message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Config(BaseModel):
    # populate_by_name: future-proofing for aliases; extra=forbid: reject unknown keys
    # (the loader already does, this is a backstop); validate_assignment=False: the
    # parent's cfg.rembg_provider rewrite must not re-trigger validation.
    model_config = ConfigDict(populate_by_name=True, extra="forbid", validate_assignment=False)

    # ---- I/O ----
    # Required. The CLI fills these from --sources/--out or the --config file; parse_args
    # checks for their presence and raises a friendly ConfigError before constructing.
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
    max_rotation: float = 180.0  # degrees, applied as uniform(-r, +r)
    aug_shear: float = 0.0  # max per-seed shear in degrees (0 = off); label-safe
    flip: bool = True
    jitter_hue: float = 5.0  # degrees (OpenCV hue is 0..180)
    jitter_sat: float = 0.15  # +/- fraction
    jitter_val: float = 0.15  # +/- fraction
    edge_feather: float = 0.5  # gaussian sigma for the RGB blend edge only (px)

    # drop shadow
    shadow: bool = True
    shadow_opacity: float = 0.25
    shadow_blur: float = 9.0
    shadow_offset: int = 4

    # placement / overlap
    overlap_mode: Literal["allow", "limit"] = "allow"
    max_iou: float = 0.60  # only used when overlap_mode == "limit"
    min_visibility: float = 0.40  # fraction of seed that must land on-canvas
    min_visible_occluded: float = 0.30  # fraction left unoccluded to keep label
    min_box_px: int = 8  # drop boxes whose shorter side is smaller

    # backgrounds
    backgrounds: list[str] = Field(default_factory=lambda: ["solid", "gradient", "noise"])
    textures: Path | None = None
    neg_frac: float = 0.0  # fraction of background-only images (suppress false positives)
    check_seeds: int = 0  # dump N segmented cut-outs on a checkerboard to verify segmentation
    bg_from_sources: bool = False  # build backgrounds by inpainting the seed out of sources

    # segmentation
    segment: Literal["auto", "alpha", "threshold", "otsu", "grabcut", "rembg", "sam"] = "auto"
    bg_tolerance: float = 28.0  # color distance for border-thresholding
    max_sources: int = 0  # 0 = use all
    max_seed_px: int = 0  # OPT-IN: cap stored seed patch long side (0 = off, full res)
    seg_conf_thresh: float = 0.55  # below this, route a source to rembg / skip
    rembg_fallback: bool = True  # auto-route low-confidence sources to rembg
    rembg_provider: Literal["auto", "cuda", "cpu"] = "auto"  # onnxruntime EP for rembg
    strip_shadow: bool = True  # drop dark low-sat cast-shadow lobes (auto)
    shadow_sat_max: float = 0.35  # max saturation treated as "shadow" pixel
    cache_dir: Path | None = None  # default: <out>/.seg_cache
    use_cache: bool = True
    refresh_cache: bool = False
    segment_map: Path | None = None  # per-source method override manifest
    # per-method segmentation parameters (config: segmentation.<method>.<param>)
    morph_kernel: int = 5  # ellipse kernel px for mask cleanup (classical)
    grabcut_iters: int = 5  # grabcut: GrabCut iterations
    grabcut_rect_margin: float = 0.06  # grabcut: init-rect inset as frac of min(h,w)
    grabcut_min_object_area: float = 0.0  # grabcut: drop masks below this frame frac
    otsu_blur_kernel: int = 0  # otsu: pre-threshold gaussian blur kernel (0=off)
    otsu_threshold_adjustment: int = 0  # otsu: +/- offset added to the Otsu threshold
    rembg_model: str | None = None  # u2net: rembg model name (None = rembg default)
    sam_checkpoint: Path | None = None  # enables the (optional) SAM path
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_b"  # vit_b fits 6GB GPU
    sam_prompt_type: Literal["automatic", "box", "points"] = "automatic"
    sam_points_per_side: int = 32  # sam(automatic): grid density
    sam_pred_iou_thresh: float = 0.88  # sam(automatic): predicted-IoU filter
    sam_stability_score_thresh: float = 0.95  # sam(automatic): stability filter
    jobs: int = -1  # segmentation workers; -1 = all cores
    gpu_seg_jobs: int = 4  # max concurrent GPU rembg sessions (VRAM cap)

    # parallel scene generation
    workers: int = 0  # 0 = auto (min(cpu, 8)); 1 = serial reference
    parallel_backend: Literal["auto", "loky", "threading"] = "auto"

    # realism
    blend: Literal["alpha", "seamless"] = "alpha"
    shadow_dir_mode: Literal["fixed", "scene"] = "scene"  # per-scene light direction
    ghost_blend: float = 1.0  # strength of low-freq tray model in bg extraction
    degrade: bool = False  # master switch for camera-matched degradation
    noise_sigma: float = 0.0
    poisson_noise: bool = False
    jpeg_quality_min: int = 95
    jpeg_quality_max: int = 95
    aug_gamma: float = 0.0  # max random gamma jitter on the composite (0 = off)
    blur_max_sigma: float = 0.0
    motion_blur_max: int = 0
    vignette: float = 0.0
    perspective: float = 0.0  # OFF by default; applied pre-label if >0

    # data split
    class_mode: Literal["single", "subfolders"] = "single"
    val_split: float = 0.15  # fraction of generated IMAGES that go to val
    val_seed_holdout: float = 0.0  # fraction of SOURCE seeds reserved for val
    val_seeds: list[str] | None = None  # globs of source files reserved for val
    # class-balanced scene sampling (config: sampling.*)
    class_balanced: bool = False  # sample a class first (uniform), then a seed in it
    class_probs: list[float] | None = None  # per-class weights (by class id order)
    # export extras (config: export.*)
    extra_annotations: bool = False  # add seed source/dims/scale/method to COCO anns
    multilabel_csv: bool = False  # write a per-image multi-label CSV alongside COCO
    fmt: Literal["yolo", "coco", "both", "yolo-seg"] = "yolo"
    masks: bool = False  # compute/export per-seed visible polygons
    img_ext: Literal["jpg", "png"] = "jpg"
    visualize_count: int = 0
    provenance: bool = False
    qa_report: bool = False
    qa_montage: int = 12
    config: Path | None = None  # the --config path itself (round-trips into run_config)
    seed: int = 0

    @field_validator("backgrounds", "val_seeds", mode="before")
    @classmethod
    def _wrap_bare_str(cls, v: object) -> object:
        """Accept a bare string for a list field: ``"solid"`` -> ``["solid"]`` (mirrors
        the old _coerce_config; argparse nargs already yields a list)."""
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def _derive_and_validate(self) -> Config:
        # --- sanctioned derivations (identical to the former _coerce_config) ---
        if self.class_probs is not None:
            self.class_balanced = True  # explicit weights imply balanced mode
        if self.fmt == "yolo-seg":
            self.masks = True  # seg labels require per-seed polygons
        # --- range / ordering checks (raise only; never mutate values) ---
        if self.min_seeds > self.max_seeds:
            raise ValueError(
                f"min_seeds ({self.min_seeds}) must be <= max_seeds ({self.max_seeds})"
            )
        if self.scale_min > self.scale_max:
            raise ValueError(
                f"scale_min ({self.scale_min}) must be <= scale_max ({self.scale_max})"
            )
        if not (0.0 <= self.val_split <= 1.0):
            raise ValueError(f"val_split ({self.val_split}) must be in [0, 1]")
        return self
