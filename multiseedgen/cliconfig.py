"""argparse builder, --config loading, Config assembly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_core import PydanticUndefined

from .errors import ConfigError
from .models import Config
from .optdeps import _HAVE_YAML, yaml


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a synthetic multi-seed detection dataset from single-seed images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("I/O")
    g.add_argument(
        "--sources",
        type=Path,
        default=None,
        help="folder of single-seed images (required unless in --config)",
    )
    g.add_argument(
        "--out", type=Path, default=None, help="output dataset folder (required unless in --config)"
    )
    g.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML/JSON config file; CLI flags override file values",
    )
    g.add_argument("--num-images", type=int, default=1000)
    g.add_argument("--img-size", type=int, default=640, help="square canvas size")
    g.add_argument("--img-ext", choices=["jpg", "png"], default="jpg")
    g.add_argument("--val-split", type=float, default=0.15)
    g.add_argument(
        "--format", dest="fmt", choices=["yolo", "coco", "both", "yolo-seg"], default="yolo"
    )
    g.add_argument(
        "--masks",
        action="store_true",
        help="compute per-seed visible polygons (auto-on for yolo-seg)",
    )
    g.add_argument(
        "--class-mode",
        choices=["single", "subfolders"],
        default="single",
        help="'subfolders' = one class per source subdirectory",
    )
    g.add_argument(
        "--visualize-count",
        type=int,
        default=0,
        help="save N annotated previews from the exported labels",
    )
    g.add_argument(
        "--check-seeds",
        type=int,
        default=0,
        help="dump N segmented cut-outs on a checkerboard to verify "
        "segmentation; use with --num-images 0 for a quick check",
    )
    g.add_argument(
        "--provenance",
        action="store_true",
        help="emit provenance_<split>.jsonl mapping boxes to source seeds",
    )
    g.add_argument(
        "--qa-report", action="store_true", help="write a dataset QA report (qa/report.md + charts)"
    )
    g.add_argument("--qa-montage", type=int, default=12, help="composites in the QA montage grid")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("scene composition")
    g.add_argument("--min-seeds", type=int, default=6)
    g.add_argument("--max-seeds", type=int, default=25)
    g.add_argument(
        "--scale-min", type=float, default=0.06, help="min seed long-side as fraction of canvas"
    )
    g.add_argument("--scale-max", type=float, default=0.18)
    g.add_argument("--overlap-mode", choices=["allow", "limit"], default="allow")
    g.add_argument(
        "--max-iou", type=float, default=0.60, help="max box IoU when --overlap-mode limit"
    )
    g.add_argument(
        "--min-visibility",
        type=float,
        default=0.40,
        help="fraction of a seed that must be on-canvas to keep it",
    )
    g.add_argument(
        "--min-visible-occluded",
        type=float,
        default=0.30,
        help="fraction left unoccluded to keep the label",
    )
    g.add_argument("--min-box-px", type=int, default=8)

    g = p.add_argument_group("per-seed augmentation")
    g.add_argument("--max-rotation", type=float, default=180.0)
    g.add_argument(
        "--aug-shear",
        type=float,
        default=0.0,
        help="max per-seed shear in degrees (0 = off); applied pre-label "
        "on the cut-out so boxes stay correct",
    )
    g.add_argument("--no-flip", dest="flip", action="store_false")
    g.add_argument("--jitter-hue", type=float, default=5.0)
    g.add_argument("--jitter-sat", type=float, default=0.15)
    g.add_argument("--jitter-val", type=float, default=0.15)
    g.add_argument("--edge-feather", type=float, default=0.5)

    g = p.add_argument_group("drop shadow")
    g.add_argument("--no-shadow", dest="shadow", action="store_false")
    g.add_argument("--shadow-opacity", type=float, default=0.25)
    g.add_argument("--shadow-blur", type=float, default=9.0)
    g.add_argument("--shadow-offset", type=int, default=4)
    g.add_argument(
        "--shadow-dir-mode",
        choices=["fixed", "scene"],
        default="scene",
        help="'scene' = one random light direction per scene (recommended for realism)",
    )

    g = p.add_argument_group("backgrounds")
    g.add_argument(
        "--backgrounds",
        nargs="+",
        default=["solid", "gradient", "noise"],
        choices=["solid", "gradient", "noise", "seedbed", "texture"],
        help="synthetic background types when no --textures given "
        "('seedbed' is discouraged: unlabelled seed clutter)",
    )
    g.add_argument(
        "--textures",
        type=Path,
        default=None,
        help="folder of REAL background photos (strongly preferred); "
        "used for ~70%% of scenes when provided",
    )
    g.add_argument(
        "--bg-from-sources",
        action="store_true",
        help="build backgrounds by inpainting the seed out of your "
        "source images (recovers your real surface, e.g. the "
        "empty tray); combine with --neg-frac for realistic "
        "background-only negatives",
    )
    g.add_argument(
        "--ghost-blend",
        type=float,
        default=1.0,
        help="strength (0..1) of the smooth low-frequency tray model "
        "used to erase the inpaint ghost in bg extraction",
    )
    g.add_argument(
        "--neg-frac",
        type=float,
        default=0.0,
        help="fraction of background-only images with no seeds "
        "(e.g. 0.1-0.15) to suppress background false positives",
    )

    g = p.add_argument_group("realism")
    g.add_argument(
        "--blend",
        choices=["alpha", "seamless"],
        default="alpha",
        help="paste blend mode; 'alpha' (default) is the recommended path",
    )
    g.add_argument(
        "--degrade",
        action="store_true",
        help="apply camera-matched degradation to the final composite",
    )
    g.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help="gaussian sensor-noise stddev on the composite (needs --degrade)",
    )
    g.add_argument(
        "--poisson-noise", action="store_true", help="add shot (Poisson) noise (needs --degrade)"
    )
    g.add_argument("--jpeg-quality-min", type=int, default=95)
    g.add_argument("--jpeg-quality-max", type=int, default=95)
    g.add_argument(
        "--aug-gamma",
        type=float,
        default=0.0,
        help="max random gamma jitter on the composite (needs --degrade); "
        "draws uniform(1-g, 1+g). Simulates lighting shift",
    )
    g.add_argument(
        "--blur-max-sigma",
        type=float,
        default=0.0,
        help="max defocus gaussian sigma (needs --degrade)",
    )
    g.add_argument(
        "--motion-blur-max",
        type=int,
        default=0,
        help="max motion-blur kernel length px (needs --degrade)",
    )
    g.add_argument(
        "--vignette",
        type=float,
        default=0.0,
        help="synthetic vignette strength on the composite (needs --degrade)",
    )
    g.add_argument(
        "--perspective",
        type=float,
        default=0.0,
        help="max corner jitter as fraction of canvas; OFF by default. "
        "Applied pre-label with boxes transformed",
    )

    g = p.add_argument_group("segmentation")
    g.add_argument(
        "--segment",
        default="auto",
        choices=["auto", "alpha", "threshold", "otsu", "grabcut", "rembg", "sam"],
    )
    g.add_argument(
        "--bg-tolerance",
        type=float,
        default=28.0,
        help="color distance from border treated as background",
    )
    g.add_argument(
        "--seg-conf-thresh",
        type=float,
        default=0.55,
        help="cut-outs below this confidence are routed to rembg / skipped",
    )
    g.add_argument(
        "--no-rembg-fallback",
        dest="rembg_fallback",
        action="store_false",
        help="disable the automatic rembg fallback for hard sources",
    )
    g.add_argument(
        "--rembg-provider",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="onnxruntime execution provider for rembg (auto = GPU if available, else CPU)",
    )
    g.add_argument(
        "--no-strip-shadow",
        dest="strip_shadow",
        action="store_false",
        help="disable dark cast-shadow removal in auto segmentation",
    )
    g.add_argument(
        "--shadow-sat-max",
        type=float,
        default=0.35,
        help="max saturation a pixel can have to count as cast shadow",
    )
    g.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="cut-out cache directory (default: <out>/.seg_cache)",
    )
    g.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="do not read or write the cut-out cache",
    )
    g.add_argument(
        "--refresh-cache", action="store_true", help="recompute and overwrite cached cut-outs"
    )
    g.add_argument(
        "--segment-map",
        type=Path,
        default=None,
        help="JSON/CSV manifest of per-source segmentation overrides",
    )
    g.add_argument(
        "--morph-kernel",
        type=int,
        default=5,
        help="ellipse kernel px for classical mask cleanup "
        "(config: segmentation.grabcut.morph_kernel)",
    )
    g.add_argument(
        "--grabcut-iters",
        type=int,
        default=5,
        help="grabcut: GrabCut iterations (segmentation.grabcut.iterations)",
    )
    g.add_argument(
        "--grabcut-rect-margin",
        type=float,
        default=0.06,
        help="grabcut: init-rect inset as fraction of min(h,w)",
    )
    g.add_argument(
        "--grabcut-min-object-area",
        type=float,
        default=0.0,
        help="grabcut: drop masks whose area is below this frame fraction",
    )
    g.add_argument(
        "--otsu-blur-kernel",
        type=int,
        default=0,
        help="otsu: pre-threshold gaussian blur kernel px (0 = off)",
    )
    g.add_argument(
        "--otsu-threshold-adjustment",
        type=int,
        default=0,
        help="otsu: signed offset added to the computed Otsu threshold",
    )
    g.add_argument(
        "--rembg-model", default=None, help="u2net: rembg model name (default = rembg's u2net)"
    )
    g.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="path to a SAM checkpoint (enables --segment sam; optional)",
    )
    g.add_argument(
        "--sam-model-type",
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b"],
        help="sam: backbone (vit_b ~358MB fits a 6GB GPU; vit_h ~2.4GB)",
    )
    g.add_argument(
        "--sam-prompt-type",
        default="automatic",
        choices=["automatic", "box", "points"],
        help="sam: prompt mode for the single-seed crop",
    )
    g.add_argument(
        "--sam-points-per-side",
        type=int,
        default=32,
        help="sam(automatic): mask-generator grid density",
    )
    g.add_argument(
        "--sam-pred-iou-thresh",
        type=float,
        default=0.88,
        help="sam(automatic): predicted-IoU quality filter",
    )
    g.add_argument(
        "--sam-stability-score-thresh",
        type=float,
        default=0.95,
        help="sam(automatic): stability-score quality filter",
    )
    g.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="segmentation worker processes; -1 = all cores, 1 = serial",
    )
    g.add_argument(
        "--gpu-seg-jobs",
        type=int,
        default=0,
        help="max concurrent GPU rembg sessions when --rembg-provider resolves to "
        "CUDA (bounds VRAM); 0 = auto (scale to free VRAM, ~0.9GB/session, "
        "fallback 4), set a positive value to cap manually",
    )
    g.add_argument("--max-sources", type=int, default=0, help="0 = use all")
    g.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="segment only a representative round-robin sample of N sources "
        "(0 = all); speeds up --check-seeds without skewing per-class coverage",
    )
    g.add_argument(
        "--max-seed-px",
        type=int,
        default=0,
        help="OPT-IN: cap each stored seed patch's long side (px) to bound the "
        "in-RAM seed pool (0 = off). Resamples oversized seeds -> changes "
        "rendered pixels (memory/quality tradeoff); keep well above "
        "ceil(scale_max*img_size) for near-lossless output",
    )

    g = p.add_argument_group("data split")
    g.add_argument(
        "--val-seed-holdout",
        type=float,
        default=0.0,
        help="fraction of SOURCE seeds reserved for val only (prevents "
        "data leakage: a held-out seed never appears in train). "
        "0 = legacy behaviour (both splits share every seed)",
    )
    g.add_argument(
        "--val-seeds",
        nargs="+",
        default=None,
        help="globs of source files reserved for val (e.g. '*_holdout*'); "
        "overrides --val-seed-holdout when given",
    )

    g = p.add_argument_group("sampling")
    g.add_argument(
        "--class-balanced",
        action="store_true",
        help="sample a class uniformly first, then a seed within it "
        "(counters source-count imbalance across classes)",
    )
    g.add_argument(
        "--class-probs",
        nargs="+",
        type=float,
        default=None,
        help="explicit per-class sampling weights, in class-id order (implies --class-balanced)",
    )

    g = p.add_argument_group("export")
    g.add_argument(
        "--extra-annotations",
        action="store_true",
        help="add per-annotation seed source/dims/scale/method to COCO",
    )
    g.add_argument(
        "--multilabel-csv",
        action="store_true",
        help="write multilabel.csv (one row/image, per-class counts) alongside the dataset",
    )

    g = p.add_argument_group("performance")
    g.add_argument(
        "--workers",
        type=int,
        default=0,
        help="scene-generation workers; 0 = auto(min(cpu,8)), "
        "1 = serial (deterministic reference path)",
    )
    g.add_argument(
        "--parallel-backend",
        choices=["auto", "loky", "threading"],
        default="auto",
        help="'loky' = separate processes (fastest, but each holds a full "
        "copy of the seed/texture assets in RAM); 'threading' = shared "
        "memory (~1x RAM, still multi-core via cv2/numpy). 'auto' uses "
        "loky only when the copies fit in free RAM, else threading.",
    )
    return p


def _config_defaults() -> dict:
    """All Config field defaults as a plain dict (paths kept as Path/None)."""
    d = {}
    for name, fld in Config.model_fields.items():
        if fld.default_factory is not None:
            d[name] = fld.default_factory()  # type: ignore[call-arg]  # our factories take no args
        elif fld.default is not PydanticUndefined:
            d[name] = fld.default
        else:
            d[name] = None
    return d


def _format_validation_error(exc: ValidationError) -> str:
    """Flatten a Pydantic ValidationError into a single human-readable config message."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "config"
        parts.append(f"{loc}: {err['msg']}")
    return "invalid config: " + "; ".join(parts)


# Nested config sections (the checklist's layout) -> flat Config field names. The
# config FILE may group settings as e.g. `segmentation: {method:, grabcut: {...}}`;
# _normalize_nested flattens that onto the flat Config below. Legacy flat keys keep
# working unchanged (none of these SECTION names are themselves Config fields, so a
# dict under one is unambiguously a nested section).
_NESTED_MAP: dict[str, Any] = {
    "segmentation": {
        "method": "segment",
        "bg_tolerance": "bg_tolerance",
        "conf_thresh": "seg_conf_thresh",
        "seg_conf_thresh": "seg_conf_thresh",
        "strip_shadow": "strip_shadow",
        "shadow_sat_max": "shadow_sat_max",
        "segment_map": "segment_map",
        "morph_kernel": "morph_kernel",
        "grabcut": {
            "iterations": "grabcut_iters",
            "iters": "grabcut_iters",
            "rect_margin": "grabcut_rect_margin",
            "morph_kernel": "morph_kernel",
            "min_object_area": "grabcut_min_object_area",
        },
        "otsu": {
            "blur_kernel": "otsu_blur_kernel",
            "threshold_adjustment": "otsu_threshold_adjustment",
        },
        "u2net": {
            "model_path": "rembg_model",
            "model": "rembg_model",
            "device": "rembg_provider",
            "provider": "rembg_provider",
            "fallback": "rembg_fallback",
        },
        "sam": {
            "checkpoint": "sam_checkpoint",
            "model_type": "sam_model_type",
            "prompt_type": "sam_prompt_type",
            "points_per_side": "sam_points_per_side",
            "pred_iou_thresh": "sam_pred_iou_thresh",
            "stability_score_thresh": "sam_stability_score_thresh",
        },
    },
    "split": {
        "val_split": "val_split",
        "val_seed_holdout": "val_seed_holdout",
        "val_seeds": "val_seeds",
    },
    "sampling": {
        "class_balanced": "class_balanced",
        "class_probs": "class_probs",
    },
    "augmentation": {
        "gamma": "aug_gamma",
        "shear": "aug_shear",
        "rotation": "max_rotation",
        "max_rotation": "max_rotation",
        "flip": "flip",
        "jitter_hue": "jitter_hue",
        "jitter_sat": "jitter_sat",
        "jitter_val": "jitter_val",
        "edge_feather": "edge_feather",
    },
    "export": {
        "format": "fmt",
        "fmt": "fmt",
        "masks": "masks",
        "extra_annotations": "extra_annotations",
        "multilabel_csv": "multilabel_csv",
    },
}


def _normalize_nested(data: dict) -> dict:
    """Flatten any nested config sections (see _NESTED_MAP) onto flat field names.
    Flat keys pass through untouched. Setting both a flat key and its nested twin to
    different values in one file is a conflict (fail fast rather than silently pick)."""
    out: dict = {}

    def assign(field: str, value: Any, where: str) -> None:
        if field in out and out[field] != value:
            raise ConfigError(f"config: {where} conflicts with an earlier value for {field!r}")
        out[field] = value

    def walk(block: dict, spec: dict, prefix: str) -> None:
        for kk, vv in block.items():
            key = str(kk).replace("-", "_")
            target = spec.get(key)
            if isinstance(target, dict):
                if not isinstance(vv, dict):
                    raise ConfigError(f"config: {prefix}.{kk} must be a mapping")
                walk(vv, target, f"{prefix}.{kk}")
            elif isinstance(target, str):
                assign(target, vv, f"{prefix}.{kk}")
            else:
                raise ConfigError(f"unknown config key: {prefix}.{kk}")

    for k, v in data.items():
        kk = str(k).replace("-", "_")
        spec = _NESTED_MAP.get(kk)
        if spec is not None and isinstance(v, dict):
            walk(v, spec, kk)
        else:
            assign(kk, v, kk)
    return out


def _load_config_file(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"--config file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        if not _HAVE_YAML:
            raise ConfigError("--config with a .yaml file needs pyyaml (pip install pyyaml)")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError("config file must be a mapping of field -> value")
    # a run_config.yaml manifest nests the settings under 'config:'
    if "config" in data and isinstance(data["config"], dict):
        data = data["config"]
    data = _normalize_nested(data)  # accept the checklist's nested layout
    valid = set(Config.model_fields)
    out: dict = {}
    for k, v in data.items():
        kk = str(k).replace("-", "_")
        if kk == "format":
            kk = "fmt"
        if kk not in valid:
            raise ConfigError(f"unknown config key: {k!r}")
        out[kk] = v
    return out


def _explicit_dests(parser: argparse.ArgumentParser, argv) -> set:
    """Which Config dests were explicitly given on the command line."""
    if argv is None:
        argv = sys.argv[1:]
    opt_to_dest = {}
    for act in parser._actions:
        for opt in act.option_strings:
            opt_to_dest[opt] = act.dest
    present = set()
    for tok in argv:
        if tok.startswith("-"):
            key = tok.split("=", 1)[0]
            if key in opt_to_dest:
                present.add(opt_to_dest[key])
    return present


def parse_args(argv=None) -> Config:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    file_vals = _load_config_file(known.config) if known.config else {}

    a = parser.parse_args(argv)
    present = _explicit_dests(parser, argv)

    merged = _config_defaults()
    merged.update(file_vals)  # file overrides defaults
    for dest in present:  # CLI overrides file
        if hasattr(a, dest):
            merged[dest] = getattr(a, dest)

    valid = set(Config.model_fields)
    merged = {k: v for k, v in merged.items() if k in valid}
    if merged.get("sources") is None or merged.get("out") is None:
        raise ConfigError("both --sources and --out are required (via CLI or --config)")
    try:
        cfg = Config(**merged)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
    return cfg
