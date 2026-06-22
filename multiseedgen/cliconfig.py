from __future__ import annotations
"""argparse builder, --config loading, Config assembly."""

from dataclasses import dataclass, field, asdict, fields as dataclass_fields, MISSING
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import sys

from .models import Config
from .optdeps import _HAVE_YAML, yaml




# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a synthetic multi-seed detection dataset from "
                    "single-seed images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("I/O")
    g.add_argument("--sources", type=Path, default=None,
                   help="folder of single-seed images (required unless in --config)")
    g.add_argument("--out", type=Path, default=None,
                   help="output dataset folder (required unless in --config)")
    g.add_argument("--config", type=Path, default=None,
                   help="YAML/JSON config file; CLI flags override file values")
    g.add_argument("--num-images", type=int, default=1000)
    g.add_argument("--img-size", type=int, default=640, help="square canvas size")
    g.add_argument("--img-ext", choices=["jpg", "png"], default="jpg")
    g.add_argument("--val-split", type=float, default=0.15)
    g.add_argument("--format", dest="fmt",
                   choices=["yolo", "coco", "both", "yolo-seg"], default="yolo")
    g.add_argument("--masks", action="store_true",
                   help="compute per-seed visible polygons (auto-on for yolo-seg)")
    g.add_argument("--class-mode", choices=["single", "subfolders"], default="single",
                   help="'subfolders' = one class per source subdirectory")
    g.add_argument("--visualize-count", type=int, default=0,
                   help="save N annotated previews from the exported labels")
    g.add_argument("--check-seeds", type=int, default=0,
                   help="dump N segmented cut-outs on a checkerboard to verify "
                        "segmentation; use with --num-images 0 for a quick check")
    g.add_argument("--provenance", action="store_true",
                   help="emit provenance_<split>.jsonl mapping boxes to source seeds")
    g.add_argument("--qa-report", action="store_true",
                   help="write a dataset QA report (qa/report.md + charts)")
    g.add_argument("--qa-montage", type=int, default=12,
                   help="composites in the QA montage grid")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("scene composition")
    g.add_argument("--min-seeds", type=int, default=6)
    g.add_argument("--max-seeds", type=int, default=25)
    g.add_argument("--scale-min", type=float, default=0.06,
                   help="min seed long-side as fraction of canvas")
    g.add_argument("--scale-max", type=float, default=0.18)
    g.add_argument("--overlap-mode", choices=["allow", "limit"], default="allow")
    g.add_argument("--max-iou", type=float, default=0.60,
                   help="max box IoU when --overlap-mode limit")
    g.add_argument("--min-visibility", type=float, default=0.40,
                   help="fraction of a seed that must be on-canvas to keep it")
    g.add_argument("--min-visible-occluded", type=float, default=0.30,
                   help="fraction left unoccluded to keep the label")
    g.add_argument("--min-box-px", type=int, default=8)

    g = p.add_argument_group("per-seed augmentation")
    g.add_argument("--max-rotation", type=float, default=180.0)
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
    g.add_argument("--shadow-dir-mode", choices=["fixed", "scene"], default="scene",
                   help="'scene' = one random light direction per scene "
                        "(recommended for realism)")

    g = p.add_argument_group("backgrounds")
    g.add_argument("--backgrounds", nargs="+",
                   default=["solid", "gradient", "noise"],
                   choices=["solid", "gradient", "noise", "seedbed", "texture"],
                   help="synthetic background types when no --textures given "
                        "('seedbed' is discouraged: unlabelled seed clutter)")
    g.add_argument("--textures", type=Path, default=None,
                   help="folder of REAL background photos (strongly preferred); "
                        "used for ~70%% of scenes when provided")
    g.add_argument("--bg-from-sources", action="store_true",
                   help="build backgrounds by inpainting the seed out of your "
                        "source images (recovers your real surface, e.g. the "
                        "empty tray); combine with --neg-frac for realistic "
                        "background-only negatives")
    g.add_argument("--ghost-blend", type=float, default=1.0,
                   help="strength (0..1) of the smooth low-frequency tray model "
                        "used to erase the inpaint ghost in bg extraction")
    g.add_argument("--neg-frac", type=float, default=0.0,
                   help="fraction of background-only images with no seeds "
                        "(e.g. 0.1-0.15) to suppress background false positives")

    g = p.add_argument_group("realism")
    g.add_argument("--blend", choices=["alpha", "seamless"], default="alpha",
                   help="paste blend mode; 'alpha' (default) is the recommended path")
    g.add_argument("--degrade", action="store_true",
                   help="apply camera-matched degradation to the final composite")
    g.add_argument("--noise-sigma", type=float, default=0.0,
                   help="gaussian sensor-noise stddev on the composite (needs --degrade)")
    g.add_argument("--poisson-noise", action="store_true",
                   help="add shot (Poisson) noise (needs --degrade)")
    g.add_argument("--jpeg-quality-min", type=int, default=95)
    g.add_argument("--jpeg-quality-max", type=int, default=95)
    g.add_argument("--blur-max-sigma", type=float, default=0.0,
                   help="max defocus gaussian sigma (needs --degrade)")
    g.add_argument("--motion-blur-max", type=int, default=0,
                   help="max motion-blur kernel length px (needs --degrade)")
    g.add_argument("--vignette", type=float, default=0.0,
                   help="synthetic vignette strength on the composite (needs --degrade)")
    g.add_argument("--perspective", type=float, default=0.0,
                   help="max corner jitter as fraction of canvas; OFF by default. "
                        "Applied pre-label with boxes transformed")

    g = p.add_argument_group("segmentation")
    g.add_argument("--segment", default="auto",
                   choices=["auto", "alpha", "threshold", "otsu", "grabcut",
                            "rembg", "sam"])
    g.add_argument("--bg-tolerance", type=float, default=28.0,
                   help="color distance from border treated as background")
    g.add_argument("--seg-conf-thresh", type=float, default=0.55,
                   help="cut-outs below this confidence are routed to rembg / skipped")
    g.add_argument("--no-rembg-fallback", dest="rembg_fallback",
                   action="store_false",
                   help="disable the automatic rembg fallback for hard sources")
    g.add_argument("--rembg-provider", default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="onnxruntime execution provider for rembg "
                        "(auto = GPU if available, else CPU)")
    g.add_argument("--no-strip-shadow", dest="strip_shadow", action="store_false",
                   help="disable dark cast-shadow removal in auto segmentation")
    g.add_argument("--shadow-sat-max", type=float, default=0.35,
                   help="max saturation a pixel can have to count as cast shadow")
    g.add_argument("--cache-dir", type=Path, default=None,
                   help="cut-out cache directory (default: <out>/.seg_cache)")
    g.add_argument("--no-cache", dest="use_cache", action="store_false",
                   help="do not read or write the cut-out cache")
    g.add_argument("--refresh-cache", action="store_true",
                   help="recompute and overwrite cached cut-outs")
    g.add_argument("--segment-map", type=Path, default=None,
                   help="JSON/CSV manifest of per-source segmentation overrides")
    g.add_argument("--sam-checkpoint", type=Path, default=None,
                   help="path to a SAM checkpoint (enables --segment sam; optional)")
    g.add_argument("--jobs", type=int, default=-1,
                   help="segmentation worker processes; -1 = all cores, 1 = serial")
    g.add_argument("--gpu-seg-jobs", type=int, default=4,
                   help="max concurrent GPU rembg sessions when --rembg-provider "
                        "resolves to CUDA (bounds VRAM; default 4 ~= 3.7GB on a 6GB card)")
    g.add_argument("--max-sources", type=int, default=0, help="0 = use all")

    g = p.add_argument_group("performance")
    g.add_argument("--workers", type=int, default=0,
                   help="scene-generation workers; 0 = auto(min(cpu,8)), "
                        "1 = serial (deterministic reference path)")
    g.add_argument("--parallel-backend", choices=["auto", "loky", "threading"],
                   default="auto",
                   help="'loky' = separate processes (fastest, but each holds a full "
                        "copy of the seed/texture assets in RAM); 'threading' = shared "
                        "memory (~1x RAM, still multi-core via cv2/numpy). 'auto' uses "
                        "loky only when the copies fit in free RAM, else threading.")
    return p



_PATH_FIELDS = {"sources", "out", "textures", "cache_dir", "segment_map",
                "sam_checkpoint", "config"}



def _config_defaults() -> dict:
    """All Config field defaults as a plain dict (paths kept as Path/None)."""
    d = {}
    for f in dataclass_fields(Config):
        if f.default is not MISSING:
            d[f.name] = f.default
        elif f.default_factory is not MISSING:        # type: ignore[misc]
            d[f.name] = f.default_factory()
        else:
            d[f.name] = None
    return d



def _load_config_file(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"--config file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        if not _HAVE_YAML:
            sys.exit("--config with a .yaml file needs pyyaml (pip install pyyaml)")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        sys.exit("config file must be a mapping of field -> value")
    # a run_config.yaml manifest nests the settings under 'config:'
    if "config" in data and isinstance(data["config"], dict):
        data = data["config"]
    valid = {f.name for f in dataclass_fields(Config)}
    out: dict = {}
    for k, v in data.items():
        kk = str(k).replace("-", "_")
        if kk == "format":
            kk = "fmt"
        if kk not in valid:
            sys.exit(f"unknown config key: {k!r}")
        out[kk] = v
    return out



def _coerce_config(d: dict) -> dict:
    out = dict(d)
    for k in _PATH_FIELDS:
        v = out.get(k)
        if isinstance(v, str):
            out[k] = Path(v)
    if isinstance(out.get("backgrounds"), str):
        out["backgrounds"] = [out["backgrounds"]]
    if out.get("fmt") == "yolo-seg":
        out["masks"] = True
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
    merged.update(file_vals)                           # file overrides defaults
    for dest in present:                               # CLI overrides file
        if hasattr(a, dest):
            merged[dest] = getattr(a, dest)
    merged = _coerce_config(merged)

    valid = {f.name for f in dataclass_fields(Config)}
    merged = {k: v for k, v in merged.items() if k in valid}
    cfg = Config(**merged)
    if cfg.sources is None or cfg.out is None:
        sys.exit("both --sources and --out are required (via CLI or --config)")
    return cfg
