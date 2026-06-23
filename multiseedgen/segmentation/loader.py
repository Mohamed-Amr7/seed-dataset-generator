"""Source loading + segmentation orchestration: the per-source override manifest
(--segment-map), the cache-aware parallel ``_segment_one`` worker, seed-level train/val
hold-out, ``load_seeds`` (the driver), and the human-readable segmentation report."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ..constants import IMG_EXTS
from ..errors import ConfigError, SegmentationError
from ..models import Config, Seed, SegResult
from ..optdeps import _HAVE_JOBLIB, Parallel, _tqdm, delayed
from .cache import (
    _cache_load,
    _cache_load_bg,
    _cache_store,
    _cache_store_bg,
    _seg_cache_key,
)
from .segment import extract_background, segment_seed
from .session import _effective_rembg_provider, _resolve_rembg_device, _set_rembg_provider

logger = logging.getLogger(__name__)


# ---- per-source method override manifest (--segment-map) ---- #
_VALID_METHODS = {"auto", "alpha", "threshold", "otsu", "grabcut", "rembg", "sam"}


def _load_segment_map(path: Path) -> list:
    if not path.exists():
        raise ConfigError(f"--segment-map file not found: {path}")
    rules = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        items = data.get("rules", []) if isinstance(data, dict) else data
        for r in items:
            rules.append(
                {
                    "glob": r["glob"],
                    "method": r.get("method", "auto"),
                    "bg_tolerance": r.get("bg_tolerance"),
                    "strip_shadow": r.get("strip_shadow"),
                }
            )
    else:  # CSV: pattern,method[,tol]
        for row in csv.reader(path.read_text().splitlines()):
            if not row or row[0].strip().startswith("#"):
                continue
            rule = {
                "glob": row[0].strip(),
                "method": row[1].strip() if len(row) > 1 and row[1].strip() else "auto",
                "bg_tolerance": float(row[2]) if len(row) > 2 and row[2].strip() else None,
                "strip_shadow": None,
            }
            rules.append(rule)
    for r in rules:
        if r["method"] not in _VALID_METHODS:
            raise ConfigError(f"--segment-map: invalid method {r['method']!r} for {r['glob']!r}")
    return rules


def _match_override(rel: str, name: str, rules: list | None) -> dict | None:
    if not rules:
        return None
    for r in rules:
        g = r["glob"]
        if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g):
            ov = {"method": r["method"]}
            if r.get("bg_tolerance") is not None:
                ov["bg_tolerance"] = float(r["bg_tolerance"])
            if r.get("strip_shadow") is not None:
                ov["strip_shadow"] = bool(r["strip_shadow"])
            return ov
    return None


def _segment_one(
    path_str: str, class_id: int, cfg: Config, rules: list | None, cache_dir: Path
) -> SegResult:
    """Picklable per-file worker: hash -> cache -> imread -> segment -> store.
    Pure per-file (no shared mutable state, no RNG) so it is deterministic and
    order-preserving under joblib."""
    cv2.setNumThreads(1)  # avoid thread oversubscription
    _set_rembg_provider(cfg.rembg_provider)  # survive the loky fork
    path = Path(path_str)
    name = path.name
    try:
        rel = str(path.relative_to(cfg.sources))
    except Exception:
        rel = name
    override = _match_override(rel, name, rules)

    try:
        content = path.read_bytes()
    except Exception:
        return SegResult(
            None,
            {"skipped": True, "skip_reason": "read_error", "method": "-", "confidence": 0.0},
            rel,
            class_id,
        )
    chash = hashlib.blake2b(content, digest_size=16).hexdigest()
    key = _seg_cache_key(chash, cfg, override)
    bg_key = _seg_cache_key(chash, cfg, override, ":bg")

    patch = background = None
    report = None
    cache_hit = False
    if cfg.use_cache and not cfg.refresh_cache:
        cached = _cache_load(cache_dir, key)
        if cached is not None:
            patch, report = cached
            if cfg.bg_from_sources:
                status, bgarr = _cache_load_bg(cache_dir, bg_key)
                if status == "hit":
                    background, cache_hit = bgarr, True
            else:
                cache_hit = True

    if not cache_hit:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            patch, report = (
                None,
                {"skipped": True, "skip_reason": "read_error", "method": "-", "confidence": 0.0},
            )
        else:
            patch, report = segment_seed(img, cfg, override)
            if cfg.bg_from_sources:
                background = extract_background(img, cfg, override)
        if cfg.use_cache:
            _cache_store(cache_dir, key, patch, report)
            if cfg.bg_from_sources:
                _cache_store_bg(cache_dir, bg_key, background)

    report = dict(report or {})
    report["source"] = rel
    if override:
        report["override"] = override.get("method")
    return SegResult(patch, report, rel, class_id, background, cache_hit)


def _assign_seed_splits(seeds: list[Seed], cfg: Config) -> None:
    """Tag each Seed.split for seed-level train/val hold-out (prevents data leakage:
    a source seed used in val is never used in train). Default (no val_seeds globs and
    val_seed_holdout<=0): every seed stays 'both' -> legacy behaviour where both splits
    share all seeds. With a hold-out the partition is per-class stratified, deterministic
    in cfg.seed, and never empties a class from the train pool."""
    val_globs = cfg.val_seeds or None
    frac = float(cfg.val_seed_holdout or 0.0)
    if not val_globs and frac <= 0.0:
        return  # legacy: leave every seed "both"
    if val_globs:
        for s in seeds:
            name = Path(s.source).name
            is_val = any(
                fnmatch.fnmatch(s.source, g) or fnmatch.fnmatch(name, g) for g in val_globs
            )
            s.split = "val" if is_val else "train"
        return
    by_class: dict[int, list[int]] = {}
    for i, s in enumerate(seeds):
        by_class.setdefault(s.class_id, []).append(i)
    rng = np.random.default_rng(int(cfg.seed))
    for _cid, idxs in by_class.items():
        order = list(idxs)
        rng.shuffle(order)
        n_val = min(int(round(frac * len(order))), max(0, len(order) - 1))
        val_set = set(order[:n_val])  # keep >=1 per class in train
        for i in order:
            seeds[i].split = "val" if i in val_set else "train"


def load_seeds(cfg: Config):
    """Read and segment every source image (in parallel, with a disk cache).

    Returns (seeds, class_names, source_backgrounds, seg_results). seg_results is
    the per-source list (kept + skipped) used by write_seg_report; source_bgs is
    populated only when cfg.bg_from_sources is set."""
    root = cfg.sources
    if not root.exists():
        raise ConfigError(f"sources directory not found: {root}")
    # Resolve the rembg device ONCE in the parent against what ORT actually grants,
    # and rewrite to a concrete verified device so every worker's cache key + the job
    # cap reflect the masks really produced (not the build-time provider list).
    if cfg.segment in ("rembg", "sam") or cfg.rembg_fallback or cfg.segment_map is not None:
        cfg.rembg_provider = _resolve_rembg_device(cfg.rembg_provider)
    _set_rembg_provider(cfg.rembg_provider)  # parent process (serial path)

    if cfg.class_mode == "subfolders":
        subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not subdirs:
            raise ConfigError("--class-mode subfolders given but no subdirectories found")
        class_names = [p.name for p in subdirs]
        groups = [(i, sorted(p.rglob("*"))) for i, p in enumerate(subdirs)]
    else:
        class_names = ["seed"]
        groups = [(0, sorted(root.rglob("*")))]

    rules = _load_segment_map(cfg.segment_map) if cfg.segment_map else None
    cache_dir = cfg.cache_dir if cfg.cache_dir is not None else (cfg.out / ".seg_cache")
    tasks = [(str(f), cid) for cid, files in groups for f in files if f.suffix.lower() in IMG_EXTS]

    # Bound concurrency when rembg runs on the GPU: many worker processes would each
    # open a CUDA session and exhaust VRAM. Only caps when GPU rembg is the method
    # for many sources (forced --segment rembg/sam, or a --segment-map); classical
    # parallelism is untouched in the default auto path.
    jobs = cfg.jobs
    gpu_rembg = _effective_rembg_provider(cfg.rembg_provider) == "cuda" and (
        cfg.segment in ("rembg", "sam") or cfg.rembg_fallback or cfg.segment_map is not None
    )
    # SAM holds a full model per worker process (VRAM on GPU, large RAM on CPU), so it
    # is capped like GPU rembg regardless of the onnxruntime provider.
    heavy_sam = cfg.segment == "sam" and cfg.sam_checkpoint is not None
    if gpu_rembg or heavy_sam:
        cap = max(1, int(cfg.gpu_seg_jobs))
        if jobs <= 0 or jobs > cap:
            jobs = cap
        what = "SAM" if heavy_sam else "GPU rembg"
        logger.info(
            f"  [{'sam' if heavy_sam else 'gpu'}] segmentation capped to {jobs} "
            f"worker(s) to fit memory (--gpu-seg-jobs); {what} stays faster per image"
        )

    if _HAVE_JOBLIB and jobs != 1 and len(tasks) > 1:
        results: list[SegResult] = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_segment_one)(p, cid, cfg, rules, cache_dir) for p, cid in tasks
        )
    else:
        results = [
            _segment_one(p, cid, cfg, rules, cache_dir)
            for p, cid in _tqdm(tasks, desc="segment", unit="img")
        ]

    seeds: list[Seed] = []
    source_bgs: list[np.ndarray] = []
    for r in results:
        if r.patch is not None and r.patch.shape[0] >= 4 and r.patch.shape[1] >= 4:
            seeds.append(
                Seed(
                    rgba=r.patch,
                    class_id=r.class_id,
                    source=r.source,
                    method=r.report.get("method", ""),
                )
            )
            if cfg.bg_from_sources and r.background is not None:
                source_bgs.append(r.background)

    # OPT-IN downscale cap: shrink oversized patches BEFORE staging/pooling so the
    # in-RAM pool (and the loky workers' shared copy) is bounded. Applied after the
    # cache load on purpose, so it never enters the segmentation cache key. Off (0) by
    # default -> output byte-identical; any value changes pixels, hence opt-in.
    if cfg.max_seed_px and cfg.max_seed_px > 0:
        from ..compositing import _resize_to_long_side

        for s in seeds:
            h, w = s.rgba.shape[:2]
            if max(h, w) > cfg.max_seed_px:
                s.rgba = _resize_to_long_side(s.rgba, cfg.max_seed_px)

    if cfg.max_sources and len(seeds) > cfg.max_sources:
        idx = np.random.default_rng(cfg.seed).choice(len(seeds), cfg.max_sources, replace=False)
        seeds = [seeds[i] for i in idx]

    _assign_seed_splits(seeds, cfg)  # seed-level train/val hold-out
    n_val_seeds = sum(1 for s in seeds if s.split == "val")
    if n_val_seeds:
        logger.info(
            f"  seed hold-out: {len(seeds) - n_val_seeds} train-only / "
            f"{n_val_seeds} val-only source seed(s) (no overlap)"
        )

    n_skip = sum(1 for r in results if r.report.get("skipped"))
    n_hit = sum(1 for r in results if r.cache_hit)
    logger.info(
        f"  loaded {len(seeds)} seed patches ({n_skip} skipped, "
        f"{n_hit} from cache) across {len(class_names)} class(es): {class_names}"
    )
    if cfg.bg_from_sources:
        logger.info(f"  extracted {len(source_bgs)} domain background(s) from sources")
    if not seeds:
        raise SegmentationError(
            "no usable seeds were segmented - check --segment / "
            "--seg-conf-thresh / --bg-tolerance, and read seg_report.txt"
        )

    # Release the heavy pixel buffers we no longer need. Kept patches are already
    # referenced by Seed.rgba (the SAME ndarray, no copy) and any extracted
    # backgrounds by source_bgs, so this frees only the dropped/sub-sampled patches
    # and every background once collected. write_seg_report reads r.report /
    # r.cache_hit only, so the returned results stay valid for the report.
    for r in results:
        r.patch = None
        r.background = None
    return seeds, class_names, source_bgs, results


def write_seg_report(out: Path, results: list[SegResult], cfg: Config) -> dict:
    """Write seg_report.{json,txt} and print a summary: per-method kept counts,
    skips grouped by reason, a confidence histogram, and cache stats."""
    out.mkdir(parents=True, exist_ok=True)
    method_counts: dict[str, int] = {}
    skip_counts: dict[str, int] = {}
    skipped_list = []
    confs = []
    for r in results:
        rep = r.report
        if rep.get("skipped"):
            reason = rep.get("skip_reason", "?")
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            skipped_list.append(
                {
                    "source": rep.get("source", "?"),
                    "reason": reason,
                    "confidence": round(float(rep.get("confidence", 0.0)), 3),
                }
            )
        else:
            m = rep.get("method", "?")
            method_counts[m] = method_counts.get(m, 0) + 1
            confs.append(float(rep.get("confidence", 0.0)))

    bins = [0] * 10
    for c in confs:
        bins[min(9, int(c * 10))] += 1
    n_hit = sum(1 for r in results if r.cache_hit)
    summary = {
        "scanned": len(results),
        "kept": len(confs),
        "skipped": len(skipped_list),
        "cache_hits": n_hit,
        "method_counts": method_counts,
        "skip_counts": skip_counts,
        "confidence": {
            "min": round(min(confs), 3) if confs else None,
            "mean": round(float(np.mean(confs)), 3) if confs else None,
            "median": round(float(np.median(confs)), 3) if confs else None,
            "histogram_0_1_by_0.1": bins,
        },
    }
    (out / "seg_report.json").write_text(
        json.dumps({"summary": summary, "skipped": skipped_list}, indent=2)
    )

    lines = [
        "seedgen segmentation report",
        f"  scanned {summary['scanned']}  kept {summary['kept']}  "
        f"skipped {summary['skipped']}  cache_hits {n_hit}",
        f"  by method: {method_counts}",
        f"  skipped by reason: {skip_counts}",
        f"  confidence min/median/mean: {summary['confidence']['min']}/"
        f"{summary['confidence']['median']}/{summary['confidence']['mean']}",
        f"  confidence histogram [0..1 by 0.1]: {bins}",
    ]
    if skipped_list:
        lines.append("  skipped sources (first 40):")
        for s in skipped_list[:40]:
            lines.append(f"    {s['reason']:18s} conf={s['confidence']:.3f}  {s['source']}")
    text = "\n".join(lines)
    (out / "seg_report.txt").write_text(text + "\n")
    logger.info(text)
    return summary
