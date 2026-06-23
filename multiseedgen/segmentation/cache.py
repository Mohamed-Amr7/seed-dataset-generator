"""Cut-out cache: content-hash + SEG_VERSION + a per-method parameter signature form
the on-disk key (so tuning one method never invalidates another's cached cut-outs), with
load/store for both segmented patches and extracted backgrounds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..constants import SEG_VERSION
from ..models import Config
from .session import _effective_rembg_provider


# ---- cut-out cache (keyed by file content hash + seg params + SEG_VERSION) ---- #
def _seg_params_signature(cfg: Config, override: dict | None) -> str:
    ov = override or {}
    method = ov.get("method", cfg.segment)
    sig = {
        "segment": method,
        "bg_tolerance": round(float(ov.get("bg_tolerance", cfg.bg_tolerance)), 3),
        "seg_conf_thresh": round(float(cfg.seg_conf_thresh), 3),
        "strip_shadow": bool(ov.get("strip_shadow", cfg.strip_shadow)),
        "shadow_sat_max": round(float(cfg.shadow_sat_max), 3),
        "rembg_fallback": bool(cfg.rembg_fallback),
        "ghost_blend": round(float(cfg.ghost_blend), 3),
    }
    # GPU vs CPU rembg masks differ by a few pixels; only namespace the cache by
    # device when rembg can actually be used (keeps pure-classical keys device-free).
    if cfg.rembg_fallback or method in ("rembg", "sam"):
        sig["rembg_device"] = _effective_rembg_provider(cfg.rembg_provider)
    # per-method params only enter the key for the method that actually uses them, so
    # tuning one method never invalidates another's cached cut-outs.
    if method == "grabcut":
        sig["grabcut_iters"] = int(cfg.grabcut_iters)
        sig["grabcut_rect_margin"] = round(float(cfg.grabcut_rect_margin), 4)
        sig["grabcut_min_object_area"] = round(float(cfg.grabcut_min_object_area), 4)
        sig["morph_kernel"] = int(cfg.morph_kernel)
    elif method == "otsu":
        sig["otsu_blur_kernel"] = int(cfg.otsu_blur_kernel)
        sig["otsu_threshold_adjustment"] = int(cfg.otsu_threshold_adjustment)
        sig["morph_kernel"] = int(cfg.morph_kernel)
    elif method == "threshold":
        sig["morph_kernel"] = int(cfg.morph_kernel)
    elif method == "sam":
        sig["sam"] = {
            "model_type": cfg.sam_model_type,
            "prompt_type": cfg.sam_prompt_type,
            "points_per_side": int(cfg.sam_points_per_side),
            "pred_iou_thresh": round(float(cfg.sam_pred_iou_thresh), 4),
            "stability_score_thresh": round(float(cfg.sam_stability_score_thresh), 4),
            "checkpoint": str(cfg.sam_checkpoint) if cfg.sam_checkpoint else None,
        }
    return json.dumps(sig, sort_keys=True)


def _seg_cache_key(content_hash: str, cfg: Config, override: dict | None, suffix: str = "") -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{SEG_VERSION}|{content_hash}|".encode())
    h.update(_seg_params_signature(cfg, override).encode())
    h.update(f"|{suffix}".encode())
    return h.hexdigest()


def _cache_load(cache_dir: Path, key: str):
    js = cache_dir / f"{key}.json"
    if not js.exists():
        return None
    try:
        meta = json.loads(js.read_text())
    except Exception:
        return None
    if meta.get("seg_version") != SEG_VERSION:
        return None
    patch = None
    if not meta.get("skipped"):
        png = cache_dir / f"{key}.png"
        if not png.exists():
            return None
        patch = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if patch is None:
            return None
    return patch, meta.get("report", {})


def _cache_store(cache_dir: Path, key: str, patch: np.ndarray | None, report: dict) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if patch is not None:
            cv2.imwrite(str(cache_dir / f"{key}.png"), patch)
        (cache_dir / f"{key}.json").write_text(
            json.dumps({"seg_version": SEG_VERSION, "skipped": patch is None, "report": report})
        )
    except Exception:
        pass


def _cache_load_bg(cache_dir: Path, key: str):
    """Return ('hit', array_or_None) or ('miss', None)."""
    js = cache_dir / f"{key}.bg.json"
    if not js.exists():
        return ("miss", None)
    try:
        meta = json.loads(js.read_text())
    except Exception:
        return ("miss", None)
    if not meta.get("present"):
        return ("hit", None)
    png = cache_dir / f"{key}.bg.png"
    arr = cv2.imread(str(png)) if png.exists() else None
    return ("hit", arr) if arr is not None else ("miss", None)


def _cache_store_bg(cache_dir: Path, key: str, bg: np.ndarray | None) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if bg is not None:
            cv2.imwrite(str(cache_dir / f"{key}.bg.png"), bg)
        (cache_dir / f"{key}.bg.json").write_text(json.dumps({"present": bg is not None}))
    except Exception:
        pass
