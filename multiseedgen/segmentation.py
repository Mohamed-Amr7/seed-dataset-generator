from __future__ import annotations
"""Confidence-scored segmentation cascade, rembg fallback, cache, loader."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import cv2
import fnmatch
import hashlib
import json
import numpy as np
import sys

from .constants import IMG_EXTS, SEG_VERSION
from .models import Config, Seed, SegResult
from .optdeps import Parallel, _HAVE_JOBLIB, _tqdm, delayed




# --------------------------------------------------------------------------- #
# Segmentation                                                                #
# --------------------------------------------------------------------------- #
def _largest_component_fill(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Clean a binary mask: morphology, keep the biggest blob, fill holes.
    ``ksize`` is the ellipse structuring-element size (default 5 = legacy behaviour;
    configurable per classical method via cfg.morph_kernel)."""
    ksize = max(1, int(ksize))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [biggest], -1, 255, thickness=cv2.FILLED)
    return out



def _mask_border_color(bgr: np.ndarray, tol: float) -> np.ndarray:
    """Foreground = pixels far (in color) from the median border color."""
    h, w = bgr.shape[:2]
    border = np.concatenate(
        [bgr[0, :, :], bgr[-1, :, :], bgr[:, 0, :], bgr[:, -1, :]], axis=0
    ).reshape(-1, 3).astype(np.float32)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    return ((dist > tol).astype(np.uint8)) * 255



def _mask_color_dist(bgr: np.ndarray) -> np.ndarray:
    """Foreground = pixels whose COLOUR is far from the global-median colour.
    The background is the majority of a single-seed crop, so its median colour
    is a reliable reference. Uses full colour (brightness + hue), so it isolates
    a PALE seed on a light background where brightness alone gives no contrast."""
    bg = np.median(bgr.reshape(-1, 3), axis=0).astype(np.float32)
    dist = np.linalg.norm(bgr.astype(np.float32) - bg[None, None, :], axis=2)
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (m > 0).mean() > 0.5:          # a single seed should be the minority
        m = 255 - m
    return m



def _mask_flatten_brightness(bgr: np.ndarray) -> np.ndarray:
    """Illumination-flattened brightness split (good for a dark seed on an
    unevenly-lit / vignetted light background)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sigma = max(8.0, max(h, w) / 6.0)
    bg = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    norm = np.clip(gray.astype(np.float32) / (bg + 1e-6), 0.0, 2.0)
    flat = (norm / 2.0 * 255.0).astype(np.uint8)
    med = float(np.median(flat))
    _, otsu = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = otsu == 255
    bm = float(flat[bright].mean()) if bright.any() else 255.0
    dm = float(flat[~bright].mean()) if (~bright).any() else 0.0
    m = (bright if abs(bm - med) >= abs(dm - med) else ~bright).astype(np.uint8) * 255
    if (m > 0).mean() > 0.5:
        m = 255 - m
    return m



def _mask_quality(bgr: np.ndarray, raw: np.ndarray, clean: np.ndarray) -> dict:
    """Score how plausibly ``clean`` is a single seed (vs background/shadow/text).

    Returns a dict with a 0..1 ``confidence`` and the component signals. The
    foreground-fraction band is a MULTIPLIER, so an implausible footprint (or one
    that trips the >70%-of-frame safeguard) zeroes the score regardless of the
    other signals - that is the dataset-poisoning guard."""
    H, W = clean.shape[:2]
    total = float(H * W)
    area = float((clean > 0).sum())
    frac = area / total if total else 0.0

    # 1. foreground-fraction trapezoid (0 outside a single-seed footprint). The
    #    plateau runs out to 0.60 because single-seed SOURCE photos legitimately
    #    have the seed filling much of the frame; the >70% safeguard still guards
    #    the "grabbed the whole background" failure.
    if area == 0 or frac > 0.7:
        s_frac = 0.0
    elif frac < 5e-4:
        s_frac = 0.0
    elif frac < 2e-3:
        s_frac = (frac - 5e-4) / (2e-3 - 5e-4)
    elif frac <= 0.60:
        s_frac = 1.0
    else:
        s_frac = max(0.0, (0.72 - frac) / (0.72 - 0.60))

    # 2. solidity of the largest contour (shadow lobes / text are non-convex)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sol = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        ha = cv2.contourArea(cv2.convexHull(c))
        sol = a / ha if ha > 0 else 0.0
    s_sol = float(np.clip((sol - 0.6) / 0.35, 0, 1))

    # 3. largest-component ratio on the RAW mask (watermark text / speckle splits it)
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(
        (raw > 0).astype(np.uint8), 8)
    if n_lbl > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        s_comp = float(areas.max() / areas.sum()) if areas.sum() > 0 else 0.0
    else:
        s_comp = 0.0

    # 4. boundary edge strength on the mask outline. A GOOD mask sits on the real
    #    seed/background edge (strong gradient); a halo's outline lies in flat tray
    #    and a near-invisible seed has no edge at all. We combine a RELATIVE measure
    #    (boundary gradient vs the image's overall gradient -> catches halos placed
    #    off the true edge) with an ABSOLUTE one (boundary gradient magnitude ->
    #    catches the no-contrast/faint case where nothing is a real edge), taking the
    #    MIN so either failure mode vetoes. This term MULTIPLIES the score.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring = (cv2.dilate(clean, k) > 0) & (cv2.erode(clean, k) == 0)
    ring_grad = float(mag[ring].mean()) if ring.any() else 0.0
    s_edge_rel = float(np.clip(ring_grad / (2 * mag.mean() + 1e-6), 0, 1))
    s_edge_abs = float(np.clip(ring_grad / 90.0, 0, 1))
    s_edge = min(s_edge_rel, s_edge_abs)

    # 5. border-touch (a halo that reaches the frame edge is suspect)
    border = np.concatenate([clean[0, :], clean[-1, :], clean[:, 0], clean[:, -1]])
    bt = float((border > 0).mean())
    s_border = 1.0 - float(np.clip(bt / 0.15, 0, 1))

    # 6. fg/bg colour separation (absolute), reported and used as a soft floor
    fg_mean = bgr[clean > 0].astype(np.float32).mean(0) if (clean > 0).any() else None
    bg_mean = bgr[clean == 0].astype(np.float32).mean(0) if (clean == 0).any() else None
    colordiff = float(np.linalg.norm(fg_mean - bg_mean)) \
        if fg_mean is not None and bg_mean is not None else 0.0
    s_contrast = float(np.clip(colordiff / 25.0, 0, 1))

    # Edge strength gates the score (multiplier): a confident cut-out needs a real,
    # well-placed boundary. Shape terms then grade the blob quality.
    shape = 0.55 * s_sol + 0.28 * s_comp + 0.17 * s_border
    conf = s_frac * s_edge * max(s_contrast, 0.4) * shape
    return {
        "confidence": float(conf), "fg_fraction": float(frac),
        "solidity": float(sol), "n_components": int(n_lbl - 1),
        "border_touch": float(bt), "edge_contrast": float(s_edge),
        "ring_grad": round(ring_grad, 1), "colordiff": round(colordiff, 1),
        "hit_safeguard": bool(frac > 0.7),
    }



def _strip_cast_shadow(bgr: np.ndarray, mask: np.ndarray,
                       sat_max: float = 0.35) -> np.ndarray:
    """Drop cast-shadow pixels that got attached to the seed blob, then keep the
    largest remaining component. A shadow is, relative to the seed: low-saturation,
    darker, and far in colour from the seed's own (often saturated) body. Using the
    seed's reference colour catches the lighter feathered shadow fringe that a plain
    value threshold misses. Aborts (returns the original mask) if the strip removes
    >45% of the area - that means it was eating the seed, not a shadow."""
    inside = mask > 0
    if int(inside.sum()) == 0:
        return mask
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2]
    # reference seed colour: median of the more-saturated core inside the mask
    s_in = S[inside]
    core = inside
    if s_in.size >= 10:
        thr = float(np.quantile(s_in, 0.6))
        cand = inside & (S >= thr)
        if int(cand.sum()) >= 10:
            core = cand
    seed_col = np.median(bgr[core].astype(np.float32), axis=0)
    seed_v = float(np.median(V[core]))
    dist = np.linalg.norm(bgr.astype(np.float32) - seed_col[None, None, :], axis=2)
    drop = inside & (S < sat_max) & (dist > 40.0) & (V < seed_v)
    if not drop.any():
        return mask
    new = mask.copy()
    new[drop] = 0
    new = _largest_component_fill(new)
    if int((new > 0).sum()) < 0.55 * int(inside.sum()):
        return mask
    return new



_AUTO_METHODS = (
    ("color_dist", _mask_color_dist),
    ("flatten", _mask_flatten_brightness),
)



def _auto_mask_scored(bgr: np.ndarray, cfg: Config):
    """Colour-first cascade that scores each candidate. Preserves the original
    ordering by short-circuiting on the first in-band, confident colour-distance
    mask; otherwise scores colour-distance, flattened-brightness and border-colour
    and returns the highest-confidence candidate. Returns (clean_mask, report)."""
    best = None
    methods = list(_AUTO_METHODS) + [
        ("border", lambda b: _mask_border_color(b, cfg.bg_tolerance))]
    for name, fn in methods:
        raw = fn(bgr)
        clean = _largest_component_fill(raw)
        q = _mask_quality(bgr, raw, clean)
        q["method"] = name
        if (5e-4 <= q["fg_fraction"] <= 0.6
                and q["confidence"] >= cfg.seg_conf_thresh):
            return clean, q
        if best is None or q["confidence"] > best[1]["confidence"]:
            best = (clean, q)
    return best



def _auto_mask(bgr: np.ndarray, tol: float) -> Optional[np.ndarray]:
    """Back-compat shim: return the highest-confidence auto mask, or None when it
    trips the oversize safeguard (kept so extract_background's legacy path works)."""
    class _Shim:
        bg_tolerance = tol
        seg_conf_thresh = 1.1   # never short-circuit; always pick best-of-three
    clean, q = _auto_mask_scored(bgr, _Shim())
    if clean is None or q.get("fg_fraction", 1.0) > 0.7:
        return None
    return clean



def _mask_otsu(bgr: np.ndarray, blur_kernel: int = 0,
               threshold_adjustment: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur_kernel and int(blur_kernel) > 0:
        kk = int(blur_kernel) | 1                   # gaussian needs an odd kernel
        gray = cv2.GaussianBlur(gray, (kk, kk), 0)
    thr, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if threshold_adjustment:                        # re-threshold at Otsu +/- offset
        t2 = float(np.clip(thr + threshold_adjustment, 0, 255))
        _, m = cv2.threshold(gray, t2, 255, cv2.THRESH_BINARY)
    # assume the seed is the darker-or-lighter minority; flip if bg got selected
    if m.mean() > 127:
        m = 255 - m
    return m



def _mask_grabcut(bgr: np.ndarray, iters: int = 5,
                  rect_margin: float = 0.06) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    inset = max(2, int(float(rect_margin) * min(h, w)))
    rect = (inset, inset, max(1, w - 2 * inset), max(1, h - 2 * inset))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, max(1, int(iters)),
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)



_REMBG_SESSION = None
_REMBG_PROVIDER = "auto"        # auto|cuda|cpu - set per process from cfg
_ORT_PROVIDERS_CACHE = None     # memoized onnxruntime.get_available_providers()


def _set_rembg_provider(pref: str) -> None:
    """Set the desired execution provider for rembg in THIS process. Called at the
    top of the parent loader and every segmentation worker, so the choice survives
    the loky fork. Changing it resets any built session."""
    global _REMBG_PROVIDER, _REMBG_SESSION
    pref = (pref or "auto").lower()
    if pref != _REMBG_PROVIDER:
        _REMBG_PROVIDER, _REMBG_SESSION = pref, None


def _ort_available() -> list:
    global _ORT_PROVIDERS_CACHE
    if _ORT_PROVIDERS_CACHE is None:
        try:
            import onnxruntime as ort
            _ORT_PROVIDERS_CACHE = list(ort.get_available_providers())
        except Exception:
            _ORT_PROVIDERS_CACHE = []
    return _ORT_PROVIDERS_CACHE


_RESOLVED_DEVICE: dict = {}     # pref -> "cuda"/"cpu" (REAL probe result, per process)


def _resolve_rembg_device(pref: str) -> str:
    """Probe the device rembg can ACTUALLY use (not the build-time provider list):
    build a real rembg session and read the providers ORT *granted*. Memoized.
    If the GPU build is present but CUDA can't initialise (no driver / missing or
    incompatible runtime libs), this correctly returns 'cpu' so the cache key and
    the worker cap match the masks actually produced. 'cpu' short-circuits (no load)."""
    pref = (pref or "auto").lower()
    if pref in _RESOLVED_DEVICE:
        return _RESOLVED_DEVICE[pref]
    if pref == "cpu" or "CUDAExecutionProvider" not in _ort_available():
        _RESOLVED_DEVICE[pref] = "cpu"
        return "cpu"
    dev = "cpu"
    global _REMBG_SESSION
    try:
        from rembg import new_session
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        s = new_session(providers=[("CUDAExecutionProvider",
                                    {"arena_extend_strategy": "kSameAsRequested"}),
                                   "CPUExecutionProvider"])
        granted = s.inner_session.get_providers()
        dev = "cuda" if "CUDAExecutionProvider" in granted else "cpu"
        _REMBG_SESSION = s          # reuse the probe session; don't waste the load
        if pref == "cuda" and dev == "cpu":
            print("  [rembg][warn] CUDA requested but ORT granted CPU; check "
                  "onnxruntime-gpu / CUDA runtime libs", file=sys.stderr)
    except Exception:
        dev = "cpu"
    _RESOLVED_DEVICE[pref] = dev
    return dev


def _effective_rembg_provider(pref: str) -> str:
    """Device the cache key + worker cap should assume. Concrete 'cpu'/'cuda' are
    taken as already-resolved -- the parent loader rewrites cfg.rembg_provider to a
    PROBED concrete device before dispatch, so workers never re-probe. 'auto' probes."""
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if "CUDAExecutionProvider" in _ort_available() else "cpu"
    return _resolve_rembg_device("auto")


def _rembg_ort_providers(pref: str) -> list:
    """ORT provider list (priority order) for a desired preference. On CUDA we cap
    arena growth (kSameAsRequested) so several concurrent sessions share the 6GB
    card without one greedily grabbing it all."""
    if _effective_rembg_provider(pref) == "cuda":
        return [("CUDAExecutionProvider",
                 {"arena_extend_strategy": "kSameAsRequested"}),
                "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _mask_rembg(bgr: np.ndarray) -> Optional[np.ndarray]:
    global _REMBG_SESSION
    try:
        from rembg import remove, new_session
    except ImportError:
        print("  [warn] rembg not installed; falling back to threshold", file=sys.stderr)
        return None
    if _REMBG_SESSION is None:
        want_cuda = _effective_rembg_provider(_REMBG_PROVIDER) == "cuda"
        providers = _rembg_ort_providers(_REMBG_PROVIDER)
        if want_cuda:
            try:                                # make pip-installed CUDA libs loadable
                import onnxruntime as ort
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
            except Exception:
                pass
        try:
            _REMBG_SESSION = new_session(providers=providers)
        except TypeError:                       # very old rembg without providers kw
            _REMBG_SESSION = new_session()
        try:                                    # report what ORT actually granted
            actual = _REMBG_SESSION.inner_session.get_providers()
        except Exception:
            actual = providers
        print(f"  [rembg] execution providers: {actual}", file=sys.stderr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = remove(rgb, session=_REMBG_SESSION)  # RGBA
    return out[..., 3]



def _crop_to_alpha(bgr: np.ndarray, alpha: np.ndarray, pad: int = 2) -> Optional[np.ndarray]:
    """Tight-crop BGR+alpha to the alpha extent and return a BGRA patch."""
    ys, xs = np.where(alpha > 8)
    if ys.size == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + 1 + pad)
    bgr_c = bgr[y0:y1, x0:x1]
    a_c = alpha[y0:y1, x0:x1]
    return np.dstack([bgr_c, a_c]).astype(np.uint8)



_SAM_SEGMENTER = None       # process-global SAMSegmenter (survives the loky fork)


def _mask_sam(bgr: np.ndarray, cfg: "Config") -> Optional[np.ndarray]:
    """Optional SAM path. Import-guarded: returns None (callers then fall back) unless
    a checkpoint is configured AND torch + segment_anything are installed. Returns a
    0/255 mask for the seed, matching the rembg alpha contract. The segmenter is built
    once per process and reused. Prompt mode is cfg.sam_prompt_type."""
    global _SAM_SEGMENTER
    if getattr(cfg, "sam_checkpoint", None) is None:
        return None
    if _SAM_SEGMENTER is None:
        try:
            from .sam_segment import SAMSegmenter, sam_available
        except Exception:
            return None
        if not sam_available():
            print("  [warn] --segment sam needs torch + segment-anything "
                  "(pip install segment-anything torch); skipping", file=sys.stderr)
            return None
        try:
            _SAM_SEGMENTER = SAMSegmenter(
                checkpoint=cfg.sam_checkpoint, model_type=cfg.sam_model_type,
                points_per_side=cfg.sam_points_per_side,
                pred_iou_thresh=cfg.sam_pred_iou_thresh,
                stability_score_thresh=cfg.sam_stability_score_thresh)
            print(f"  [sam] loaded {cfg.sam_model_type} on {_SAM_SEGMENTER.device}",
                  file=sys.stderr)
        except Exception as e:                      # bad checkpoint / OOM / etc.
            print(f"  [warn] SAM load failed ({e}); skipping", file=sys.stderr)
            return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    pt = (cfg.sam_prompt_type or "automatic").lower()
    try:
        if pt == "box":
            inset = max(2, int(float(cfg.grabcut_rect_margin) * min(h, w)))
            mask, _ = _SAM_SEGMENTER.segment_with_box(
                rgb, [inset, inset, max(1, w - 2 * inset), max(1, h - 2 * inset)])
        elif pt == "points":
            mask, _ = _SAM_SEGMENTER.segment_with_points(rgb, [(w // 2, h // 2)], [1])
        else:                                       # automatic (default)
            mask, _ = _SAM_SEGMENTER.segment_automatic(rgb)
    except Exception as e:                          # pragma: no cover
        print(f"  [warn] SAM inference failed ({e}); skipping", file=sys.stderr)
        return None
    return mask if mask is not None and int((mask > 0).sum()) else None



def segment_seed(img: np.ndarray, cfg: Config,
                 override: Optional[dict] = None) -> Tuple[Optional[np.ndarray], dict]:
    """Segment one single-seed source.

    Returns (tightly-cropped BGRA patch or None, report). The report carries the
    method used, a 0..1 ``confidence``, and skip bookkeeping (``skipped`` /
    ``skip_reason``). In ``auto`` mode a low-confidence or None classical result is
    automatically routed to rembg (when available) before giving up, and a final
    confidence gate skips genuinely hopeless sources rather than baking a
    background patch into the dataset."""
    ov = override or {}
    method = ov.get("method", cfg.segment)
    tol = float(ov.get("bg_tolerance", cfg.bg_tolerance))
    strip = bool(ov.get("strip_shadow", cfg.strip_shadow))
    is_auto = (method == "auto")
    rep = {"method": method, "confidence": 0.0, "fg_fraction": 0.0,
           "solidity": 0.0, "n_components": 0, "border_touch": 0.0,
           "edge_contrast": 0.0, "otsu_separation": 0.0,
           "hit_safeguard": False, "skipped": False, "skip_reason": ""}

    # Use an existing alpha channel when it carries information.
    if img.ndim == 3 and img.shape[2] == 4:
        bgr, a = img[..., :3], img[..., 3]
        if method == "alpha" or a.min() < 250:
            patch = _crop_to_alpha(bgr, a)
            rep.update(method="alpha", confidence=1.0,
                       fg_fraction=float((a > 8).mean()))
            if patch is None:
                rep.update(skipped=True, skip_reason="empty_mask")
            return patch, rep
        img = bgr  # opaque alpha -> ignore and segment

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgr = img[..., :3]

    soft_alpha = None      # soft 0..255 alpha (rembg/sam) preserves feathered edges
    clean = None

    if is_auto:
        res = _auto_mask_scored(bgr, cfg)
        if res is not None:
            clean, q = res
            rep.update(q)
    elif method in ("rembg", "sam"):
        a = _mask_rembg(bgr) if method == "rembg" else _mask_sam(bgr, cfg)
        if a is not None:
            soft_alpha = a
            rawbin = ((a > 16).astype(np.uint8)) * 255
            clean = _largest_component_fill(rawbin)
            rep.update(_mask_quality(bgr, rawbin, clean))
            rep["method"] = method
        else:
            rep["skip_reason"] = "rembg_unavailable"
    else:                                           # otsu | grabcut | threshold
        if method == "otsu":
            raw = _mask_otsu(bgr, cfg.otsu_blur_kernel, cfg.otsu_threshold_adjustment)
        elif method == "grabcut":
            raw = _mask_grabcut(bgr, cfg.grabcut_iters, cfg.grabcut_rect_margin)
        else:
            raw = _mask_border_color(bgr, tol)
        clean = _largest_component_fill(raw, cfg.morph_kernel)
        rep.update(_mask_quality(bgr, raw, clean))
        rep["method"] = method
        if (method == "grabcut" and cfg.grabcut_min_object_area > 0
                and float((clean > 0).mean()) < float(cfg.grabcut_min_object_area)):
            clean = None                            # below the configured min area
            rep["skip_reason"] = "below_min_object_area"

    # ---- confidence-gated rembg fallback (auto mode only) ----
    wanted_fallback = (is_auto and cfg.rembg_fallback
                       and (clean is None or rep["confidence"] < cfg.seg_conf_thresh))
    if wanted_fallback:
        a = _mask_rembg(bgr)
        if a is not None:
            rbin = ((a > 16).astype(np.uint8)) * 255
            rclean = _largest_component_fill(rbin)
            rq = _mask_quality(bgr, rbin, rclean)
            if clean is None or rq["confidence"] > rep["confidence"]:
                clean, soft_alpha = rclean, a
                rep.update(rq)
                rep["method"] = "rembg"
        elif clean is None:
            rep["skip_reason"] = "rembg_unavailable"

    if clean is None or int((clean > 0).sum()) == 0:
        rep.update(skipped=True, skip_reason=rep["skip_reason"] or "empty_mask")
        return None, rep

    # ---- cast-shadow strip (classical masks only; rembg is already clean) ----
    if strip and rep["method"] in ("color_dist", "flatten", "border",
                                   "threshold", "otsu"):
        clean = _strip_cast_shadow(bgr, clean, cfg.shadow_sat_max)

    # paste alpha: soft within the cleaned blob when we have a learned/alpha mask
    if soft_alpha is not None:
        alpha = cv2.bitwise_and(soft_alpha, soft_alpha,
                                mask=(clean > 0).astype(np.uint8))
    else:
        alpha = clean

    # ---- oversize safeguard: never bake a >70%-of-frame background patch ----
    frac = float((clean > 0).mean())
    if frac > 0.7:
        rep.update(skipped=True, skip_reason="oversize_safeguard", hit_safeguard=True)
        return None, rep

    # ---- final confidence gate (auto only; fallback already attempted) ----
    if is_auto and rep["confidence"] < cfg.seg_conf_thresh:
        rep.update(skipped=True, skip_reason=rep["skip_reason"] or "low_confidence")
        return None, rep

    patch = _crop_to_alpha(bgr, alpha)
    if patch is None or patch.shape[0] < 4 or patch.shape[1] < 4:
        rep.update(skipped=True, skip_reason="too_small")
        return None, rep
    return patch, rep



def extract_background(img: np.ndarray, cfg: Config,
                       override: Optional[dict] = None) -> Optional[np.ndarray]:
    """Recover the bare background (e.g. the empty tray) from a single-seed
    source by masking the seed and inpainting it out. Returns a BGR image, or
    None for sources with a transparent background (nothing behind the seed).

    The inpaint ghost is erased by blending the filled region toward a smooth
    low-frequency tray model (heavy blur of the inpaint), so the vignette/gradient
    survive but the seed-shaped texture residue does not. ``cfg.ghost_blend``
    (0..1) tunes the strength."""
    if img.ndim == 3 and img.shape[2] == 4:
        if img[..., 3].min() < 250:        # transparent bg -> no surface to keep
            return None
        img = img[..., :3]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgr = img[..., :3]

    ov = override or {}
    method = ov.get("method", cfg.segment)
    tol = float(ov.get("bg_tolerance", cfg.bg_tolerance))
    if method == "otsu":
        mask = _mask_otsu(bgr, cfg.otsu_blur_kernel, cfg.otsu_threshold_adjustment)
    elif method == "grabcut":
        mask = _mask_grabcut(bgr, cfg.grabcut_iters, cfg.grabcut_rect_margin)
    elif method == "threshold":
        mask = _mask_border_color(bgr, tol)
    elif method in ("rembg", "sam"):
        a = _mask_rembg(bgr) if method == "rembg" else _mask_sam(bgr, cfg)
        mask = (((a > 16).astype(np.uint8)) * 255) if a is not None \
            else _auto_mask(bgr, tol)
    else:
        mask = _auto_mask(bgr, tol)
    if mask is None:
        return None
    mask = _largest_component_fill(mask, cfg.morph_kernel)
    if int(mask.sum()) == 0:
        return bgr                          # no seed found; whole image is bg

    h, w = bgr.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.dilate(mask, k, iterations=2)   # cover soft edge + slight shadow
    inp = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    # erase the residual ghost: blend the filled region toward a low-frequency
    # (seed-free) tray model rather than a blur that still contains the ghost.
    sigma = max(h, w) / 8.0
    low = cv2.GaussianBlur(inp, (0, 0), sigma)
    strength = float(np.clip(cfg.ghost_blend, 0.0, 1.0))
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 8)[..., None] * strength
    return (inp * (1 - soft) + low * soft).astype(np.uint8)



# ---- cut-out cache (keyed by file content hash + seg params + SEG_VERSION) ---- #
def _seg_params_signature(cfg: Config, override: Optional[dict]) -> str:
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



def _seg_cache_key(content_hash: str, cfg: Config,
                   override: Optional[dict], suffix: str = "") -> str:
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



def _cache_store(cache_dir: Path, key: str,
                 patch: Optional[np.ndarray], report: dict) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if patch is not None:
            cv2.imwrite(str(cache_dir / f"{key}.png"), patch)
        (cache_dir / f"{key}.json").write_text(json.dumps(
            {"seg_version": SEG_VERSION, "skipped": patch is None, "report": report}))
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



def _cache_store_bg(cache_dir: Path, key: str, bg: Optional[np.ndarray]) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if bg is not None:
            cv2.imwrite(str(cache_dir / f"{key}.bg.png"), bg)
        (cache_dir / f"{key}.bg.json").write_text(
            json.dumps({"present": bg is not None}))
    except Exception:
        pass



# ---- per-source method override manifest (--segment-map) ---- #
_VALID_METHODS = {"auto", "alpha", "threshold", "otsu", "grabcut", "rembg", "sam"}



def _load_segment_map(path: Path) -> list:
    if not path.exists():
        sys.exit(f"--segment-map file not found: {path}")
    rules = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        items = data.get("rules", []) if isinstance(data, dict) else data
        for r in items:
            rules.append({"glob": r["glob"], "method": r.get("method", "auto"),
                          "bg_tolerance": r.get("bg_tolerance"),
                          "strip_shadow": r.get("strip_shadow")})
    else:                                           # CSV: pattern,method[,tol]
        for row in csv.reader(path.read_text().splitlines()):
            if not row or row[0].strip().startswith("#"):
                continue
            rule = {"glob": row[0].strip(),
                    "method": row[1].strip() if len(row) > 1 and row[1].strip() else "auto",
                    "bg_tolerance": float(row[2]) if len(row) > 2 and row[2].strip() else None,
                    "strip_shadow": None}
            rules.append(rule)
    for r in rules:
        if r["method"] not in _VALID_METHODS:
            sys.exit(f"--segment-map: invalid method {r['method']!r} for {r['glob']!r}")
    return rules



def _match_override(rel: str, name: str, rules: Optional[list]) -> Optional[dict]:
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



def _segment_one(path_str: str, class_id: int, cfg: Config,
                 rules: Optional[list], cache_dir: Path) -> SegResult:
    """Picklable per-file worker: hash -> cache -> imread -> segment -> store.
    Pure per-file (no shared mutable state, no RNG) so it is deterministic and
    order-preserving under joblib."""
    cv2.setNumThreads(1)                            # avoid thread oversubscription
    _set_rembg_provider(cfg.rembg_provider)         # survive the loky fork
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
        return SegResult(None, {"skipped": True, "skip_reason": "read_error",
                                "method": "-", "confidence": 0.0}, rel, class_id)
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
            patch, report = None, {"skipped": True, "skip_reason": "read_error",
                                   "method": "-", "confidence": 0.0}
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



def _assign_seed_splits(seeds: List[Seed], cfg: Config) -> None:
    """Tag each Seed.split for seed-level train/val hold-out (prevents data leakage:
    a source seed used in val is never used in train). Default (no val_seeds globs and
    val_seed_holdout<=0): every seed stays 'both' -> legacy behaviour where both splits
    share all seeds. With a hold-out the partition is per-class stratified, deterministic
    in cfg.seed, and never empties a class from the train pool."""
    val_globs = cfg.val_seeds or None
    frac = float(cfg.val_seed_holdout or 0.0)
    if not val_globs and frac <= 0.0:
        return                                      # legacy: leave every seed "both"
    if val_globs:
        for s in seeds:
            name = Path(s.source).name
            is_val = any(fnmatch.fnmatch(s.source, g) or fnmatch.fnmatch(name, g)
                         for g in val_globs)
            s.split = "val" if is_val else "train"
        return
    by_class: Dict[int, List[int]] = {}
    for i, s in enumerate(seeds):
        by_class.setdefault(s.class_id, []).append(i)
    rng = np.random.default_rng(int(cfg.seed))
    for _cid, idxs in by_class.items():
        order = list(idxs)
        rng.shuffle(order)
        n_val = min(int(round(frac * len(order))), max(0, len(order) - 1))
        val_set = set(order[:n_val])                # keep >=1 per class in train
        for i in order:
            seeds[i].split = "val" if i in val_set else "train"


def load_seeds(cfg: Config):
    """Read and segment every source image (in parallel, with a disk cache).

    Returns (seeds, class_names, source_backgrounds, seg_results). seg_results is
    the per-source list (kept + skipped) used by write_seg_report; source_bgs is
    populated only when cfg.bg_from_sources is set."""
    root = cfg.sources
    if not root.exists():
        sys.exit(f"sources directory not found: {root}")
    # Resolve the rembg device ONCE in the parent against what ORT actually grants,
    # and rewrite to a concrete verified device so every worker's cache key + the job
    # cap reflect the masks really produced (not the build-time provider list).
    if (cfg.segment in ("rembg", "sam") or cfg.rembg_fallback
            or cfg.segment_map is not None):
        cfg.rembg_provider = _resolve_rembg_device(cfg.rembg_provider)
    _set_rembg_provider(cfg.rembg_provider)         # parent process (serial path)

    if cfg.class_mode == "subfolders":
        subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not subdirs:
            sys.exit("--class-mode subfolders given but no subdirectories found")
        class_names = [p.name for p in subdirs]
        groups = [(i, sorted(p.rglob("*"))) for i, p in enumerate(subdirs)]
    else:
        class_names = ["seed"]
        groups = [(0, sorted(root.rglob("*")))]

    rules = _load_segment_map(cfg.segment_map) if cfg.segment_map else None
    cache_dir = cfg.cache_dir if cfg.cache_dir is not None else (cfg.out / ".seg_cache")
    tasks = [(str(f), cid) for cid, files in groups for f in files
             if f.suffix.lower() in IMG_EXTS]

    # Bound concurrency when rembg runs on the GPU: many worker processes would each
    # open a CUDA session and exhaust VRAM. Only caps when GPU rembg is the method
    # for many sources (forced --segment rembg/sam, or a --segment-map); classical
    # parallelism is untouched in the default auto path.
    jobs = cfg.jobs
    gpu_rembg = (_effective_rembg_provider(cfg.rembg_provider) == "cuda"
                 and (cfg.segment in ("rembg", "sam") or cfg.rembg_fallback
                      or cfg.segment_map is not None))
    # SAM holds a full model per worker process (VRAM on GPU, large RAM on CPU), so it
    # is capped like GPU rembg regardless of the onnxruntime provider.
    heavy_sam = cfg.segment == "sam" and cfg.sam_checkpoint is not None
    if gpu_rembg or heavy_sam:
        cap = max(1, int(cfg.gpu_seg_jobs))
        if jobs <= 0 or jobs > cap:
            jobs = cap
        what = "SAM" if heavy_sam else "GPU rembg"
        print(f"  [{'sam' if heavy_sam else 'gpu'}] segmentation capped to {jobs} "
              f"worker(s) to fit memory (--gpu-seg-jobs); {what} stays faster per image")

    if _HAVE_JOBLIB and jobs != 1 and len(tasks) > 1:
        results: List[SegResult] = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_segment_one)(p, cid, cfg, rules, cache_dir) for p, cid in tasks)
    else:
        results = [_segment_one(p, cid, cfg, rules, cache_dir)
                   for p, cid in _tqdm(tasks, desc="segment", unit="img")]

    seeds: List[Seed] = []
    source_bgs: List[np.ndarray] = []
    for r in results:
        if r.patch is not None and r.patch.shape[0] >= 4 and r.patch.shape[1] >= 4:
            seeds.append(Seed(rgba=r.patch, class_id=r.class_id,
                              source=r.source, method=r.report.get("method", "")))
            if cfg.bg_from_sources and r.background is not None:
                source_bgs.append(r.background)

    # OPT-IN downscale cap: shrink oversized patches BEFORE staging/pooling so the
    # in-RAM pool (and the loky workers' shared copy) is bounded. Applied after the
    # cache load on purpose, so it never enters the segmentation cache key. Off (0) by
    # default -> output byte-identical; any value changes pixels, hence opt-in.
    if cfg.max_seed_px and cfg.max_seed_px > 0:
        from .compositing import _resize_to_long_side
        for s in seeds:
            h, w = s.rgba.shape[:2]
            if max(h, w) > cfg.max_seed_px:
                s.rgba = _resize_to_long_side(s.rgba, cfg.max_seed_px)

    if cfg.max_sources and len(seeds) > cfg.max_sources:
        idx = np.random.default_rng(cfg.seed).choice(
            len(seeds), cfg.max_sources, replace=False)
        seeds = [seeds[i] for i in idx]

    _assign_seed_splits(seeds, cfg)                 # seed-level train/val hold-out
    n_val_seeds = sum(1 for s in seeds if s.split == "val")
    if n_val_seeds:
        print(f"  seed hold-out: {len(seeds) - n_val_seeds} train-only / "
              f"{n_val_seeds} val-only source seed(s) (no overlap)")

    n_skip = sum(1 for r in results if r.report.get("skipped"))
    n_hit = sum(1 for r in results if r.cache_hit)
    print(f"  loaded {len(seeds)} seed patches ({n_skip} skipped, "
          f"{n_hit} from cache) across {len(class_names)} class(es): {class_names}")
    if cfg.bg_from_sources:
        print(f"  extracted {len(source_bgs)} domain background(s) from sources")
    if not seeds:
        sys.exit("no usable seeds were segmented - check --segment / "
                 "--seg-conf-thresh / --bg-tolerance, and read seg_report.txt")

    # Release the heavy pixel buffers we no longer need. Kept patches are already
    # referenced by Seed.rgba (the SAME ndarray, no copy) and any extracted
    # backgrounds by source_bgs, so this frees only the dropped/sub-sampled patches
    # and every background once collected. write_seg_report reads r.report /
    # r.cache_hit only, so the returned results stay valid for the report.
    for r in results:
        r.patch = None
        r.background = None
    return seeds, class_names, source_bgs, results



def write_seg_report(out: Path, results: List["SegResult"], cfg: Config) -> dict:
    """Write seg_report.{json,txt} and print a summary: per-method kept counts,
    skips grouped by reason, a confidence histogram, and cache stats."""
    out.mkdir(parents=True, exist_ok=True)
    method_counts: Dict[str, int] = {}
    skip_counts: Dict[str, int] = {}
    skipped_list = []
    confs = []
    for r in results:
        rep = r.report
        if rep.get("skipped"):
            reason = rep.get("skip_reason", "?")
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            skipped_list.append({"source": rep.get("source", "?"), "reason": reason,
                                 "confidence": round(float(rep.get("confidence", 0.0)), 3)})
        else:
            m = rep.get("method", "?")
            method_counts[m] = method_counts.get(m, 0) + 1
            confs.append(float(rep.get("confidence", 0.0)))

    bins = [0] * 10
    for c in confs:
        bins[min(9, int(c * 10))] += 1
    n_hit = sum(1 for r in results if r.cache_hit)
    summary = {
        "scanned": len(results), "kept": len(confs), "skipped": len(skipped_list),
        "cache_hits": n_hit, "method_counts": method_counts,
        "skip_counts": skip_counts,
        "confidence": {
            "min": round(min(confs), 3) if confs else None,
            "mean": round(float(np.mean(confs)), 3) if confs else None,
            "median": round(float(np.median(confs)), 3) if confs else None,
            "histogram_0_1_by_0.1": bins,
        },
    }
    (out / "seg_report.json").write_text(json.dumps(
        {"summary": summary, "skipped": skipped_list}, indent=2))

    lines = ["seedgen segmentation report",
             f"  scanned {summary['scanned']}  kept {summary['kept']}  "
             f"skipped {summary['skipped']}  cache_hits {n_hit}",
             f"  by method: {method_counts}",
             f"  skipped by reason: {skip_counts}",
             f"  confidence min/median/mean: {summary['confidence']['min']}/"
             f"{summary['confidence']['median']}/{summary['confidence']['mean']}",
             f"  confidence histogram [0..1 by 0.1]: {bins}"]
    if skipped_list:
        lines.append("  skipped sources (first 40):")
        for s in skipped_list[:40]:
            lines.append(f"    {s['reason']:18s} conf={s['confidence']:.3f}  {s['source']}")
    text = "\n".join(lines)
    (out / "seg_report.txt").write_text(text + "\n")
    print(text)
    return summary
