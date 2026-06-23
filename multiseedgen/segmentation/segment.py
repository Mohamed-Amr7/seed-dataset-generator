"""Per-source segmentation dispatch: ``segment_seed`` (method cascade + confidence
gating + cast-shadow removal) and ``extract_background`` (inpaint the seed out to recover
the domain background, e.g. the empty tray)."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import Config
from .masks import (
    _auto_mask,
    _auto_mask_scored,
    _largest_component_fill,
    _mask_border_color,
    _mask_grabcut,
    _mask_otsu,
    _mask_quality,
    _strip_cast_shadow,
)
from .session import _crop_to_alpha, _mask_rembg, _mask_sam


def segment_seed(
    img: np.ndarray, cfg: Config, override: dict | None = None
) -> tuple[np.ndarray | None, dict]:
    """Segment one single-seed source.

    Returns (tightly-cropped BGRA patch or None, report). The report carries the
    method used, a 0..1 ``confidence``, and skip bookkeeping (``skipped`` /
    ``skip_reason``). In ``auto`` mode a low-confidence or None classical result is
    automatically routed to rembg (when available) before giving up. A final
    confidence gate then skips genuinely hopeless sources rather than baking a
    background patch into the dataset: it always runs for ``auto`` and is opt-in for
    explicit methods (active when ``seg_conf_thresh`` > 0, which defaults to 0 there)."""
    ov = override or {}
    method = ov.get("method", cfg.segment)
    tol = float(ov.get("bg_tolerance", cfg.bg_tolerance))
    strip = bool(ov.get("strip_shadow", cfg.strip_shadow))
    is_auto = method == "auto"
    rep = {
        "method": method,
        "confidence": 0.0,
        "fg_fraction": 0.0,
        "solidity": 0.0,
        "n_components": 0,
        "border_touch": 0.0,
        "edge_contrast": 0.0,
        "otsu_separation": 0.0,
        "hit_safeguard": False,
        "skipped": False,
        "skip_reason": "",
    }

    # Use an existing alpha channel when it carries information.
    if img.ndim == 3 and img.shape[2] == 4:
        bgr, a = img[..., :3], img[..., 3]
        if method == "alpha" or a.min() < 250:
            patch = _crop_to_alpha(bgr, a)
            rep.update(method="alpha", confidence=1.0, fg_fraction=float((a > 8).mean()))
            if patch is None:
                rep.update(skipped=True, skip_reason="empty_mask")
            return patch, rep
        img = bgr  # opaque alpha -> ignore and segment

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgr = img[..., :3]

    soft_alpha = None  # soft 0..255 alpha (rembg/sam) preserves feathered edges
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
    else:  # otsu | grabcut | threshold
        if method == "otsu":
            raw = _mask_otsu(bgr, cfg.otsu_blur_kernel, cfg.otsu_threshold_adjustment)
        elif method == "grabcut":
            raw = _mask_grabcut(bgr, cfg.grabcut_iters, cfg.grabcut_rect_margin)
        else:
            raw = _mask_border_color(bgr, tol)
        clean = _largest_component_fill(raw, cfg.morph_kernel)
        rep.update(_mask_quality(bgr, raw, clean))
        rep["method"] = method
        if (
            method == "grabcut"
            and cfg.grabcut_min_object_area > 0
            and float((clean > 0).mean()) < float(cfg.grabcut_min_object_area)
        ):
            clean = None  # below the configured min area
            rep["skip_reason"] = "below_min_object_area"

    # ---- confidence-gated rembg fallback (auto mode only) ----
    wanted_fallback = (
        is_auto
        and cfg.rembg_fallback
        and (clean is None or rep["confidence"] < cfg.seg_conf_thresh)
    )
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
    if strip and rep["method"] in ("color_dist", "flatten", "border", "threshold", "otsu"):
        clean = _strip_cast_shadow(bgr, clean, cfg.shadow_sat_max)

    # paste alpha: soft within the cleaned blob when we have a learned/alpha mask
    if soft_alpha is not None:
        alpha = cv2.bitwise_and(soft_alpha, soft_alpha, mask=(clean > 0).astype(np.uint8))
    else:
        alpha = clean

    # ---- oversize safeguard: never bake a >70%-of-frame background patch ----
    frac = float((clean > 0).mean())
    if frac > 0.7:
        rep.update(skipped=True, skip_reason="oversize_safeguard", hit_safeguard=True)
        return None, rep

    # ---- final confidence gate ----
    # Auto always applies it (fallback already attempted above); explicit-method modes
    # apply it only when opted in. The opt-in is a non-zero ``seg_conf_thresh``: it
    # defaults to 0 for non-auto methods (see cliconfig.parse_args / the tuner), so an
    # un-tuned explicit run keeps every mask and existing behaviour is unchanged.
    if rep["confidence"] < cfg.seg_conf_thresh:
        rep.update(skipped=True, skip_reason=rep["skip_reason"] or "low_confidence")
        return None, rep

    patch = _crop_to_alpha(bgr, alpha)
    if patch is None or patch.shape[0] < 4 or patch.shape[1] < 4:
        rep.update(skipped=True, skip_reason="too_small")
        return None, rep
    return patch, rep


def extract_background(
    img: np.ndarray, cfg: Config, override: dict | None = None
) -> np.ndarray | None:
    """Recover the bare background (e.g. the empty tray) from a single-seed
    source by masking the seed and inpainting it out. Returns a BGR image, or
    None for sources with a transparent background (nothing behind the seed).

    The inpaint ghost is erased by blending the filled region toward a smooth
    low-frequency tray model (heavy blur of the inpaint), so the vignette/gradient
    survive but the seed-shaped texture residue does not. ``cfg.ghost_blend``
    (0..1) tunes the strength."""
    if img.ndim == 3 and img.shape[2] == 4:
        if img[..., 3].min() < 250:  # transparent bg -> no surface to keep
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
        mask = (((a > 16).astype(np.uint8)) * 255) if a is not None else _auto_mask(bgr, tol)
    else:
        mask = _auto_mask(bgr, tol)
    if mask is None:
        return None
    mask = _largest_component_fill(mask, cfg.morph_kernel)
    if int(mask.sum()) == 0:
        return bgr  # no seed found; whole image is bg

    h, w = bgr.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.dilate(mask, k, iterations=2)  # cover soft edge + slight shadow
    inp = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    # erase the residual ghost: blend the filled region toward a low-frequency
    # (seed-free) tray model rather than a blur that still contains the ghost.
    sigma = max(h, w) / 8.0
    low = cv2.GaussianBlur(inp, (0, 0), sigma)
    strength = float(np.clip(cfg.ghost_blend, 0.0, 1.0))
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 8)[..., None] * strength
    return (inp * (1 - soft) + low * soft).astype(np.uint8)
