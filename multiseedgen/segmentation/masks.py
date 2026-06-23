"""Classical mask algorithms: border/colour/brightness cues, quality scoring,
cast-shadow stripping, and the confidence-scored auto-cascade."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import Config


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
    border = (
        np.concatenate([bgr[0, :, :], bgr[-1, :, :], bgr[:, 0, :], bgr[:, -1, :]], axis=0)
        .reshape(-1, 3)
        .astype(np.float32)
    )
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
    if (m > 0).mean() > 0.5:  # a single seed should be the minority
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
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats((raw > 0).astype(np.uint8), 8)
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
    colordiff = (
        float(np.linalg.norm(fg_mean - bg_mean))
        if fg_mean is not None and bg_mean is not None
        else 0.0
    )
    s_contrast = float(np.clip(colordiff / 25.0, 0, 1))

    # Edge strength gates the score (multiplier): a confident cut-out needs a real,
    # well-placed boundary. Shape terms then grade the blob quality.
    shape = 0.55 * s_sol + 0.28 * s_comp + 0.17 * s_border
    conf = s_frac * s_edge * max(s_contrast, 0.4) * shape
    return {
        "confidence": float(conf),
        "fg_fraction": float(frac),
        "solidity": float(sol),
        "n_components": int(n_lbl - 1),
        "border_touch": float(bt),
        "edge_contrast": float(s_edge),
        "ring_grad": round(ring_grad, 1),
        "colordiff": round(colordiff, 1),
        "hit_safeguard": bool(frac > 0.7),
    }


def _strip_cast_shadow(bgr: np.ndarray, mask: np.ndarray, sat_max: float = 0.35) -> np.ndarray:
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
    methods = list(_AUTO_METHODS) + [("border", lambda b: _mask_border_color(b, cfg.bg_tolerance))]
    for name, fn in methods:
        raw = fn(bgr)
        clean = _largest_component_fill(raw)
        q = _mask_quality(bgr, raw, clean)
        q["method"] = name
        if 5e-4 <= q["fg_fraction"] <= 0.6 and q["confidence"] >= cfg.seg_conf_thresh:
            return clean, q
        if best is None or q["confidence"] > best[1]["confidence"]:
            best = (clean, q)
    return best


def _auto_mask(bgr: np.ndarray, tol: float) -> np.ndarray | None:
    """Back-compat shim: return the highest-confidence auto mask, or None when it
    trips the oversize safeguard (kept so extract_background's legacy path works)."""

    class _Shim:
        bg_tolerance = tol
        seg_conf_thresh = 1.1  # never short-circuit; always pick best-of-three

    clean, q = _auto_mask_scored(bgr, _Shim())
    if clean is None or q.get("fg_fraction", 1.0) > 0.7:
        return None
    return clean


def _mask_otsu(bgr: np.ndarray, blur_kernel: int = 0, threshold_adjustment: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur_kernel and int(blur_kernel) > 0:
        kk = int(blur_kernel) | 1  # gaussian needs an odd kernel
        gray = cv2.GaussianBlur(gray, (kk, kk), 0)
    thr, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if threshold_adjustment:  # re-threshold at Otsu +/- offset
        t2 = float(np.clip(thr + threshold_adjustment, 0, 255))
        _, m = cv2.threshold(gray, t2, 255, cv2.THRESH_BINARY)
    # assume the seed is the darker-or-lighter minority; flip if bg got selected
    if m.mean() > 127:
        m = 255 - m
    return m


def _mask_grabcut(bgr: np.ndarray, iters: int = 5, rect_margin: float = 0.06) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    inset = max(2, int(float(rect_margin) * min(h, w)))
    rect = (inset, inset, max(1, w - 2 * inset), max(1, h - 2 * inset))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, max(1, int(iters)), cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
