"""Compositing primitives: region intersection, alpha/seamless blend, placement, IoU."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import Config


def _intersect(x: int, y: int, pw: int, ph: int, W: int, H: int):
    """Return (cx0,cy0,cx1,cy1, px0,py0) for the on-canvas region, or None."""
    cx0, cy0 = max(0, x), max(0, y)
    cx1, cy1 = min(W, x + pw), min(H, y + ph)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return cx0, cy0, cx1, cy1, cx0 - x, cy0 - y


def _blend_region(canvas: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite an rgb patch (float) with alpha (H,W,1 float 0..1)."""
    H, W = canvas.shape[:2]
    ph, pw = alpha.shape[:2]
    r = _intersect(x, y, pw, ph, W, H)
    if r is None:
        return
    cx0, cy0, cx1, cy1, px0, py0 = r
    a = alpha[py0 : py0 + (cy1 - cy0), px0 : px0 + (cx1 - cx0)]
    src = rgb[py0 : py0 + (cy1 - cy0), px0 : px0 + (cx1 - cx0)]
    dst = canvas[cy0:cy1, cx0:cx1]
    canvas[cy0:cy1, cx0:cx1] = dst * (1 - a) + src * a


def _try_seamless(
    canvas: np.ndarray,
    rgb_reg: np.ndarray,
    on_opaque: np.ndarray,
    cx0: int,
    cy0: int,
    cx1: int,
    cy1: int,
) -> bool:
    """Poisson/seamless-clone the seed region into the canvas. Returns False (a
    no-op) when seamlessClone cannot run safely (tiny region, mask touching the
    region border, too-small mask, or a cv2 error) so the caller falls back to
    alpha. Only the canvas PIXELS change here; box/owner math stays alpha-derived."""
    rh, rw = on_opaque.shape
    if rh < 6 or rw < 6:
        return False
    m = (on_opaque.astype(np.uint8)) * 255
    if int((m > 0).sum()) < 16:
        return False
    if m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any():
        return False  # mask must be strictly interior
    ys, xs = np.where(on_opaque)
    center = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))
    dst = np.clip(canvas[cy0:cy1, cx0:cx1], 0, 255).astype(np.uint8)
    src = np.clip(rgb_reg, 0, 255).astype(np.uint8)
    try:
        out = cv2.seamlessClone(src, dst, m, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return False
    canvas[cy0:cy1, cx0:cx1] = out.astype(np.float32)
    return True


def place_seed(
    canvas: np.ndarray,
    owner: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int,
    seed_id: int,
    cfg: Config,
    light: tuple[float, float] = (1.0, 1.0),
) -> dict | None:
    """Composite one seed (with optional shadow), claim ownership, return box.

    Returns dict(box=(x0,y0,x1,y1), denom=int) where box is the seed's
    on-canvas silhouette extent and denom is its on-canvas opaque pixel count,
    or None if the seed lands (almost) entirely off-frame.
    """
    H, W = canvas.shape[:2]
    ph, pw = rgba.shape[:2]
    # Hard silhouette drives ALL label geometry (owner/box/denom); the feathered
    # copy is used ONLY for the RGB alpha-composite (anti-aliased edge). Keeping
    # them separate means edge_feather never moves boxes/polygons.
    alpha_hard = rgba[..., 3].astype(np.float32) / 255.0
    alpha: np.ndarray = alpha_hard
    if cfg.edge_feather > 0:
        alpha = cv2.GaussianBlur(alpha_hard, (0, 0), cfg.edge_feather)
    rgb = rgba[..., :3].astype(np.float32)

    full_opaque = int((alpha_hard >= 0.5).sum())
    if full_opaque == 0:
        return None

    # ---- drop shadow (under the seed) ----
    # Rendered on a ZERO-PADDED copy of the silhouette so the Gaussian tapers to 0
    # and fades into the background. Blurring the tight (ph,pw) cut-out directly
    # clipped the shadow to the bbox (and reflect-border kept it ~full strength at
    # the edge), producing a hard rectangular "box" halo around every seed.
    if cfg.shadow:
        lx, ly = light
        ox = int(round(cfg.shadow_offset * lx))
        oy = int(round(cfg.shadow_offset * ly))
        blur = cfg.shadow_blur
        if cfg.shadow_dir_mode == "scene":  # softness tied to seed size
            blur = cfg.shadow_blur * (0.5 + 2.0 * max(ph, pw) / max(1, cfg.img_size))
        blur = max(0.1, blur)
        pad = int(np.ceil(3.0 * blur)) + 1
        a_pad = cv2.copyMakeBorder(alpha_hard, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)
        sh = (cv2.GaussianBlur(a_pad, (0, 0), blur) * cfg.shadow_opacity)[..., None]
        shadow_rgb = np.zeros((sh.shape[0], sh.shape[1], 3), np.float32)  # black shadow
        _blend_region(canvas, shadow_rgb, sh, x + ox - pad, y + oy - pad)

    # ---- the seed itself ----
    r = _intersect(x, y, pw, ph, W, H)
    if r is None:
        return None
    cx0, cy0, cx1, cy1, px0, py0 = r
    a_reg = alpha[py0 : py0 + (cy1 - cy0), px0 : px0 + (cx1 - cx0)]  # feathered (blend)
    ah_reg = alpha_hard[py0 : py0 + (cy1 - cy0), px0 : px0 + (cx1 - cx0)]  # hard (labels)
    rgb_reg = rgb[py0 : py0 + (cy1 - cy0), px0 : px0 + (cx1 - cx0)]

    on_opaque = ah_reg >= 0.5
    denom = int(on_opaque.sum())
    if denom == 0 or denom / full_opaque < cfg.min_visibility:
        return None  # mostly off-frame: don't paste, don't label

    # Paste pixels: seamless clone (optional) or alpha-composite (default). The
    # owner-map and box below are derived from the HARD alpha regardless of blend mode.
    pasted = False
    if cfg.blend == "seamless":
        pasted = _try_seamless(canvas, rgb_reg, on_opaque, cx0, cy0, cx1, cy1)
    if not pasted:
        a3 = a_reg[..., None]
        dst = canvas[cy0:cy1, cx0:cx1]
        canvas[cy0:cy1, cx0:cx1] = dst * (1 - a3) + rgb_reg * a3
    owner[cy0:cy1, cx0:cx1][on_opaque] = seed_id

    ys, xs = np.where(on_opaque)
    box = (
        cx0 + int(xs.min()),
        cy0 + int(ys.min()),
        cx0 + int(xs.max()) + 1,
        cy0 + int(ys.max()) + 1,
    )
    return {"box": box, "denom": denom}


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)
