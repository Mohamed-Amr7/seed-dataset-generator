from __future__ import annotations
"""Seed transforms, backgrounds, placement, scene composition."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .degrade import apply_degradation
from .models import Config, Seed




# --------------------------------------------------------------------------- #
# Per-seed transforms                                                         #
# --------------------------------------------------------------------------- #
def _resize_to_long_side(rgba: np.ndarray, long_px: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = long_px / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(rgba, (nw, nh), interpolation=interp)



def _rotate(rgba: np.ndarray, angle: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))



def _jitter(rgba: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    rgb = rgba[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.jitter_hue, cfg.jitter_hue)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - cfg.jitter_sat, 1 + cfg.jitter_sat), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(1 - cfg.jitter_val, 1 + cfg.jitter_val), 0, 255)
    out = rgba.copy()
    out[..., :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out



def transform_seed(seed: Seed, cfg: Config, canvas: int,
                   rng: np.random.Generator, record: bool = False):
    """Scale/flip/rotate/jitter a seed cut-out. Draw order is unchanged whether or
    not ``record`` is set (it only collects the sampled params), so determinism is
    preserved. With record=True returns (rgba, params); else just rgba."""
    rgba = seed.rgba
    scale_frac = rng.uniform(cfg.scale_min, cfg.scale_max)
    long_px = max(4, int(scale_frac * canvas))
    rgba = _resize_to_long_side(rgba, long_px)
    flip_h = cfg.flip and rng.random() < 0.5
    if flip_h:
        rgba = rgba[:, ::-1].copy()
    flip_v = cfg.flip and rng.random() < 0.5
    if flip_v:
        rgba = rgba[::-1, :].copy()
    rot = 0.0
    if cfg.max_rotation > 0:
        rot = float(rng.uniform(-cfg.max_rotation, cfg.max_rotation))
        rgba = _rotate(rgba, rot)
    if cfg.jitter_hue or cfg.jitter_sat or cfg.jitter_val:
        rgba = _jitter(rgba, cfg, rng)
    if record:
        return rgba, {"scale": round(float(scale_frac), 4), "flip_h": bool(flip_h),
                      "flip_v": bool(flip_v), "rotation": round(rot, 2)}
    return rgba



# --------------------------------------------------------------------------- #
# Backgrounds                                                                 #
# --------------------------------------------------------------------------- #
def _bg_solid(size: int, rng) -> np.ndarray:
    color = rng.integers(30, 226, size=3)
    return np.full((size, size, 3), color, dtype=np.uint8)



def _bg_gradient(size: int, rng) -> np.ndarray:
    c0 = rng.integers(20, 236, size=3).astype(np.float32)
    c1 = rng.integers(20, 236, size=3).astype(np.float32)
    line = np.linspace(0, 1, size, dtype=np.float32)
    if rng.random() < 0.5:                       # vertical gradient
        t = np.repeat(line[:, None], size, axis=1)
    else:                                        # horizontal gradient
        t = np.repeat(line[None, :], size, axis=0)
    t = t[..., None]                             # (H, W, 1)
    grad = c0[None, None, :] * (1 - t) + c1[None, None, :] * t
    return np.clip(grad, 0, 255).astype(np.uint8)



def _bg_noise(size: int, rng) -> np.ndarray:
    base = rng.integers(40, 216, size=3).astype(np.float32)
    noise = rng.normal(0, 18, size=(size, size, 3)).astype(np.float32)
    img = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.5))



def _bg_texture(size: int, textures: List[np.ndarray], rng) -> np.ndarray:
    """Random square crop from a real photo, with random zoom + flip, resized to
    the canvas. A handful of real photos thus yield many distinct backgrounds."""
    img = textures[rng.integers(0, len(textures))]
    h, w = img.shape[:2]
    short = min(h, w)
    # crop between 40% and 100% of the short side -> scale variety ("zoom")
    crop = int(max(16, min(short, rng.uniform(0.4, 1.0) * short)))
    y = int(rng.integers(0, h - crop + 1))
    x = int(rng.integers(0, w - crop + 1))
    patch = img[y:y + crop, x:x + crop]
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    if rng.random() < 0.5:
        patch = patch[::-1, :]
    interp = cv2.INTER_AREA if crop > size else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interp)



def _bg_seedbed(size: int, seeds: List[Seed], rng) -> np.ndarray:
    """A blurred clutter background made of out-of-focus seeds: forces the
    detector to separate in-focus seeds from seed-coloured background."""
    canvas = _bg_solid(size, rng).astype(np.float32)
    for _ in range(rng.integers(8, 20)):
        s = seeds[rng.integers(0, len(seeds))].rgba
        big = _resize_to_long_side(s, int(rng.uniform(0.25, 0.6) * size))
        a = cv2.GaussianBlur(big[..., 3].astype(np.float32) / 255.0, (0, 0), 6.0)[..., None]
        rgb = cv2.GaussianBlur(big[..., :3], (0, 0), 6.0).astype(np.float32)
        ph, pw = big.shape[:2]
        x = int(rng.integers(-pw // 2, size - pw // 2))
        y = int(rng.integers(-ph // 2, size - ph // 2))
        _blend_region(canvas, rgb, a, x, y)
    return cv2.GaussianBlur(canvas, (0, 0), rng.uniform(2.0, 5.0)).astype(np.uint8)



def make_background(cfg: Config, seeds: List[Seed], textures: List[np.ndarray],
                    rng: np.random.Generator) -> np.ndarray:
    # When real background photos are supplied, use them most of the time and
    # keep a little synthetic variety (domain randomization aids generalization
    # even with real backgrounds). Real backgrounds are strongly preferred.
    if textures and rng.random() < 0.7:
        return _bg_texture(cfg.img_size, textures, rng).astype(np.float32)

    kinds = [k for k in cfg.backgrounds if k != "texture"] or ["solid"]
    kind = kinds[rng.integers(0, len(kinds))]
    if kind == "solid":
        bg = _bg_solid(cfg.img_size, rng)
    elif kind == "gradient":
        bg = _bg_gradient(cfg.img_size, rng)
    elif kind == "noise":
        bg = _bg_noise(cfg.img_size, rng)
    elif kind == "seedbed":
        bg = _bg_seedbed(cfg.img_size, seeds, rng)
    elif kind == "texture" and textures:
        bg = _bg_texture(cfg.img_size, textures, rng)
    else:
        bg = _bg_solid(cfg.img_size, rng)
    return bg.astype(np.float32)



# --------------------------------------------------------------------------- #
# Compositing primitives                                                      #
# --------------------------------------------------------------------------- #
def _intersect(x: int, y: int, pw: int, ph: int, W: int, H: int):
    """Return (cx0,cy0,cx1,cy1, px0,py0) for the on-canvas region, or None."""
    cx0, cy0 = max(0, x), max(0, y)
    cx1, cy1 = min(W, x + pw), min(H, y + ph)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return cx0, cy0, cx1, cy1, cx0 - x, cy0 - y



def _blend_region(canvas: np.ndarray, rgb: np.ndarray, alpha: np.ndarray,
                  x: int, y: int) -> None:
    """Alpha-composite an rgb patch (float) with alpha (H,W,1 float 0..1)."""
    H, W = canvas.shape[:2]
    ph, pw = alpha.shape[:2]
    r = _intersect(x, y, pw, ph, W, H)
    if r is None:
        return
    cx0, cy0, cx1, cy1, px0, py0 = r
    a = alpha[py0:py0 + (cy1 - cy0), px0:px0 + (cx1 - cx0)]
    src = rgb[py0:py0 + (cy1 - cy0), px0:px0 + (cx1 - cx0)]
    dst = canvas[cy0:cy1, cx0:cx1]
    canvas[cy0:cy1, cx0:cx1] = dst * (1 - a) + src * a



def _try_seamless(canvas: np.ndarray, rgb_reg: np.ndarray, on_opaque: np.ndarray,
                  cx0: int, cy0: int, cx1: int, cy1: int) -> bool:
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
        return False                                # mask must be strictly interior
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



def place_seed(canvas: np.ndarray, owner: np.ndarray, rgba: np.ndarray,
               x: int, y: int, seed_id: int, cfg: Config,
               light: Tuple[float, float] = (1.0, 1.0)) -> Optional[dict]:
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
    alpha = alpha_hard
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
        if cfg.shadow_dir_mode == "scene":          # softness tied to seed size
            blur = cfg.shadow_blur * (0.5 + 2.0 * max(ph, pw) / max(1, cfg.img_size))
        blur = max(0.1, blur)
        pad = int(np.ceil(3.0 * blur)) + 1
        a_pad = cv2.copyMakeBorder(alpha_hard, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=0.0)
        sh = (cv2.GaussianBlur(a_pad, (0, 0), blur) * cfg.shadow_opacity)[..., None]
        shadow_rgb = np.zeros((sh.shape[0], sh.shape[1], 3), np.float32)  # black shadow
        _blend_region(canvas, shadow_rgb, sh, x + ox - pad, y + oy - pad)

    # ---- the seed itself ----
    r = _intersect(x, y, pw, ph, W, H)
    if r is None:
        return None
    cx0, cy0, cx1, cy1, px0, py0 = r
    a_reg = alpha[py0:py0 + (cy1 - cy0), px0:px0 + (cx1 - cx0)]        # feathered (blend)
    ah_reg = alpha_hard[py0:py0 + (cy1 - cy0), px0:px0 + (cx1 - cx0)]  # hard (labels)
    rgb_reg = rgb[py0:py0 + (cy1 - cy0), px0:px0 + (cx1 - cx0)]

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
    box = (cx0 + int(xs.min()), cy0 + int(ys.min()),
           cx0 + int(xs.max()) + 1, cy0 + int(ys.max()) + 1)
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



# --------------------------------------------------------------------------- #
# Scene generation                                                            #
# --------------------------------------------------------------------------- #
def _mask_to_polygons(bin_mask: np.ndarray, min_area: float = 6.0) -> list:
    """Contour polygons (Nx2 int arrays) of a binary mask, lightly simplified.
    Used for the VISIBLE region of a seed (owner == seed_id), so polygons are
    occlusion-aware and in-frame by construction."""
    cnts, _ = cv2.findContours(bin_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 1.0, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(np.int32))
    return polys



def generate_scene(cfg: Config, seeds: List[Seed], textures: List[np.ndarray],
                   rng: np.random.Generator, negative: bool = False):
    """Return (image_uint8, labels, polys, prov).

    labels = [(class_id, x0, y0, x1, y1), ...]; polys is a parallel list (one entry
    per label, each a list of Nx2 int polygon pieces) when cfg.masks else None;
    prov is a parallel list of provenance dicts when cfg.provenance else None.
    If ``negative`` is True, returns a background-only image with empty labels."""
    canvas = make_background(cfg, seeds, textures, rng)
    H, W = canvas.shape[:2]
    if negative:
        img = canvas
        if cfg.degrade:
            img = apply_degradation(canvas, cfg, rng)
        return np.clip(img, 0, 255).astype(np.uint8), [], None, None
    owner = np.full((H, W), -1, dtype=np.int32)

    # one light direction per scene (all seeds cast consistently) in scene mode
    if cfg.shadow and cfg.shadow_dir_mode == "scene":
        ang = float(rng.uniform(0, 2 * np.pi))
        light = (float(np.cos(ang)), float(np.sin(ang)))
    else:
        light = (1.0, 1.0)

    n = int(rng.integers(cfg.min_seeds, cfg.max_seeds + 1))
    placed: List[dict] = []
    boxes_for_iou: List[Tuple[int, int, int, int]] = []

    seed_id = 0
    for _ in range(n):
        seed_index = int(rng.integers(0, len(seeds)))
        chosen = seeds[seed_index]
        if cfg.provenance:
            rgba, tparams = transform_seed(chosen, cfg, cfg.img_size, rng, record=True)
        else:
            rgba, tparams = transform_seed(chosen, cfg, cfg.img_size, rng), None
        ph, pw = rgba.shape[:2]

        # try a few positions (so overlap-limit mode has room to work)
        result = None
        for _try in range(8):
            x = int(rng.integers(int(-0.5 * pw), int(W - 0.5 * pw) + 1))
            y = int(rng.integers(int(-0.5 * ph), int(H - 0.5 * ph) + 1))

            if cfg.overlap_mode == "limit" and boxes_for_iou:
                provisional = (max(0, x), max(0, y),
                               min(W, x + pw), min(H, y + ph))
                if any(_iou(provisional, b) > cfg.max_iou for b in boxes_for_iou):
                    continue

            result = place_seed(canvas, owner, rgba, x, y, seed_id, cfg, light)
            if result is not None:
                break

        if result is None:
            continue
        placed.append({"seed_id": seed_id, "class_id": chosen.class_id,
                       "seed_index": seed_index, "tparams": tparams, **result})
        boxes_for_iou.append(result["box"])
        seed_id += 1

    # ---- occlusion filter: how much of each seed survived later pastes ----
    labels: List[Tuple[int, int, int, int, int]] = []
    polys = [] if cfg.masks else None
    prov = [] if cfg.provenance else None
    for p in placed:
        visible = int((owner == p["seed_id"]).sum())
        if p["denom"] == 0:
            continue
        if visible / p["denom"] < cfg.min_visible_occluded:
            continue  # buried below threshold -> not labelled
        x0, y0, x1, y1 = p["box"]
        if min(x1 - x0, y1 - y0) < cfg.min_box_px:
            continue
        labels.append((p["class_id"], x0, y0, x1, y1))
        if polys is not None:
            pieces = _mask_to_polygons((owner == p["seed_id"]).astype(np.uint8))
            polys.append(pieces)
        if prov is not None:
            prov.append({"seed_index": p["seed_index"], "class_id": p["class_id"],
                         "box": [x0, y0, x1, y1], "transform": p["tparams"]})

    img = apply_degradation(canvas, cfg, rng) if cfg.degrade else canvas
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, labels, polys, prov
