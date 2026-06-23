"""QA charts, montages, preview/checkerboard renderers."""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np

from .models import Config, Seed, _QAStats

logger = logging.getLogger(__name__)


def _bar_chart(values, labels, title, w=560, h=340) -> np.ndarray:
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = 50, 40, w - 16, h - 60
    cv2.line(img, (x0, y1), (x1, y1), (0, 0, 0), 1)
    cv2.line(img, (x0, y0), (x0, y1), (0, 0, 0), 1)
    vmax = max(values) if values and max(values) > 0 else 1
    n = max(1, len(values))
    bw = max(1, int((x1 - x0) / (n * 1.3)))
    for i, v in enumerate(values):
        bx = x0 + 8 + int(i * (x1 - x0) / n)
        bh = int((y1 - y0) * v / vmax)
        cv2.rectangle(img, (bx, y1 - bh), (bx + bw, y1), (180, 110, 40), -1)
        cv2.putText(
            img,
            str(v),
            (bx, y1 - bh - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        lab = str(labels[i])[:10] if i < len(labels) else ""
        cv2.putText(
            img, lab, (bx, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA
        )
    return img


def _hist_chart(samples, title, bins=20) -> np.ndarray:
    if not samples:
        return _bar_chart([], [], title)
    counts, edges = np.histogram(np.asarray(samples, np.float64), bins=bins)
    labels = [f"{edges[i]:.0f}" for i in range(len(counts))]
    return _bar_chart([int(c) for c in counts], labels, title)


def _montage_from_disk(cfg: Config, class_names: list[str], n: int) -> np.ndarray | None:
    """Tile the first n train composites with their EXPORTED boxes drawn on (read
    back from disk), so the montage validates the written labels."""
    img_dir = cfg.out / "train" / "images"
    imgs = sorted(img_dir.glob(f"*.{cfg.img_ext}"))[: max(1, n)]
    if not imgs:
        return None
    coco = None
    if cfg.fmt in ("coco", "both"):
        data = json.loads((cfg.out / "annotations_train.json").read_text())
        by_name = {im["file_name"].split("/")[-1]: im["id"] for im in data["images"]}
        anns: dict = {}
        for a in data["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        coco = (by_name, anns)
    thumbs = []
    for ip in imgs:
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        lp = cfg.out / "train" / "labels" / (ip.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                vals = [float(v) for v in parts[1:]]
                if len(vals) == 4:
                    cx, cy, bw, bh = vals[0] * W, vals[1] * H, vals[2] * W, vals[3] * H
                    cv2.rectangle(
                        im,
                        (int(cx - bw / 2), int(cy - bh / 2)),
                        (int(cx + bw / 2), int(cy + bh / 2)),
                        (0, 0, 255),
                        2,
                    )
                else:
                    pts = np.array(vals, np.float32).reshape(-1, 2)
                    pts[:, 0] *= W
                    pts[:, 1] *= H
                    cv2.polylines(im, [pts.astype(np.int32)], True, (0, 0, 255), 2)
        elif coco is not None:
            by_name, anns = coco
            for a in anns.get(by_name.get(ip.name, -1), []):
                x, y, bw, bh = a["bbox"]
                cv2.rectangle(im, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        thumbs.append(cv2.resize(im, (320, 320)))
    if not thumbs:
        return None
    cols = min(4, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.full((rows * 320, cols * 320, 3), 255, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * 320 : (r + 1) * 320, c * 320 : (c + 1) * 320] = t
    return grid


def write_qa_report(cfg: Config, class_names: list[str], qa: _QAStats, summary: dict) -> None:
    """Dataset QA report: totals + box/size/aspect/class-balance charts (opencv-drawn,
    no matplotlib) + a montage of composites with their EXPORTED labels drawn on."""
    qadir = cfg.out / "qa"
    qadir.mkdir(parents=True, exist_ok=True)
    charts = []
    if qa.boxes_per_image:
        cv2.imwrite(
            str(qadir / "hist_boxes_per_image.png"),
            _hist_chart(
                qa.boxes_per_image,
                "boxes per image",
                bins=min(20, max(1, max(qa.boxes_per_image) + 1)),
            ),
        )
        charts.append("hist_boxes_per_image.png")
    if qa.sizes:
        cv2.imwrite(
            str(qadir / "hist_seed_size.png"), _hist_chart(qa.sizes, "seed size (long side, px)")
        )
        charts.append("hist_seed_size.png")
    if qa.aspects:
        cv2.imwrite(
            str(qadir / "hist_aspect.png"), _hist_chart(qa.aspects, "box aspect ratio (w/h)")
        )
        charts.append("hist_aspect.png")
    cv2.imwrite(
        str(qadir / "class_balance.png"),
        _bar_chart(
            [qa.class_counts.get(i, 0) for i in range(len(class_names))],
            class_names,
            "class balance (boxes/class)",
        ),
    )
    charts.append("class_balance.png")
    montage = _montage_from_disk(cfg, class_names, cfg.qa_montage)
    if montage is not None:
        cv2.imwrite(str(qadir / "montage.png"), montage)
        charts.append("montage.png")

    lines = [
        "# Dataset QA report",
        "",
        f"- images: **{summary['num_images']}** "
        f"({summary['n_train']} train / {summary['n_val']} val)",
        f"- background-only negatives: **{summary['total_negatives']}**",
        f"- total boxes: **{summary['total_boxes']}** "
        f"(avg **{summary['avg_boxes_per_image']}**/image)",
        f"- classes ({summary['n_classes']}): {class_names}",
        f"- unique source seeds: **{summary['n_seeds']}**",
        f"- wall time: {summary['wall_seconds']}s on {summary['workers']} worker(s)",
        "",
    ]
    if qa.sizes:
        s = np.asarray(qa.sizes)
        lines.append(
            f"- seed long-side px: min {s.min():.0f} / median "
            f"{np.median(s):.0f} / max {s.max():.0f}"
        )
    lines.append(
        "- class balance: "
        + ", ".join(
            f"{class_names[i]}={qa.class_counts.get(i, 0)}" for i in range(len(class_names))
        )
    )
    lines.append("")
    for c in charts:
        lines.append(f"![{c}]({c})")
    (qadir / "report.md").write_text("\n".join(lines) + "\n")
    logger.info(f"  wrote QA report to {qadir}")


# --------------------------------------------------------------------------- #
# Preview (round-trips the EXPORTED labels to verify them)                    #
# --------------------------------------------------------------------------- #
def _checkerboard(h: int, w: int, sq: int = 12) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((xx // sq) + (yy // sq)) % 2
    hi = np.array([205, 90, 205], np.uint8)  # magenta-ish, BGR
    lo = np.array([120, 45, 120], np.uint8)
    return np.where(c[..., None] == 0, hi, lo).astype(np.uint8)


def _label_bar(img: np.ndarray, text: str, color=(255, 255, 255)) -> None:
    """Draw a readable caption on a dark bar across the top of ``img`` (in place)."""
    h, w = img.shape[:2]
    bar = max(14, int(h * 0.16))
    cv2.rectangle(img, (0, 0), (w, bar), (0, 0, 0), -1)
    cv2.putText(img, text, (3, bar - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _fit_thumb(bgr: np.ndarray, long_side: int = 220) -> np.ndarray:
    """Downscale ``bgr`` so its longest side is at most ``long_side`` px (no upscaling)."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return bgr.copy()
    s = long_side / m
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def check_segmentation(
    seeds: list[Seed],
    class_names: list[str],
    cfg: Config,
    n: int,
    seg_results: list | None = None,
) -> None:
    """Composite each cut-out onto a magenta checkerboard so any retained
    background is glaringly obvious, and draw the exact box the generator uses.
    A clean cut-out shows ONLY the seed over the checker pattern.

    Full runs (``cfg.num_images > 0``) keep the original kept-only output, byte for
    byte. In tuner mode (``num_images <= 0``) it additionally stamps each kept cut-out
    with its method+confidence, renders the *skipped* sources (re-read from disk) so a
    user can see what was rejected and why, and writes a ``manifest.json`` listing every
    rendered entry's status/confidence/reason for the web tuner. The manifest is written
    LAST, so its presence means the render completed (a stopped/crashed check has none)."""
    out = cfg.out
    d = out / "seed_check"
    d.mkdir(parents=True, exist_ok=True)
    tuner = cfg.num_images <= 0
    pad = 18
    manifest: list[dict] = []

    def _cls(cid: int) -> str:
        return class_names[cid] if 0 <= cid < len(class_names) else str(cid)

    for i, s in enumerate(seeds[:n]):
        patch = s.rgba
        ph, pw = patch.shape[:2]
        H, W = ph + 2 * pad, pw + 2 * pad
        canvas = _checkerboard(H, W).astype(np.float32)
        a = patch[..., 3:4].astype(np.float32) / 255.0
        region = canvas[pad : pad + ph, pad : pad + pw]
        canvas[pad : pad + ph, pad : pad + pw] = region * (1 - a) + patch[..., :3] * a
        img = canvas.astype(np.uint8)
        ys, xs = np.where(patch[..., 3] >= 128)
        if xs.size:
            cv2.rectangle(
                img,
                (pad + int(xs.min()), pad + int(ys.min())),
                (pad + int(xs.max()), pad + int(ys.max())),
                (0, 0, 255),
                1,
            )
        if tuner:
            fname = f"seed_kept_{i:04d}.png"
            _label_bar(img, f"{s.method or '?'}  {s.confidence:.2f}")
            cv2.imwrite(str(d / fname), img)
            manifest.append(
                {
                    "file": fname,
                    "source": s.source,
                    "status": "kept",
                    "confidence": round(float(s.confidence), 3),
                    "method": s.method,
                    "reason": "",
                }
            )
        else:  # full-run mode: original filename + no overlay -> output unchanged
            cv2.imwrite(str(d / f"seed_{i:03d}_{_cls(s.class_id)}.png"), img)

    if tuner and seg_results:
        skipped = [r for r in seg_results if r.report.get("skipped")]
        for j, r in enumerate(skipped[:n]):
            rep = r.report
            src_rel = rep.get("source") or r.source
            orig = cv2.imread(str(cfg.sources / src_rel), cv2.IMREAD_COLOR)
            if orig is None:  # moved/unreadable original -> skip its thumbnail
                continue
            reason = rep.get("skip_reason", "?")
            conf = float(rep.get("confidence", 0.0))
            thumb = _fit_thumb(orig, 220)
            _label_bar(thumb, f"SKIP {reason}  {conf:.2f}", color=(80, 80, 255))
            fname = f"seed_skip_{j:04d}.png"
            cv2.imwrite(str(d / fname), thumb)
            manifest.append(
                {
                    "file": fname,
                    "source": src_rel,
                    "status": "skipped",
                    "confidence": round(conf, 3),
                    "method": rep.get("method", "-"),
                    "reason": reason,
                }
            )

    if tuner:  # write LAST: presence == complete render (stop/crash leaves none)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        f"  wrote {len(manifest) if tuner else min(n, len(seeds))} cut-out check "
        f"image(s) to {d}\n"
        f"  -> open them: a clean cut-out shows ONLY the seed on the checker "
        f"pattern. Any white/coloured patch around it means segmentation kept "
        f"background; try --segment otsu|grabcut|rembg or adjust --bg-tolerance."
    )


def make_previews(cfg: Config, class_names: list[str], k: int) -> None:
    split = "train"
    img_dir = cfg.out / split / "images"
    prev_dir = cfg.out / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(img_dir.glob(f"*.{cfg.img_ext}"))[:k]

    coco = None
    if cfg.fmt in ("coco", "both"):
        data = json.loads((cfg.out / f"annotations_{split}.json").read_text())
        by_name = {im["file_name"].split("/")[-1]: im["id"] for im in data["images"]}
        anns: dict = {}
        for a in data["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        coco = (by_name, anns)

    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:  # matches _montage_from_disk: skip an unreadable preview image
            continue
        H, W = img.shape[:2]
        drawn = img.copy()
        if cfg.fmt in ("yolo", "both", "yolo-seg"):
            lp = cfg.out / split / "labels" / (ip.stem + ".txt")
            for line in lp.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                c = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                if len(vals) == 4:  # box line
                    cx, cy, bw, bh = (vals[0] * W, vals[1] * H, vals[2] * W, vals[3] * H)
                    x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
                    x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
                    cv2.rectangle(drawn, (x0, y0), (x1, y1), (0, 0, 255), 2)
                else:  # polygon (yolo-seg)
                    pts = np.array(vals, np.float32).reshape(-1, 2)
                    pts[:, 0] *= W
                    pts[:, 1] *= H
                    cv2.polylines(drawn, [pts.astype(np.int32)], True, (0, 0, 255), 2)
                    x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
                cv2.putText(
                    drawn,
                    class_names[c],
                    (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
        elif coco is not None:
            by_name, anns = coco
            for a in anns.get(by_name.get(ip.name, -1), []):
                x, y, w, h = a["bbox"]
                cv2.rectangle(drawn, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(str(prev_dir / ip.name), drawn)
    logger.info(f"  wrote {len(imgs)} annotated preview(s) to {prev_dir}")
