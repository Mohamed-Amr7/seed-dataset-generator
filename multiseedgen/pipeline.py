"""Deterministic per-scene workers and the run() driver."""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from .compositing import generate_scene
from .constants import IMG_EXTS
from .degrade import _jpeg_quality
from .errors import SplitError
from .export import (
    write_coco,
    write_data_yaml,
    write_multilabel_csv,
    write_run_manifest,
    write_yolo_label,
    write_yolo_seg_label,
)
from .models import Config, Seed, _QAStats
from .optdeps import _HAVE_JOBLIB, Parallel, _tqdm, delayed
from .qa import check_segmentation, make_previews, write_qa_report
from .segmentation import load_seeds, write_seg_report

logger = logging.getLogger(__name__)


class SceneResult(NamedTuple):
    """One rendered scene's metadata returned by a worker (the image is already on
    disk). Replaces the former positional 8-tuple so consumers read named fields."""

    idx: int
    name: str
    width: int
    height: int
    labels: list
    polys: list | None
    is_neg: bool
    prov: list | None


# --------------------------------------------------------------------------- #
# Scene rendering workers (deterministic per scene; parallel-safe)            #
# --------------------------------------------------------------------------- #
def _scene_seeds(master_seed: int, split: str, count: int) -> list:
    """A stable, independent RNG stream per (split, scene index). Per-split
    namespacing (spawn_key) means changing --val-split never renumbers the other
    split's streams, and SeedSequence.spawn is prefix-stable, so scene i is
    reproducible regardless of count, worker scheduling, or degree of parallelism."""
    split_tag = {"train": 0, "val": 1}.get(split, 2)
    parent = np.random.SeedSequence(entropy=int(master_seed), spawn_key=(split_tag,))
    return list(parent.spawn(count))


_ASSET_CACHE: dict[str, tuple] = {}


def _stage_assets_mmap(stage_dir: Path, manifest_path: Path, seeds: list, textures: list) -> None:
    """Stage seeds + textures as individual raw .npy files plus a JSON manifest so
    every loky worker can memory-map the SAME files read-only (one shared physical
    copy via the OS page cache) instead of unpickling its own full duplicate. The
    seed order in the manifest mirrors the parent's `seeds` list exactly, so the
    pools reconstructed in each worker are byte- and order-identical -> deterministic."""
    shutil.rmtree(stage_dir, ignore_errors=True)  # clear any stale staging dir
    manifest_path.unlink(missing_ok=True)  # ...and any stale manifest from a hard kill
    stage_dir.mkdir(parents=True, exist_ok=True)
    seed_entries = []
    for i, s in enumerate(seeds):
        fn = f"seed_{i}.npy"
        np.save(stage_dir / fn, np.ascontiguousarray(s.rgba))
        seed_entries.append(
            {
                "file": fn,
                "class_id": int(s.class_id),
                "source": s.source,
                "method": s.method,
                "split": s.split,
            }
        )
    tex_entries = []
    for j, t in enumerate(textures):
        fn = f"tex_{j}.npy"
        np.save(stage_dir / fn, np.ascontiguousarray(t))
        tex_entries.append(fn)
    manifest_path.write_text(
        json.dumps({"dir": str(stage_dir), "seeds": seed_entries, "textures": tex_entries})
    )


def _load_assets(assets_path: str) -> tuple:
    """Reconstruct (pools, textures) from the staged manifest once per worker process
    and memoize. Each `rgba`/texture is opened with mmap_mode='r', so all workers map
    the same physical pages (~1x RAM total, independent of worker count). The pools are
    rebuilt with the SAME split filter the parent uses, preserving seed order."""
    cached = _ASSET_CACHE.get(assets_path)
    if cached is not None:
        return cached
    manifest = json.loads(Path(assets_path).read_text())
    stage_dir = Path(manifest["dir"])
    seeds = [
        Seed(
            rgba=np.load(stage_dir / e["file"], mmap_mode="r"),
            class_id=e["class_id"],
            source=e["source"],
            method=e["method"],
            split=e["split"],
        )
        for e in manifest["seeds"]
    ]
    pools = {
        "train": [s for s in seeds if s.split in ("train", "both")],
        "val": [s for s in seeds if s.split in ("val", "both")],
    }
    textures = [np.load(stage_dir / fn, mmap_mode="r") for fn in manifest["textures"]]
    cached = (pools, textures)
    _ASSET_CACHE[assets_path] = cached
    return cached


def _cleanup_staging(assets_path: str | None, stage_dir: Path | None) -> None:
    """Remove the loky staging manifest + .npy dir and drop the memo. Called from a
    finally so a crash / Ctrl-C / exception mid-generation can't orphan the multi-GB
    staging dir; must never raise (it runs while an exception may be propagating)."""
    if not assets_path:
        return
    try:
        os.remove(assets_path)
    except OSError:
        pass
    if stage_dir is not None:
        shutil.rmtree(stage_dir, ignore_errors=True)  # drop the staged .npy files
    _ASSET_CACHE.pop(assets_path, None)
    gc.collect()


def _available_ram_bytes() -> int | None:
    """Best-effort available RAM in bytes (Linux /proc/meminfo); None if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _resolve_backend(cfg: Config, seeds: list, textures: list, workers: int) -> str:
    """Choose the scene-generation parallel backend.

    'loky' (separate processes) is fastest. Its workers memory-map the staged assets
    read-only, so all processes share ONE physical copy via the OS page cache (~1x
    RAM, independent of worker count) instead of each unpickling a full duplicate.
    'threading' shares the parent's single in-memory copy (cv2/numpy release the GIL,
    so it still uses multiple cores) at ~1x memory.

    'auto' picks loky when the single shared asset copy fits in ~60% of available RAM,
    otherwise threading (which avoids the extra on-disk staging at the same ~1x cost)."""
    choice = getattr(cfg, "parallel_backend", "auto")
    if not _HAVE_JOBLIB or workers <= 1:
        return "threading"  # parallel disabled -> serial loop
    if choice in ("loky", "threading"):
        return choice
    asset_bytes = sum(int(s.rgba.nbytes) for s in seeds) + sum(int(t.nbytes) for t in textures)
    avail = _available_ram_bytes()
    need = asset_bytes  # memmap-shared: ~1x, not (workers+1)x
    if avail and need > 0.60 * avail:
        logger.info(
            f"  [mem] assets ~{asset_bytes / 1e9:.1f} GB > 60% of ~{avail / 1e9:.0f} GB "
            f"free even shared once -> using threaded workers (no on-disk staging); "
            f"consider --max-seed-px / --max-sources to shrink the seed pool"
        )
        return "threading"
    return "loky"


def _render_one(
    cfg: Config, seeds: list, textures: list, split: str, idx: int, seed_seq
) -> SceneResult:
    """Pure per-scene worker over SHARED, read-only assets: (split, idx, seed_seq) ->
    writes the image to disk and returns only the (small) labels/metadata. Writing the
    image in the worker avoids shipping a ~1MB array back per scene. Deterministic in
    its inputs; no shared mutable state, no global RNG."""
    cv2.setNumThreads(1)  # avoid thread oversubscription
    rng = np.random.default_rng(seed_seq)
    is_neg = rng.random() < cfg.neg_frac  # FIRST draw (kept first so the
    img, labels, polys, prov = generate_scene(  # stream is stable)
        cfg, seeds, textures, rng, negative=is_neg
    )
    quality = _jpeg_quality(cfg, rng)
    H, W = img.shape[:2]
    name = f"{split}_{idx:06d}.{cfg.img_ext}"
    img_path = cfg.out / split / "images" / name
    if cfg.img_ext.lower() in ("jpg", "jpeg"):
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        cv2.imwrite(str(img_path), img)
    return SceneResult(idx, name, W, H, labels, polys, is_neg, prov)


def _render_one_disk(cfg: Config, assets_path: str, split: str, idx: int, seed_seq) -> SceneResult:
    """loky entry point: load the disk-staged assets once per worker process (memoized
    in _ASSET_CACHE) and delegate to _render_one with the split's seed pool. The
    threading/serial paths skip this and pass the parent's shared in-memory assets."""
    pools, textures = _load_assets(assets_path)
    return _render_one(cfg, pools[split], textures, split, idx, seed_seq)


def _write_result(
    cfg: Config, split: str, result: SceneResult, coco_store: dict
) -> tuple[int, int]:
    """Write one scene's labels (parent process). The image was already written by
    the worker; here we only persist labels and stash the COCO record by idx."""
    idx, name, W, H, labels, polys, is_neg, prov = result
    if cfg.fmt in ("yolo", "both"):
        write_yolo_label(cfg.out / split / "labels" / f"{split}_{idx:06d}.txt", labels, W, H)
    elif cfg.fmt == "yolo-seg":
        write_yolo_seg_label(
            cfg.out / split / "labels" / f"{split}_{idx:06d}.txt", labels, polys, W, H
        )
    if cfg.fmt in ("coco", "both"):
        coco_store[idx] = (f"{split}/images/{name}", W, H, labels, polys, prov)
    return len(labels), int(is_neg)


# --------------------------------------------------------------------------- #
# Driver helpers                                                              #
# --------------------------------------------------------------------------- #
def _load_textures(cfg: Config, source_bgs: list) -> list[np.ndarray]:
    """Texture backgrounds: domain backgrounds extracted from sources, plus any real
    photos under cfg.textures (read in sorted order, so the texture pool is stable)."""
    textures: list[np.ndarray] = list(source_bgs)
    if cfg.textures:
        for f in sorted(cfg.textures.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                t = cv2.imread(str(f))
                if t is not None:
                    textures.append(t)
        logger.info(f"  loaded {len(textures) - len(source_bgs)} texture background(s)")
    return textures


def _build_split_pools(cfg: Config, seeds: list, splits: dict[str, int]) -> dict[str, list]:
    """Seed-level pools for the data-leakage hold-out. With no hold-out every seed is
    "both", so each pool is the full set in the original order -> byte-identical output;
    otherwise train and val draw from DISJOINT source seeds. Raises if a requested split
    has no seeds left after the hold-out."""
    pools = {
        "train": [s for s in seeds if s.split in ("train", "both")],
        "val": [s for s in seeds if s.split in ("val", "both")],
    }
    for sp, cnt in splits.items():
        if cnt > 0 and not pools[sp]:
            raise SplitError(
                f"no seeds available for the '{sp}' split after hold-out "
                f"(val_seed_holdout={cfg.val_seed_holdout}, val_seeds={cfg.val_seeds})"
                f"; lower the hold-out or add more source seeds"
            )
    return pools


def _write_annotations(
    cfg: Config, want_labels: bool, class_names: list, coco_records: dict, ml_rows: list | None
) -> None:
    """Write the dataset-level annotation artifacts (data.yaml, COCO jsons, multilabel
    CSV) after every scene has been generated. Frees each COCO split once written."""
    logger.info("[3/3] writing annotations")
    if want_labels:
        write_data_yaml(cfg.out, class_names)
    if cfg.fmt in ("coco", "both"):
        for split in coco_records:
            write_coco(
                cfg.out / f"annotations_{split}.json",
                coco_records[split],
                class_names,
                extra_annotations=cfg.extra_annotations,
            )
            coco_records[split] = []  # free each split's records once written
        gc.collect()
    if ml_rows is not None:
        write_multilabel_csv(cfg.out / "multilabel.csv", ml_rows, class_names)
        logger.info(f"  wrote multilabel.csv ({len(ml_rows)} rows, {len(class_names)} classes)")


def _build_summary(
    cfg: Config,
    class_names: list,
    n_train: int,
    n_val: int,
    total_boxes: int,
    total_neg: int,
    n_seeds: int,
    n_textures: int,
    workers: int,
    t_start: float,
) -> dict:
    """The run-summary dict embedded in run_config + the QA report."""
    avg = total_boxes / max(1, cfg.num_images)
    return {
        "num_images": cfg.num_images,
        "n_train": n_train,
        "n_val": n_val,
        "total_boxes": total_boxes,
        "total_negatives": total_neg,
        "avg_boxes_per_image": round(avg, 3),
        "n_seeds": n_seeds,
        "n_classes": len(class_names),
        "class_names": class_names,
        "n_textures": n_textures,
        "workers": workers,
        "seed": cfg.seed,
        "wall_seconds": round(time.time() - t_start, 1),
    }


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> None:
    t_start = time.time()
    logger.info(f"[1/3] loading + segmenting sources from {cfg.sources}")
    seeds, class_names, source_bgs, seg_results = load_seeds(cfg)
    write_seg_report(cfg.out, seg_results, cfg)
    del seg_results  # the SegResult container is unused past the report
    gc.collect()  # reclaim freed patch/background buffers before render

    if cfg.check_seeds > 0:
        check_segmentation(seeds, class_names, cfg.out, cfg.check_seeds)
    if cfg.num_images <= 0:
        logger.info("num-images is 0: segmentation check only, no dataset generated.")
        return

    if "seedbed" in cfg.backgrounds:
        logger.warning(
            "  [warn] 'seedbed' background pastes blurred seed-shaped clutter "
            "that is NOT labelled; this can teach the detector to suppress "
            "real seeds. Prefer real photos via --textures."
        )

    textures = _load_textures(cfg, source_bgs)

    n_val = int(round(cfg.num_images * cfg.val_split))
    n_train = cfg.num_images - n_val
    splits = {"train": n_train, "val": n_val}

    pools = _build_split_pools(cfg, seeds, splits)

    want_labels = cfg.fmt in ("yolo", "both", "yolo-seg")
    for split in splits:
        (cfg.out / split / "images").mkdir(parents=True, exist_ok=True)
        if want_labels:
            (cfg.out / split / "labels").mkdir(parents=True, exist_ok=True)

    workers = cfg.workers if cfg.workers > 0 else min(os.cpu_count() or 1, 8)
    backend = _resolve_backend(cfg, seeds, textures, workers)
    n_seeds, n_textures = len(seeds), len(textures)  # kept for the summary after drop

    # Only the loky (process) backend stages assets to disk: each worker memory-maps
    # the SAME .npy files read-only, so all processes share one physical copy via the
    # OS page cache instead of unpickling a full per-process duplicate. Once staged,
    # the parent drops its own in-RAM seeds/pools/textures (workers read from disk), so
    # the heavy generation phase sits at ~1x asset RAM regardless of worker count. The
    # threading/serial paths keep the parent's single shared in-memory copy.
    assets_path = None
    stage_dir = None
    if backend == "loky":
        stage_dir = cfg.out / ".assets_mmap"
        assets_path = str(cfg.out / ".assets_manifest.json")

    # Everything that touches the on-disk staging runs inside try/finally so a crash,
    # Ctrl-C, or exception mid-generation can't orphan the multi-GB .assets_mmap dir
    # (the start-of-run rmtree only self-heals a *later* loky run, not the current one).
    try:
        if backend == "loky":
            _stage_assets_mmap(stage_dir, Path(assets_path), seeds, textures)
            pools = None
            seeds = None
            textures = None
            gc.collect()

        parallel = _HAVE_JOBLIB and workers > 1
        desc = f"{workers} {backend} worker(s)" if parallel else "serial"
        logger.info(
            f"[2/3] generating {cfg.num_images} scenes ({n_train} train / {n_val} val), {desc}"
        )
        coco_records = {"train": [], "val": []}
        qa = _QAStats() if cfg.qa_report else None
        ml_rows: list = [] if cfg.multilabel_csv else None  # per-image multi-label rows
        total_boxes = 0
        total_neg = 0
        for split, count in splits.items():
            if count <= 0:
                continue
            seed_seqs = _scene_seeds(cfg.seed, split, count)
            coco_store: dict = {}
            prov_store: dict = {}
            ml_store: dict = {}
            if parallel and count > 1:
                # batch several scenes per dispatch to amortize dispatch overhead
                batch = max(1, min(64, count // (workers * 4) or 1))
                if backend == "loky":
                    tasks = (
                        delayed(_render_one_disk)(cfg, assets_path, split, i, seed_seqs[i])
                        for i in range(count)
                    )
                else:
                    cv2.setNumThreads(1)  # 1 cv2 thread/scene; joblib threads parallelize
                    tasks = (
                        delayed(_render_one)(cfg, pools[split], textures, split, i, seed_seqs[i])
                        for i in range(count)
                    )
                gen = Parallel(
                    n_jobs=workers, backend=backend, return_as="generator", batch_size=batch
                )(tasks)
            else:
                # serial path (parallel disabled or count == 1). When loky staged the
                # assets the parent dropped its in-RAM pools, so map them back from disk
                # (memmap, ~0 RAM); otherwise use the parent's shared in-memory copy.
                s_pools, s_textures = (
                    _load_assets(assets_path) if backend == "loky" else (pools, textures)
                )
                gen = (
                    _render_one(cfg, s_pools[split], s_textures, split, i, seed_seqs[i])
                    for i in range(count)
                )
            for result in _tqdm(gen, total=count, desc=split, unit="img"):
                nb, neg = _write_result(cfg, split, result, coco_store)
                total_boxes += nb
                total_neg += neg
                if qa is not None:
                    qa.add(split, result)
                if cfg.provenance:
                    prov_store[result.idx] = (result.name, result.prov)
                if ml_rows is not None:  # keyed by idx -> deterministic order
                    ml_store[result.idx] = (result.name, [int(lbl[0]) for lbl in result.labels])
            if cfg.fmt in ("coco", "both"):
                coco_records[split] = [coco_store[i] for i in sorted(coco_store)]
            if ml_rows is not None:
                ml_rows.extend(
                    (f"{split}/images/{ml_store[i][0]}", split, ml_store[i][1])
                    for i in sorted(ml_store)
                )
            if cfg.provenance:  # write in deterministic idx order
                lines = []
                for i in sorted(prov_store):
                    name, prov = prov_store[i]
                    lines.append(
                        json.dumps({"image": f"{split}/images/{name}", "boxes": prov or []})
                    )
                (cfg.out / f"provenance_{split}.jsonl").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
            del coco_store, prov_store, ml_store  # per-split metadata; freed before next split
            gc.collect()

        _write_annotations(cfg, want_labels, class_names, coco_records, ml_rows)
    finally:
        _cleanup_staging(assets_path, stage_dir)

    summary = _build_summary(
        cfg,
        class_names,
        n_train,
        n_val,
        total_boxes,
        total_neg,
        n_seeds,
        n_textures,
        workers,
        t_start,
    )
    avg = summary["avg_boxes_per_image"]
    write_run_manifest(cfg, class_names, summary)
    if cfg.visualize_count > 0:
        make_previews(cfg, class_names, cfg.visualize_count)
    if qa is not None:
        write_qa_report(cfg, class_names, qa, summary)

    logger.info(
        f"\nDone. {cfg.num_images} images ({total_neg} background-only), "
        f"{total_boxes} boxes ({avg:.1f} per image) in {summary['wall_seconds']}s "
        f"-> {cfg.out.resolve()}"
    )
