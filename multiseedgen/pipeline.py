from __future__ import annotations
"""Deterministic per-scene workers and the run() driver."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import json
import numpy as np
import os
import pickle
import sys
import time

from .compositing import generate_scene
from .constants import IMG_EXTS
from .degrade import _jpeg_quality
from .export import (write_coco, write_data_yaml, write_multilabel_csv,
                     write_run_manifest, write_yolo_label, write_yolo_seg_label)
from .models import Config, _QAStats
from .optdeps import Parallel, _HAVE_JOBLIB, _tqdm, delayed
from .qa import check_segmentation, make_previews, write_qa_report
from .segmentation import load_seeds, write_seg_report




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



_ASSET_CACHE: Dict[str, tuple] = {}



def _load_assets(assets_path: str) -> tuple:
    """Load (seeds, textures) once per worker process and memoize."""
    cached = _ASSET_CACHE.get(assets_path)
    if cached is None:
        with open(assets_path, "rb") as f:
            cached = pickle.load(f)
        _ASSET_CACHE[assets_path] = cached
    return cached



def _available_ram_bytes() -> Optional[int]:
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

    'loky' (separate processes) is fastest but each worker holds its OWN full copy of
    the seed+texture assets; for a large asset set on a small machine that multiplies
    into an out-of-memory kill. 'threading' shares the parent's single in-memory copy
    (cv2/numpy release the GIL, so it still uses multiple cores) at ~1x memory.

    'auto' picks loky only when (workers + 1) asset copies fit in ~60% of available
    RAM, otherwise threading."""
    choice = getattr(cfg, "parallel_backend", "auto")
    if not _HAVE_JOBLIB or workers <= 1:
        return "threading"                          # parallel disabled -> serial loop
    if choice in ("loky", "threading"):
        return choice
    asset_bytes = (sum(int(s.rgba.nbytes) for s in seeds)
                   + sum(int(t.nbytes) for t in textures))
    avail = _available_ram_bytes()
    need = (workers + 1) * asset_bytes
    if avail and need > 0.60 * avail:
        print(f"  [mem] assets ~{asset_bytes/1e9:.1f} GB/worker; {workers} loky "
              f"processes would need ~{need/1e9:.0f} GB > 60% of ~{avail/1e9:.0f} GB "
              f"free -> using threaded workers (shared memory, ~{asset_bytes/1e9:.1f} "
              f"GB total)")
        return "threading"
    return "loky"



def _render_one(cfg: Config, seeds: list, textures: list,
                split: str, idx: int, seed_seq):
    """Pure per-scene worker over SHARED, read-only assets: (split, idx, seed_seq) ->
    writes the image to disk and returns only the (small) labels/metadata. Writing the
    image in the worker avoids shipping a ~1MB array back per scene. Deterministic in
    its inputs; no shared mutable state, no global RNG."""
    cv2.setNumThreads(1)                            # avoid thread oversubscription
    rng = np.random.default_rng(seed_seq)
    is_neg = rng.random() < cfg.neg_frac            # FIRST draw (kept first so the
    img, labels, polys, prov = generate_scene(      # stream is stable)
        cfg, seeds, textures, rng, negative=is_neg)
    quality = _jpeg_quality(cfg, rng)
    H, W = img.shape[:2]
    name = f"{split}_{idx:06d}.{cfg.img_ext}"
    img_path = cfg.out / split / "images" / name
    if cfg.img_ext.lower() in ("jpg", "jpeg"):
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        cv2.imwrite(str(img_path), img)
    return idx, name, W, H, labels, polys, is_neg, prov



def _render_one_disk(cfg: Config, assets_path: str, split: str, idx: int, seed_seq):
    """loky entry point: load the disk-staged assets once per worker process (memoized
    in _ASSET_CACHE) and delegate to _render_one with the split's seed pool. The
    threading/serial paths skip this and pass the parent's shared in-memory assets."""
    pools, textures = _load_assets(assets_path)
    return _render_one(cfg, pools[split], textures, split, idx, seed_seq)



def _write_result(cfg: Config, split: str, result, coco_store: dict) -> Tuple[int, int]:
    """Write one scene's labels (parent process). The image was already written by
    the worker; here we only persist labels and stash the COCO record by idx."""
    idx, name, W, H, labels, polys, is_neg, prov = result
    if cfg.fmt in ("yolo", "both"):
        write_yolo_label(cfg.out / split / "labels" / f"{split}_{idx:06d}.txt",
                         labels, W, H)
    elif cfg.fmt == "yolo-seg":
        write_yolo_seg_label(cfg.out / split / "labels" / f"{split}_{idx:06d}.txt",
                             labels, polys, W, H)
    if cfg.fmt in ("coco", "both"):
        coco_store[idx] = (f"{split}/images/{name}", W, H, labels, polys, prov)
    return len(labels), int(is_neg)



# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> None:
    t_start = time.time()
    print(f"[1/3] loading + segmenting sources from {cfg.sources}")
    seeds, class_names, source_bgs, seg_results = load_seeds(cfg)
    write_seg_report(cfg.out, seg_results, cfg)

    if cfg.check_seeds > 0:
        check_segmentation(seeds, class_names, cfg.out, cfg.check_seeds)
    if cfg.num_images <= 0:
        print("num-images is 0: segmentation check only, no dataset generated.")
        return

    if "seedbed" in cfg.backgrounds:
        print("  [warn] 'seedbed' background pastes blurred seed-shaped clutter "
              "that is NOT labelled; this can teach the detector to suppress "
              "real seeds. Prefer real photos via --textures.", file=sys.stderr)

    textures: List[np.ndarray] = list(source_bgs)
    if cfg.textures:
        for f in sorted(cfg.textures.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                t = cv2.imread(str(f))
                if t is not None:
                    textures.append(t)
        print(f"  loaded {len(textures) - len(source_bgs)} texture background(s)")

    n_val = int(round(cfg.num_images * cfg.val_split))
    n_train = cfg.num_images - n_val
    splits = {"train": n_train, "val": n_val}

    # seed-level pools (data-leakage hold-out). With no hold-out every seed is "both",
    # so both pools are the full set in the original order -> byte-identical output.
    # Otherwise train and val draw from DISJOINT source seeds.
    pools = {
        "train": [s for s in seeds if s.split in ("train", "both")],
        "val":   [s for s in seeds if s.split in ("val", "both")],
    }
    for sp, cnt in splits.items():
        if cnt > 0 and not pools[sp]:
            sys.exit(f"no seeds available for the '{sp}' split after hold-out "
                     f"(val_seed_holdout={cfg.val_seed_holdout}, val_seeds={cfg.val_seeds})"
                     f"; lower the hold-out or add more source seeds")

    want_labels = cfg.fmt in ("yolo", "both", "yolo-seg")
    for split in splits:
        (cfg.out / split / "images").mkdir(parents=True, exist_ok=True)
        if want_labels:
            (cfg.out / split / "labels").mkdir(parents=True, exist_ok=True)

    workers = cfg.workers if cfg.workers > 0 else min(os.cpu_count() or 1, 8)
    backend = _resolve_backend(cfg, seeds, textures, workers)

    # Only the loky (process) backend needs the assets staged to disk: each worker
    # process loads its OWN copy once (memoized in _ASSET_CACHE) instead of re-pickling
    # per task. The threading/serial paths share the parent's single in-memory copy, so
    # they need neither the temp file nor any per-process duplication.
    assets_path = None
    if backend == "loky":
        assets_path = str(cfg.out / ".assets.pkl")
        with open(assets_path, "wb") as f:
            pickle.dump((pools, textures), f, protocol=pickle.HIGHEST_PROTOCOL)
        _ASSET_CACHE[assets_path] = (pools, textures)

    parallel = _HAVE_JOBLIB and workers > 1
    desc = f"{workers} {backend} worker(s)" if parallel else "serial"
    print(f"[2/3] generating {cfg.num_images} scenes "
          f"({n_train} train / {n_val} val), {desc}")
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
                tasks = (delayed(_render_one_disk)(cfg, assets_path, split, i, seed_seqs[i])
                         for i in range(count))
            else:
                cv2.setNumThreads(1)                # 1 cv2 thread/scene; joblib threads parallelize
                tasks = (delayed(_render_one)(cfg, pools[split], textures, split, i, seed_seqs[i])
                         for i in range(count))
            gen = Parallel(n_jobs=workers, backend=backend, return_as="generator",
                           batch_size=batch)(tasks)
        else:
            gen = (_render_one(cfg, pools[split], textures, split, i, seed_seqs[i])
                   for i in range(count))
        for result in _tqdm(gen, total=count, desc=split, unit="img"):
            nb, neg = _write_result(cfg, split, result, coco_store)
            total_boxes += nb
            total_neg += neg
            if qa is not None:
                qa.add(split, result)
            if cfg.provenance:
                prov_store[result[0]] = (result[1], result[7])
            if ml_rows is not None:                 # keyed by idx -> deterministic order
                ml_store[result[0]] = (result[1], [int(l[0]) for l in result[4]])
        if cfg.fmt in ("coco", "both"):
            coco_records[split] = [coco_store[i] for i in sorted(coco_store)]
        if ml_rows is not None:
            ml_rows.extend((f"{split}/images/{ml_store[i][0]}", split, ml_store[i][1])
                           for i in sorted(ml_store))
        if cfg.provenance:                          # write in deterministic idx order
            lines = []
            for i in sorted(prov_store):
                name, prov = prov_store[i]
                lines.append(json.dumps({"image": f"{split}/images/{name}",
                                         "boxes": prov or []}))
            (cfg.out / f"provenance_{split}.jsonl").write_text("\n".join(lines) +
                                                               ("\n" if lines else ""))

    print("[3/3] writing annotations")
    if want_labels:
        write_data_yaml(cfg.out, class_names)
    if cfg.fmt in ("coco", "both"):
        for split in splits:
            write_coco(cfg.out / f"annotations_{split}.json",
                       coco_records[split], class_names,
                       extra_annotations=cfg.extra_annotations)
    if ml_rows is not None:
        write_multilabel_csv(cfg.out / "multilabel.csv", ml_rows, class_names)
        print(f"  wrote multilabel.csv ({len(ml_rows)} rows, {len(class_names)} classes)")

    if assets_path:
        try:
            os.remove(assets_path)
        except OSError:
            pass

    avg = total_boxes / max(1, cfg.num_images)
    summary = {
        "num_images": cfg.num_images, "n_train": n_train, "n_val": n_val,
        "total_boxes": total_boxes, "total_negatives": total_neg,
        "avg_boxes_per_image": round(avg, 3), "n_seeds": len(seeds),
        "n_classes": len(class_names), "class_names": class_names,
        "n_textures": len(textures), "workers": workers, "seed": cfg.seed,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    write_run_manifest(cfg, class_names, summary)
    if cfg.visualize_count > 0:
        make_previews(cfg, class_names, cfg.visualize_count)
    if qa is not None:
        write_qa_report(cfg, class_names, qa, summary)

    print(f"\nDone. {cfg.num_images} images ({total_neg} background-only), "
          f"{total_boxes} boxes ({avg:.1f} per image) in {summary['wall_seconds']}s "
          f"-> {cfg.out.resolve()}")
