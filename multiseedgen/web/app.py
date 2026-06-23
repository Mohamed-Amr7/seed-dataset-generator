"""FastAPI backend for MultiSeedGen.

A thin layer over the verified CLI/pipeline:
  * /api/schema        - introspects build_parser() + Config -> drives the form
  * /api/runs (+ WS)   - launches `python -m multiseedgen --config <json>` as a
                         subprocess and streams its stdout/stderr live
  * /api/seg/check     - runs --check-seeds --num-images 0 and returns the
                         checkerboard cut-outs + seg_report.json (the tuner)
  * /api/datasets/*    - browses generated datasets, drawing boxes/polygons read
                         back from the EXPORTED labels on disk (reuses the same
                         parsing as qa.make_previews)
  * /api/config/*      - load/save run_config-style JSON/YAML (round-trips --config)

Heavy work runs in a subprocess so loky workers stay off the event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections import deque
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from ..cliconfig import _load_config_file
from ..errors import ConfigError
from ..models import Config
from ..optdeps import yaml
from .schemas import ConfigSaveRequest, SegCheckRequest, StartRunRequest, UploadResponse
from .services import _jsonable, build_schema

ROOT = Path(__file__).resolve().parents[2]  # the MultiSeedGen project dir
STATIC = Path(__file__).resolve().parent / "static"
# DATA_ROOT may be overridden by env; validate it points at a real dir, else fall back
# to the project root so a typo can't silently widen (or break) the browse sandbox.
_DATA_ROOT_ENV = os.environ.get("MULTISEEDGEN_DATA_ROOT")
DATA_ROOT = Path(_DATA_ROOT_ENV).resolve() if _DATA_ROOT_ENV else ROOT
if not DATA_ROOT.is_dir():
    DATA_ROOT = ROOT
TMP = Path(tempfile.gettempdir()).resolve()
ALLOWED_BASES = [ROOT, DATA_ROOT, TMP]

LOG_MAXLINES = 4000  # bounded per-run log ring buffer
RUNS_MAX = 40  # cap retained runs (evict oldest finished)
SEG_CHECK_GALLERY_CAP = 500  # max cut-outs rendered for a "whole dataset" tuner check
_DATASET_SCAN_DEPTH = 6  # max directory depth when discovering datasets (bounds the walk)
_UPLOAD_EXTS = {".png", ".jpg", ".jpeg"}  # seed-upload allowlist (also content-sniffed)
_UPLOAD_MAX_BYTES = 25 * 1024 * 1024  # 25 MB per uploaded seed image

app = FastAPI(title="MultiSeedGen")
RUNS: dict = {}  # run_id -> state dict


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _is_allowed(p: Path) -> bool:
    """True if a resolved path lives at or under an allowed base."""
    return any(p == b or b in p.parents for b in ALLOWED_BASES)


def _safe(path: str) -> Path:
    """Resolve a user-supplied path and require it to live under an allowed base."""
    p = Path(path).expanduser().resolve()
    if not _is_allowed(p):
        raise HTTPException(403, f"path outside allowed roots: {path}")
    return p


# --------------------------------------------------------------------------- #
# schema (drives the form)                                                    #
# --------------------------------------------------------------------------- #
@app.get("/api/schema")
def schema() -> dict:
    # Form descriptor is built in services.build_schema(); the route just adds the
    # host-specific roots the picker needs.
    return {**build_schema(), "data_root": str(DATA_ROOT), "project_root": str(ROOT)}


# --------------------------------------------------------------------------- #
# config build / save / load                                                  #
# --------------------------------------------------------------------------- #
def _clean_config(cfg: dict) -> dict:
    """Drop empties so optional path fields fall back to their defaults."""
    field_names = set(Config.model_fields)
    out = {}
    for k, v in cfg.items():
        if k not in field_names:
            continue
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        out[k] = v
    return out


def _write_config_json(cfg: dict) -> Path:
    fd, name = tempfile.mkstemp(suffix=".json", prefix="msg_cfg_")
    os.close(fd)
    Path(name).write_text(json.dumps(cfg, indent=2))
    return Path(name)


@app.get("/api/config/load")
def config_load(path: str = Query(...)) -> dict:
    p = _safe(path)
    try:
        raw = _load_config_file(p)  # validates keys, maps format->fmt
    except ConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {k: _jsonable(v) for k, v in raw.items()}


@app.post("/api/config/save")
async def config_save(req: ConfigSaveRequest) -> dict:
    path = req.path
    cfg = _clean_config(req.config)
    if not path:
        raise HTTPException(400, "missing 'path'")
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    want_yaml = p.suffix.lower() in (".yaml", ".yml")
    if want_yaml and yaml is None:
        raise HTTPException(
            400, "YAML support not installed (pip install pyyaml); save with a .json path instead"
        )
    if want_yaml:
        p.write_text(yaml.safe_dump({"config": cfg}, sort_keys=True))
    else:
        p.write_text(json.dumps({"config": cfg}, indent=2, sort_keys=True))
    return {"saved": str(p), "format": "yaml" if want_yaml else "json"}


# --------------------------------------------------------------------------- #
# runs (subprocess + live log stream)                                         #
# --------------------------------------------------------------------------- #
def _reap_seg_dir(run: dict | None) -> None:
    """Remove a segmentation check's PRIVATE temp output dir when its run is evicted, so
    per-check dirs don't accumulate in TMP. Only dirs we created (``TMP/msg_segcheck_*``)
    are touched - an explicit user-supplied ``out`` is left alone."""
    run = run or {}
    out = run.get("out")
    if not out or run.get("kind") != "segment":
        return
    p = Path(out)
    if p.parent == TMP and p.name.startswith("msg_segcheck_"):
        shutil.rmtree(p, ignore_errors=True)


def _evict_runs() -> None:
    """Bound RUNS: drop the oldest finished runs once over the cap (reaping each evicted
    seg-check's private output dir along the way)."""
    while len(RUNS) > RUNS_MAX:
        for k in list(RUNS):  # insertion order = oldest first
            if RUNS[k]["status"] != "running":
                _reap_seg_dir(RUNS.pop(k, None))
                break
        else:
            break  # nothing finished to evict


async def _pump(rid: str) -> None:
    run = RUNS[rid]
    proc = run["proc"]
    try:
        assert proc.stdout is not None
        while True:
            try:
                line = await proc.stdout.readline()
            except (asyncio.LimitOverrunError, ValueError):
                # a single line exceeded the stream limit (e.g. long tqdm bar);
                # drain what we can and keep going rather than dying
                line = await proc.stdout.read(65536)
                if not line:
                    break
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip("\n")
            run["log"].append(text)
            for q in list(run["subs"]):
                q.put_nowait(text)
        rc = await proc.wait()
        run["returncode"] = rc
        run["status"] = "done" if rc == 0 else "error"
    except Exception as e:  # never leave subscribers hanging
        run["status"] = "error"
        run["log"].append(f"[pump error] {e!r}")
    finally:
        try:
            Path(run["cfg_path"]).unlink(missing_ok=True)  # don't leak temp cfg
        except Exception:
            pass
        for q in list(run["subs"]):
            q.put_nowait(None)  # sentinel: completion (always)


async def _launch(cfg: dict, extra_args: list | None = None, label: str = "run") -> str:
    cfg = _clean_config(cfg)
    if not cfg.get("sources") or not cfg.get("out"):
        raise HTTPException(400, "both 'sources' and 'out' are required")
    cfg_path = _write_config_json(cfg)
    argv = [sys.executable, "-m", "multiseedgen", "--config", str(cfg_path)]
    if extra_args:
        argv += extra_args
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        limit=1 << 20,  # 1MB lines (tqdm/tracebacks)
        start_new_session=True,  # own process group -> killpg
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    rid = f"{label}-{proc.pid}"
    RUNS[rid] = {
        "proc": proc,
        "log": deque(maxlen=LOG_MAXLINES),
        "status": "running",
        "subs": set(),
        "out": cfg.get("out"),
        "cmd": " ".join(argv),
        "returncode": None,
        "cfg_path": str(cfg_path),
        # 'segment' = a tuner check (no browsable dataset); 'generate' = a real run. The UI
        # uses this to hide the "Browse dataset" hand-off for seg checks.
        "kind": "segment" if label == "seg" else "generate",
    }
    _evict_runs()
    asyncio.create_task(_pump(rid))
    return rid


@app.post("/api/runs")
async def start_run(req: StartRunRequest) -> dict:
    rid = await _launch(req.config, label="run")
    return {"run_id": rid, "out": RUNS[rid]["out"]}


@app.get("/api/runs")
def list_runs() -> dict:
    return {
        "runs": [
            {
                "id": k,
                "status": v["status"],
                "out": v["out"],
                "returncode": v["returncode"],
                "kind": v.get("kind", "generate"),
            }
            for k, v in RUNS.items()
        ]
    }


@app.get("/api/runs/{rid}")
def run_status(rid: str) -> dict:
    if rid not in RUNS:
        raise HTTPException(404, "no such run")
    v = RUNS[rid]
    return {
        "id": rid,
        "status": v["status"],
        "returncode": v["returncode"],
        "out": v["out"],
        "kind": v.get("kind", "generate"),
        "log": list(v["log"])[-400:],
    }


@app.post("/api/runs/{rid}/stop")
async def stop_run(rid: str) -> dict:
    if rid not in RUNS:
        raise HTTPException(404, "no such run")
    proc = RUNS[rid]["proc"]
    if proc.returncode is not None:
        return {"stopped": rid, "already_exited": True}
    # kill the whole process group (loky workers are grandchildren), SIGTERM then
    # escalate to SIGKILL if it doesn't exit promptly.
    import signal

    def _signal_group(sig):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    _signal_group(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        _signal_group(signal.SIGKILL)
    return {"stopped": rid}


@app.websocket("/ws/runs/{rid}")
async def ws_run(ws: WebSocket, rid: str):
    await ws.accept()
    if rid not in RUNS:
        await ws.send_json({"type": "error", "msg": "no such run"})
        await ws.close()
        return
    run = RUNS[rid]
    q: asyncio.Queue = asyncio.Queue()
    for line in run["log"]:  # backlog first
        q.put_nowait(line)
    if run["status"] != "running":
        q.put_nowait(None)
    run["subs"].add(q)
    try:
        while True:
            item = await q.get()
            if item is None:
                await ws.send_json(
                    {"type": "done", "returncode": run["returncode"], "status": run["status"]}
                )
                break
            await ws.send_json({"type": "line", "line": item})
    except WebSocketDisconnect:
        pass
    finally:
        run["subs"].discard(q)


# --------------------------------------------------------------------------- #
# segmentation tuner                                                          #
# --------------------------------------------------------------------------- #
def _seg_check_limits(n: int) -> tuple[int, int]:
    """Map the tuner's ``# seeds`` to ``(scan_limit, check_seeds)``.

    ``n >= 1`` samples exactly N sources and renders those N. ``n <= 0`` means the whole
    dataset: ``scan_limit=0`` segments every source (so the report covers the full
    dataset), and the gallery renders up to ``SEG_CHECK_GALLERY_CAP`` cut-outs so a large
    dataset can't flood the browser."""
    if n <= 0:
        return 0, SEG_CHECK_GALLERY_CAP
    return n, n


@app.post("/api/seg/check")
async def seg_check(req: SegCheckRequest) -> dict:
    """Launch a tracked, stoppable segmentation check (a tuner run that produces no
    dataset) and return its run id. Progress streams over ``/ws/runs/{run_id}``, it is
    cancelled via ``/api/runs/{run_id}/stop``, and the results (report + per-seed
    manifest) are read afterwards from ``/api/seg/result``.

    ``n >= 1`` samples exactly N sources (``scan_limit``) and renders those N cut-outs —
    report counts and gallery both reflect that sample. ``n <= 0`` means *whole dataset*:
    segment every source (``scan_limit=0``) so the report covers the full dataset, and
    render the gallery up to ``SEG_CHECK_GALLERY_CAP`` cut-outs so the browser stays
    responsive on large datasets (the report still counts all sources).

    Each check renders into its OWN output dir (a fresh temp dir unless ``out`` is given),
    so a prior check's cut-outs can never bleed into this one's gallery; these private dirs
    are reaped when their run record is evicted. The seg CACHE is shared (warm re-checks)."""
    cfg = dict(req.config)
    scan_limit, check_seeds = _seg_check_limits(int(req.n))
    # Isolated per-check output: a shared dir let a prior run's stale cut-outs leak in.
    out = (
        _safe(req.out) if req.out else Path(tempfile.mkdtemp(prefix="msg_segcheck_", dir=str(TMP)))
    )
    cfg.update(
        {"out": str(out), "num_images": 0, "check_seeds": check_seeds, "scan_limit": scan_limit}
    )
    cfg.setdefault("cache_dir", str(TMP / "msg_segcache"))  # warm across re-checks
    rid = await _launch(cfg, label="seg")
    return {"run_id": rid, "out": RUNS[rid]["out"]}


@app.get("/api/seg/result")
def seg_result(out: str = Query(...)) -> dict:
    """Read a finished (or stopped) segmentation check's report + per-seed manifest.

    Tolerates a missing/corrupt report or manifest (older runs, or a check stopped /
    crashed mid-render) by falling back to a bare glob of the rendered cut-outs."""
    d = _safe(out)
    report = None
    rp = d / "seg_report.json"
    if rp.exists():
        try:  # partial/corrupt report -> still return the images
            report = json.loads(rp.read_text())
        except (ValueError, OSError):
            report = None
    seeds: list[dict] = []
    sd = d / "seed_check"
    man = sd / "manifest.json"
    if man.exists():
        try:
            for e in json.loads(man.read_text()):
                p = sd / e["file"]
                seeds.append(
                    {
                        "url": f"/api/file?path={p}",
                        "source": e.get("source"),
                        "status": e.get("status"),
                        "confidence": e.get("confidence"),
                        "method": e.get("method"),
                        "reason": e.get("reason"),
                    }
                )
        except (ValueError, OSError, KeyError, TypeError):
            seeds = []
    if not seeds and sd.exists():  # no/partial manifest -> glob with unknown metadata
        for c in sorted(sd.glob("*.png")):
            seeds.append(
                {
                    "url": f"/api/file?path={c}",
                    "source": None,
                    "status": "kept",
                    "confidence": None,
                    "method": None,
                    "reason": None,
                }
            )
    return {"report": report, "out": str(d), "seeds": seeds}


# --------------------------------------------------------------------------- #
# dataset browser                                                             #
# --------------------------------------------------------------------------- #
def _class_names(dpath: Path) -> list:
    dy = dpath / "data.yaml"
    if dy.exists() and yaml is not None:
        try:
            d = yaml.safe_load(dy.read_text()) or {}
            names = d.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
            if isinstance(names, list):
                return names
        except Exception:
            pass
    for j in dpath.glob("annotations_*.json"):
        try:
            cats = json.loads(j.read_text()).get("categories", [])
            return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
        except Exception:
            pass
    return [str(i) for i in range(100)]


def _is_dataset(p: Path) -> bool:
    return (
        (p / "data.yaml").exists()
        or any(p.glob("annotations_*.json"))
        or (p / "train" / "images").exists()
    )


@app.get("/api/datasets")
def list_datasets(root: str | None = None) -> dict:
    base = _safe(root) if root else DATA_ROOT
    found = [str(base)] if _is_dataset(base) else []
    # Bounded walk: prune below _DATASET_SCAN_DEPTH so a deep/huge tree can't hang the
    # request (the former unbounded rglob("*") was a DoS vector).
    base_depth = len(base.parts)
    for dirpath, dirnames, _files in os.walk(base):
        dirnames.sort()
        if len(Path(dirpath).parts) - base_depth >= _DATASET_SCAN_DEPTH:
            dirnames[:] = []  # stop descending
            continue
        for d in dirnames:
            p = Path(dirpath) / d
            if _is_dataset(p):
                found.append(str(p))
    # dedupe (preserve order), cap
    seen, uniq = set(), []
    for f in found:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return {"root": str(base), "datasets": uniq[:200]}


@app.get("/api/datasets/info")
def dataset_info(path: str = Query(...)) -> dict:
    d = _safe(path)
    info = {"path": str(d), "splits": {}, "classes": _class_names(d), "qa": [], "seg_report": None}
    for split in ("train", "val"):
        imgs = sorted((d / split / "images").glob("*")) if (d / split / "images").exists() else []
        if imgs:
            info["splits"][split] = len(imgs)
    qadir = d / "qa"
    if qadir.exists():
        info["qa"] = [f"/api/file?path={p}" for p in sorted(qadir.glob("*.png"))]
    if (d / "seg_report.json").exists():
        try:
            info["seg_report"] = json.loads((d / "seg_report.json").read_text())
        except (ValueError, OSError):
            info["seg_report"] = None
    if (d / "run_config.yaml").exists():
        info["run_config"] = str(d / "run_config.yaml")
    return info


def _draw_labels(img, label_path: Path, names: list, coco=None, fname: str = "") -> np.ndarray:
    """Reuse qa.make_previews' read-back: draw boxes/polygons from disk labels."""
    H, W = img.shape[:2]
    drawn = img.copy()
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            c = int(parts[0])
            vals = [float(v) for v in parts[1:]]
            label = names[c] if c < len(names) else str(c)
            if len(vals) == 4:
                cx, cy, bw, bh = vals[0] * W, vals[1] * H, vals[2] * W, vals[3] * H
                x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
                x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
                cv2.rectangle(drawn, (x0, y0), (x1, y1), (0, 0, 255), 2)
            else:
                pts = np.array(vals, np.float32).reshape(-1, 2)
                pts[:, 0] *= W
                pts[:, 1] *= H
                cv2.polylines(drawn, [pts.astype(np.int32)], True, (0, 0, 255), 2)
                x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            cv2.putText(
                drawn,
                label,
                (x0, max(12, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    elif coco is not None:
        by_name, anns = coco
        for a in anns.get(by_name.get(fname, -1), []):
            x, y, w, h = [int(v) for v in a["bbox"]]
            cv2.rectangle(drawn, (x, y), (x + w, y + h), (0, 0, 255), 2)
            c = a.get("category_id", 1) - 1
            label = names[c] if 0 <= c < len(names) else str(c)
            cv2.putText(
                drawn,
                label,
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return drawn


@app.get("/api/datasets/image")
def dataset_image(path: str, split: Literal["train", "val"] = "train", idx: int = 0) -> Response:
    d = _safe(path)
    img_dir = d / split / "images"
    imgs = sorted(img_dir.glob("*")) if img_dir.exists() else []
    if not imgs:
        raise HTTPException(404, "no images")
    idx = max(0, min(idx, len(imgs) - 1))
    ip = imgs[idx]
    img = cv2.imread(str(ip))
    if img is None:
        raise HTTPException(404, "unreadable image")
    names = _class_names(d)
    coco = None
    cj = d / f"annotations_{split}.json"
    if cj.exists():
        try:  # in-progress/partial COCO -> fall back to YOLO txt
            data = json.loads(cj.read_text())
            by_name = {im["file_name"].split("/")[-1]: im["id"] for im in data["images"]}
            anns: dict = {}
            for a in data["annotations"]:
                anns.setdefault(a["image_id"], []).append(a)
            coco = (by_name, anns)
        except (ValueError, OSError, KeyError, TypeError):
            coco = None
    lp = d / split / "labels" / (ip.stem + ".txt")
    drawn = _draw_labels(img, lp, names, coco, ip.name)
    ok, buf = cv2.imencode(".png", drawn)
    return Response(
        content=buf.tobytes(),
        media_type="image/png",
        headers={"X-Total": str(len(imgs)), "X-Name": ip.name},
    )


@app.get("/api/datasets/download")
def dataset_download(path: str = Query(...), split: str | None = None) -> FileResponse:
    """Stream a generated dataset (or a single split) back to the browser as a .zip.

    Sandboxed via _safe(); os.walk does not follow symlinks, and dotfiles/dotdirs
    (e.g. .seg_cache) are pruned so the archive stays to the exported artifacts.
    """
    d = _safe(path)
    if not d.is_dir():
        raise HTTPException(404, "no such dataset")
    if split not in (None, "train", "val"):
        raise HTTPException(400, "split must be 'train' or 'val'")
    subtree = (d / split) if split else d
    if not subtree.is_dir():
        raise HTTPException(404, f"no '{split}' split in dataset")

    fd, name = tempfile.mkstemp(suffix=".zip", prefix="msg_dl_")
    os.close(fd)
    zip_path = Path(name)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for dirpath, dirnames, files in os.walk(subtree):
                dirnames[:] = sorted(n for n in dirnames if not n.startswith("."))
                for fn in sorted(files):
                    if fn.startswith("."):
                        continue
                    fp = Path(dirpath) / fn
                    # keep the dataset folder name as the archive root so it unzips tidily
                    zf.write(fp, arcname=str(Path(d.name) / fp.relative_to(d)))
    except OSError as e:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(500, f"could not build archive: {e}") from e

    fname = f"{d.name}{('_' + split) if split else ''}.zip"
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=fname,
        background=BackgroundTask(zip_path.unlink, missing_ok=True),
    )


@app.get("/api/fs/list")
def fs_list(path: str | None = None, kind: str = "dir") -> dict:
    """List directory entries for the form's path picker, sandboxed to ALLOWED_BASES.
    kind='dir' shows folders only (sources/out/...); kind='file' also lists files
    (segment_map/sam_checkpoint). `parent` is null at an allow-list boundary so the
    UI can't navigate above the sandbox."""
    base = _safe(path) if path else DATA_ROOT
    # climb to the nearest existing dir so a not-yet-created 'out' path (e.g.
    # datasets/soybean_healthy) opens at its parent instead of 404-ing.
    while not base.exists() and base.parent != base and _is_allowed(base.parent):
        base = base.parent
    if base.is_file():
        base = base.parent
    if not base.is_dir():
        raise HTTPException(404, "no such directory")
    entries = []
    try:
        items = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        raise HTTPException(403, f"cannot list: {e}") from e
    for item in items:
        if item.name.startswith("."):  # hide dotfiles (.seg_cache/.assets.pkl)
            continue
        try:
            is_dir = item.is_dir()
        except OSError:
            continue
        if kind == "dir" and not is_dir:  # dir-picker: folders only
            continue
        entries.append({"name": item.name, "path": str(item), "is_dir": is_dir})
    up = base.parent
    parent = str(up) if (up != base and _is_allowed(up)) else None
    return {"path": str(base), "parent": parent, "entries": entries, "data_root": str(DATA_ROOT)}


@app.get("/api/file")
def serve_file(path: str = Query(...)) -> Response:
    p = _safe(path)
    if not p.is_file():
        raise HTTPException(404, "not found")
    ext = p.suffix.lower()
    media = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "json": "application/json",
        "txt": "text/plain",
        "yaml": "text/plain",
        "yml": "text/plain",
        "md": "text/markdown",
    }
    return Response(
        content=p.read_bytes(), media_type=media.get(ext.lstrip("."), "application/octet-stream")
    )


# --------------------------------------------------------------------------- #
# seed upload                                                                 #
# --------------------------------------------------------------------------- #
@app.post("/api/seeds/upload")
async def upload_seeds(
    dest: str = Form(...),
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    """Save uploaded seed images into a sandboxed folder usable as ``sources``.

    Each file is validated three ways: the destination is forced under an allowed root
    (_safe), the extension must be in the allowlist, and the bytes must actually decode
    as an image (cv2) — a stronger check than trusting the client's content-type. Reads
    are size-bounded so a huge upload can't exhaust memory.
    """
    d = _safe(dest)
    if d.exists() and not d.is_dir():
        raise HTTPException(400, "destination exists and is not a directory")

    # Validate the whole batch BEFORE touching disk, so a single bad file can't leave a
    # half-written upload (and the destination folder isn't created on a rejected batch).
    # The endpoint is all-or-nothing.
    validated: list[tuple[str, bytes]] = []
    for uf in files:
        fname = Path(uf.filename or "").name  # strip any path components from the name
        if not fname:
            continue
        if Path(fname).suffix.lower() not in _UPLOAD_EXTS:
            raise HTTPException(400, f"unsupported file type: {fname} (allowed: png/jpg/jpeg)")
        data = await uf.read(_UPLOAD_MAX_BYTES + 1)
        if len(data) > _UPLOAD_MAX_BYTES:
            raise HTTPException(
                413, f"file too large: {fname} (max {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB)"
            )
        if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
            raise HTTPException(400, f"not a decodable image: {fname}")
        validated.append((fname, data))

    if not validated:
        raise HTTPException(400, "no valid image files uploaded")

    # Everything passed — only now create the folder and write the files.
    d.mkdir(parents=True, exist_ok=True)
    for fname, data in validated:
        (d / fname).write_bytes(data)

    return UploadResponse(
        saved_dir=str(d), files=[name for name, _ in validated], count=len(validated)
    )


# --------------------------------------------------------------------------- #
# static SPA                                                                  #
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text())


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
