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
import sys
import tempfile
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from collections import deque
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..cliconfig import _config_defaults, _load_config_file, build_parser
from ..models import Config
from ..optdeps import yaml

ROOT = Path(__file__).resolve().parents[2]            # the MultiSeedGen project dir
STATIC = Path(__file__).resolve().parent / "static"
DATA_ROOT = Path(os.environ.get("MULTISEEDGEN_DATA_ROOT", str(ROOT))).resolve()
TMP = Path(tempfile.gettempdir()).resolve()
ALLOWED_BASES = [ROOT, DATA_ROOT, TMP]

LOG_MAXLINES = 4000        # bounded per-run log ring buffer
RUNS_MAX = 40             # cap retained runs (evict oldest finished)

app = FastAPI(title="MultiSeedGen")
PARSER = build_parser()
RUNS: dict = {}                                        # run_id -> state dict


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


# Path fields that should render a "Browse…" picker in the form, and whether the
# picker selects a directory (sources/out/...) or a file (segment_map/sam_checkpoint).
# 'config' is intentionally absent — schema() never emits it as a form field.
_PATH_KIND = {"sources": "dir", "out": "dir", "textures": "dir", "cache_dir": "dir",
              "segment_map": "file", "sam_checkpoint": "file"}


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    return v


def _action_widget(action) -> dict:
    cls = type(action).__name__
    if cls in ("_StoreTrueAction", "_StoreFalseAction"):
        widget = "bool"
    elif action.choices:
        widget = "choice"
    elif action.type is int:
        widget = "int"
    elif action.type is float:
        widget = "float"
    else:
        widget = "text"
    return {
        "widget": widget,
        "choices": list(action.choices) if action.choices else None,
        "help": action.help or "",
        "flag": action.option_strings[0] if action.option_strings else "",
    }


# --------------------------------------------------------------------------- #
# recommended defaults + named presets (good starting points for the form)    #
# --------------------------------------------------------------------------- #
# Production-quality overlay on top of the bare Config defaults; mirrors the
# "Global recommendations" in RECOMMENDED_SETTINGS.md so a fresh form is ready to
# generate a quality dataset after only setting `sources` and `out`.
_RECOMMENDED = {
    "min_seeds": 8, "max_seeds": 30, "scale_min": 0.05, "scale_max": 0.16,
    "bg_from_sources": True, "neg_frac": 0.10,
    "segment": "auto", "rembg_provider": "auto",
    "fmt": "both", "masks": True,
    "degrade": True, "noise_sigma": 3.0,
    "jpeg_quality_min": 80, "jpeg_quality_max": 95, "blur_max_sigma": 0.5,
    "shadow_dir_mode": "scene", "shadow_opacity": 0.25, "edge_feather": 0.5,
    "qa_report": True, "visualize_count": 12,
    "workers": 0, "parallel_backend": "auto", "seed": 7,
}

# Named recipes, sourced from configs/*.yaml (single source of truth with the CLI).
_PRESET_FILES = [
    ("Root — clean, 27 species", "root.yaml"),
    ("Wheat — tiny, low-contrast", "wheat.yaml"),
    ("Test — hardest (polygons)", "test.yaml"),
    ("Soybean — Healthy (single class)", "soybean_healthy.yaml"),
]


def _load_presets() -> dict:
    """{"Recommended (general)": {...}, <dataset recipes from configs/*.yaml>}.
    Any missing/broken config is skipped so this endpoint can never 500."""
    presets = {"Recommended (general)": dict(_RECOMMENDED)}
    for label, fname in _PRESET_FILES:
        p = ROOT / "configs" / fname
        try:
            if p.exists():
                presets[label] = {k: _jsonable(v)
                                  for k, v in _load_config_file(p).items()}
        except (SystemExit, Exception):
            continue
    return presets


# --------------------------------------------------------------------------- #
# schema (drives the form)                                                    #
# --------------------------------------------------------------------------- #
@app.get("/api/schema")
def schema() -> dict:
    defaults = _config_defaults()
    field_names = {f.name for f in dataclass_fields(Config)}
    # map dest -> argparse action metadata
    meta = {}
    for grp in PARSER._action_groups:
        for act in grp._group_actions:
            if act.dest in ("help", "config"):
                continue
            if act.dest in field_names:
                meta.setdefault(act.dest, (grp.title, _action_widget(act)))

    groups: dict = {}
    for dest, (title, info) in meta.items():
        groups.setdefault(title, []).append({
            "name": dest,
            "default": _jsonable(defaults.get(dest)),
            "path_kind": _PATH_KIND.get(dest),    # 'dir'|'file' -> render Browse…; else None
            **info,
        })
    # any Config field with no CLI flag -> a generic group
    for f in dataclass_fields(Config):
        if f.name not in meta and f.name != "config":
            groups.setdefault("other", []).append({
                "name": f.name, "default": _jsonable(defaults.get(f.name)),
                "path_kind": _PATH_KIND.get(f.name),
                "widget": "text", "choices": None, "help": "", "flag": "",
            })
    order = ["I/O", "scene composition", "per-seed augmentation", "drop shadow",
             "backgrounds", "segmentation", "realism", "performance",
             "output", "other"]
    out = []
    for title in order:
        if title in groups:
            out.append({"title": title, "fields": groups[title]})
    for title, fields in groups.items():       # any unordered groups
        if title not in order:
            out.append({"title": title, "fields": fields})
    return {"groups": out, "data_root": str(DATA_ROOT), "project_root": str(ROOT),
            "recommended": _RECOMMENDED, "presets": _load_presets()}


# --------------------------------------------------------------------------- #
# config build / save / load                                                  #
# --------------------------------------------------------------------------- #
def _clean_config(cfg: dict) -> dict:
    """Drop empties so optional path fields fall back to their defaults."""
    field_names = {f.name for f in dataclass_fields(Config)}
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
    raw = _load_config_file(p)                 # validates keys, maps format->fmt
    return {k: _jsonable(v) for k, v in raw.items()}


@app.post("/api/config/save")
async def config_save(payload: dict) -> dict:
    path = payload.get("path")
    cfg = _clean_config(payload.get("config", {}))
    if not path:
        raise HTTPException(400, "missing 'path'")
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    want_yaml = p.suffix.lower() in (".yaml", ".yml")
    if want_yaml and yaml is None:
        raise HTTPException(400, "YAML support not installed (pip install pyyaml); "
                                 "save with a .json path instead")
    if want_yaml:
        p.write_text(yaml.safe_dump({"config": cfg}, sort_keys=True))
    else:
        p.write_text(json.dumps({"config": cfg}, indent=2, sort_keys=True))
    return {"saved": str(p), "format": "yaml" if want_yaml else "json"}


# --------------------------------------------------------------------------- #
# runs (subprocess + live log stream)                                         #
# --------------------------------------------------------------------------- #
def _evict_runs() -> None:
    """Bound RUNS: drop the oldest finished runs once over the cap."""
    while len(RUNS) > RUNS_MAX:
        for k in list(RUNS):                     # insertion order = oldest first
            if RUNS[k]["status"] != "running":
                RUNS.pop(k, None)
                break
        else:
            break                                # nothing finished to evict


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
    except Exception as e:                        # never leave subscribers hanging
        run["status"] = "error"
        run["log"].append(f"[pump error] {e!r}")
    finally:
        try:
            Path(run["cfg_path"]).unlink(missing_ok=True)   # don't leak temp cfg
        except Exception:
            pass
        for q in list(run["subs"]):
            q.put_nowait(None)                     # sentinel: completion (always)


async def _launch(cfg: dict, extra_args: Optional[list] = None,
                  label: str = "run") -> str:
    cfg = _clean_config(cfg)
    if not cfg.get("sources") or not cfg.get("out"):
        raise HTTPException(400, "both 'sources' and 'out' are required")
    cfg_path = _write_config_json(cfg)
    argv = [sys.executable, "-m", "multiseedgen", "--config", str(cfg_path)]
    if extra_args:
        argv += extra_args
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, limit=1 << 20,    # 1MB lines (tqdm/tracebacks)
        start_new_session=True,                             # own process group -> killpg
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    rid = f"{label}-{proc.pid}"
    RUNS[rid] = {"proc": proc, "log": deque(maxlen=LOG_MAXLINES), "status": "running",
                 "subs": set(), "out": cfg.get("out"), "cmd": " ".join(argv),
                 "returncode": None, "cfg_path": str(cfg_path)}
    _evict_runs()
    asyncio.create_task(_pump(rid))
    return rid


@app.post("/api/runs")
async def start_run(payload: dict) -> dict:
    rid = await _launch(payload.get("config", {}), label="run")
    return {"run_id": rid, "out": RUNS[rid]["out"]}


@app.get("/api/runs")
def list_runs() -> dict:
    return {"runs": [{"id": k, "status": v["status"], "out": v["out"],
                      "returncode": v["returncode"]} for k, v in RUNS.items()]}


@app.get("/api/runs/{rid}")
def run_status(rid: str) -> dict:
    if rid not in RUNS:
        raise HTTPException(404, "no such run")
    v = RUNS[rid]
    return {"id": rid, "status": v["status"], "returncode": v["returncode"],
            "out": v["out"], "log": list(v["log"])[-400:]}


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
    for line in run["log"]:                    # backlog first
        q.put_nowait(line)
    if run["status"] != "running":
        q.put_nowait(None)
    run["subs"].add(q)
    try:
        while True:
            item = await q.get()
            if item is None:
                await ws.send_json({"type": "done",
                                    "returncode": run["returncode"],
                                    "status": run["status"]})
                break
            await ws.send_json({"type": "line", "line": item})
    except WebSocketDisconnect:
        pass
    finally:
        run["subs"].discard(q)


# --------------------------------------------------------------------------- #
# segmentation tuner                                                          #
# --------------------------------------------------------------------------- #
@app.post("/api/seg/check")
async def seg_check(payload: dict) -> dict:
    cfg = dict(payload.get("config", {}))
    n = int(payload.get("n", 24))
    out = _safe(payload.get("out") or str(TMP / "msg_segcheck"))
    cfg.update({"out": str(out), "num_images": 0, "check_seeds": n})
    cfg.setdefault("cache_dir", str(TMP / "msg_segcache"))  # warm across re-checks
    cfg = _clean_config(cfg)
    if not cfg.get("sources"):
        raise HTTPException(400, "'sources' is required")
    cfg_path = _write_config_json(cfg)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "multiseedgen", "--config", str(cfg_path),
            cwd=str(ROOT), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, limit=1 << 20,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        raw, _ = await proc.communicate()
    finally:
        Path(cfg_path).unlink(missing_ok=True)
    log = raw.decode("utf-8", "replace")
    report = None
    rp = out / "seg_report.json"
    if rp.exists():
        try:                                       # partial/corrupt report -> keep the log
            report = json.loads(rp.read_text())
        except (ValueError, OSError):
            report = None
    checks = sorted((out / "seed_check").glob("*.png")) if (out / "seed_check").exists() else []
    imgs = [f"/api/file?path={c}" for c in checks]
    return {"returncode": proc.returncode, "report": report,
            "images": imgs, "log": log[-4000:], "out": str(out)}


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
    return (p / "data.yaml").exists() or any(p.glob("annotations_*.json")) \
        or (p / "train" / "images").exists()


@app.get("/api/datasets")
def list_datasets(root: Optional[str] = None) -> dict:
    base = _safe(root) if root else DATA_ROOT
    found = []
    if _is_dataset(base):
        found.append(str(base))
    for p in sorted(base.rglob("*")):
        if p.is_dir() and _is_dataset(p):
            found.append(str(p))
    # dedupe, cap
    seen, uniq = set(), []
    for f in found:
        if f not in seen:
            seen.add(f); uniq.append(f)
    return {"root": str(base), "datasets": uniq[:200]}


@app.get("/api/datasets/info")
def dataset_info(path: str = Query(...)) -> dict:
    d = _safe(path)
    info = {"path": str(d), "splits": {}, "classes": _class_names(d),
            "qa": [], "seg_report": None}
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


def _draw_labels(img, label_path: Path, names: list,
                 coco=None, fname: str = "") -> np.ndarray:
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
                cx, cy, bw, bh = vals[0]*W, vals[1]*H, vals[2]*W, vals[3]*H
                x0, y0 = int(cx-bw/2), int(cy-bh/2)
                x1, y1 = int(cx+bw/2), int(cy+bh/2)
                cv2.rectangle(drawn, (x0, y0), (x1, y1), (0, 0, 255), 2)
            else:
                pts = np.array(vals, np.float32).reshape(-1, 2)
                pts[:, 0] *= W; pts[:, 1] *= H
                cv2.polylines(drawn, [pts.astype(np.int32)], True, (0, 0, 255), 2)
                x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            cv2.putText(drawn, label, (x0, max(12, y0-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    elif coco is not None:
        by_name, anns = coco
        for a in anns.get(by_name.get(fname, -1), []):
            x, y, w, h = [int(v) for v in a["bbox"]]
            cv2.rectangle(drawn, (x, y), (x+w, y+h), (0, 0, 255), 2)
            c = a.get("category_id", 1) - 1
            label = names[c] if 0 <= c < len(names) else str(c)
            cv2.putText(drawn, label, (x, max(12, y-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return drawn


@app.get("/api/datasets/image")
def dataset_image(path: str, split: str = "train", idx: int = 0) -> Response:
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
        try:                                       # in-progress/partial COCO -> fall back to YOLO txt
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
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"X-Total": str(len(imgs)), "X-Name": ip.name})


@app.get("/api/fs/list")
def fs_list(path: Optional[str] = None, kind: str = "dir") -> dict:
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
        raise HTTPException(403, f"cannot list: {e}")
    for item in items:
        if item.name.startswith("."):              # hide dotfiles (.seg_cache/.assets.pkl)
            continue
        try:
            is_dir = item.is_dir()
        except OSError:
            continue
        if kind == "dir" and not is_dir:           # dir-picker: folders only
            continue
        entries.append({"name": item.name, "path": str(item), "is_dir": is_dir})
    up = base.parent
    parent = str(up) if (up != base and _is_allowed(up)) else None
    return {"path": str(base), "parent": parent, "entries": entries,
            "data_root": str(DATA_ROOT)}


@app.get("/api/file")
def serve_file(path: str = Query(...)) -> Response:
    p = _safe(path)
    if not p.is_file():
        raise HTTPException(404, "not found")
    ext = p.suffix.lower()
    media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "json": "application/json", "txt": "text/plain",
             "yaml": "text/plain", "yml": "text/plain", "md": "text/markdown"}
    return Response(content=p.read_bytes(),
                    media_type=media.get(ext.lstrip("."), "application/octet-stream"))


# --------------------------------------------------------------------------- #
# static SPA                                                                  #
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text())


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
