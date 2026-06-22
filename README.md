# MultiSeedGen

Synthetic **multi-seed detection** dataset generator (v3.0.1). Cuts individual
seeds out of source photos, composites many of them onto realistic backgrounds
with domain-matched lighting and camera degradation, and exports
detection/segmentation datasets (YOLO, YOLO-seg, COCO) with full provenance and
QA reports. Ships with both a CLI and a FastAPI web UI.

## Quick start

```bash
# Generate a dataset from a config
python seedgen.py --config configs/soybean_healthy.yaml      # or: python -m multiseedgen --config ...

# Launch the web UI
python -m multiseedgen.web                                   # then open the printed URL
```

Outputs (images, annotations, `data.yaml`, provenance, QA charts) are written to
the config's `out:` directory (e.g. `datasets/soybean_healthy`).

## GPU / Docker

Segmentation uses `rembg` and can run on GPU via `onnxruntime-gpu`. See
[`docker/GPU_SETUP.md`](docker/GPU_SETUP.md), [`docker/Dockerfile.gpu`](docker/Dockerfile.gpu),
and [`docker/compose.gpu.yaml`](docker/compose.gpu.yaml). Tuning guidance lives in
[`RECOMMENDED_SETTINGS.md`](RECOMMENDED_SETTINGS.md).

## Package layout (`multiseedgen/`)

| Module | Responsibility |
|--------|----------------|
| `models.py` | `Config` / `Seed` / `SegResult` dataclasses |
| `constants.py` | version, `SEG_VERSION` (cache key), shared literals |
| `optdeps.py` | optional-dependency guards |
| `segmentation.py` | segmentation cascade + rembg fallback + disk cache |
| `compositing.py` | transforms, backgrounds, placement, scene assembly |
| `degrade.py` | camera-matched photometric degradation |
| `export.py` | YOLO / YOLO-seg / COCO exporters, `data.yaml`, manifest |
| `qa.py` | QA charts, montages, previews, seed checker |
| `pipeline.py` | deterministic per-scene workers + `run()` driver |
| `cliconfig.py` | argparse interface + `--config` loading |
| `web/` | FastAPI backend (`app.py`) + static UI |

`seedgen.py` is a backward-compatible shim that re-exports the package API and
preserves the original `python seedgen.py ...` entry point.

## Configs

`configs/` holds ready-to-run dataset definitions: `root.yaml` (multi-class
spice/seed set), `test.yaml`, `wheat.yaml`, and `soybean_healthy.yaml`. Each sets
sources, output dir, scene/seed counts, segmentation backend, and export format.

## Data

The full source datasets and generated outputs are **not** tracked in git
(`seeds/`, `datasets/`, `out/`, and `*.zip` are gitignored — some exceed GitHub's
file-size limits). A small, demo-runnable **sample** of seed images and
annotations is included under `seeds/` so the pipeline can be exercised without
the full corpus.
