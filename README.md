# MultiSeedGen

Synthetic **multi-seed detection** dataset generator (v3.0.1). Cuts individual seeds out
of source photos, composites many of them onto realistic backgrounds with domain-matched
lighting and camera degradation, and exports detection/segmentation datasets (YOLO,
YOLO-seg, COCO) with full provenance and QA reports. Ships with both a CLI and a FastAPI
web UI, and produces **byte-reproducible** output for a fixed `(config, seed)`.

## Quick start

```bash
pip install -e .                                  # core CLI; add ".[web]" for the UI

seedgen --config configs/soybean_healthy.yaml     # generate from a recipe
multiseedgen-web                                  # or launch the web UI
```

Outputs (images, annotations, `data.yaml`, provenance, QA charts) are written to the
config's `out:` directory. `python seedgen.py …` still works as a backward-compat shim.

## Documentation

Full docs live in **[`docs/`](docs/README.md)**:

- **[Installation](docs/installation.md)** — install + optional extras (`web`/`rembg`/`gpu`/`sam`/`dev`).
- **[Usage](docs/usage.md)** — CLI, web UI, config files, helper scripts.
- **[Architecture](docs/architecture.md)** — package layout, the determinism contract, fork-safety, error/logging model.
- **[Configuration tuning](docs/RECOMMENDED_SETTINGS.md)** & **[Augmentation/segmentation guide](docs/AUGMENTATION.md)**.
- **[Contributing](docs/contributing.md)** — dev setup, the regression gate, test conventions.
- **[GPU / Docker](docker/GPU_SETUP.md)** — CUDA pins, `onnxruntime-gpu`, GPU SAM.

## Package layout

```
multiseedgen/
  config.py errors.py logging_setup.py   validated Config, typed errors, logging
  models.py constants.py optdeps.py      dataclasses, version/cache key, optional-dep guards
  cliconfig.py                           argparse + --config loading
  segmentation/                          segment a seed out of a source (masks/session/cache/loader)
  compositing/                           transforms / backgrounds / blend / scene assembly
  degrade.py export.py qa.py             degradation, YOLO/COCO writers, QA charts
  pipeline.py                            deterministic per-scene workers + run() driver
  web/                                   FastAPI backend (app/schemas/services) + static UI
```

See [`docs/architecture.md`](docs/architecture.md) for the annotated tree and data flow.

## Configs

`configs/` holds ready-to-run dataset definitions (`root`, `wheat`, `test`,
`soybean_healthy`) plus [`configs/example.yaml`](configs/example.yaml), a fully-commented
reference exercising every option. Files accept flat keys or the grouped/nested layout
(`segmentation:`, `split:`, `sampling:`, `augmentation:`, `export:`); CLI flags override
file values.

## Data

Source datasets and generated outputs are **not** tracked in git (`seeds/`, `datasets/`,
`out/`, `*.zip` are gitignored — some exceed GitHub's size limits). A small,
demo-runnable sample of seed images is included under `seeds/` so the pipeline can be
exercised without the full corpus.
