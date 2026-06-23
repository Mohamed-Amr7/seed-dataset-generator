# Installation

MultiSeedGen is a regular Python package (`pyproject.toml`, setuptools). It needs
**Python ≥ 3.10**. Only `numpy` and `opencv-python-headless` are hard imports of the
core pipeline; everything else is an optional extra.

## Core install (CLI + classical segmentation)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This pulls the core runtime (`numpy`, `opencv-python-headless`, `joblib`, `pyyaml`,
`tqdm`, `pydantic`) and installs two console scripts:

| Script | Equivalent | Purpose |
|--------|------------|---------|
| `seedgen` | `python -m multiseedgen` | run a generation |
| `multiseedgen-web` | `python -m multiseedgen.web` | launch the web UI |

> `python seedgen.py …` still works — it is a backward-compatible shim that re-exports
> the package API.

## Optional extras

Install only what you need (extras are additive, e.g. `pip install -e ".[web,rembg]"`):

| Extra | `pip install -e ".[…]"` | Adds |
|-------|--------------------------|------|
| `web` | `web` | FastAPI + uvicorn + websockets (the web UI) |
| `rembg` | `rembg` | CPU `rembg` (U²-Net) for the learned-mask fallback |
| `gpu` | `gpu` | GPU `rembg` via `onnxruntime-gpu==1.26.0` + CUDA 12 runtime wheels |
| `sam` | `sam` | Segment Anything backend (`torch`, `torchvision`, `segment-anything`) |
| `dev` | `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy` |

GPU notes (CUDA pins, the `onnxruntime-gpu` 1.26 vs 1.27 choice, SAM checkpoints) live in
[`../docker/GPU_SETUP.md`](../docker/GPU_SETUP.md). The SAM backend additionally needs a
downloaded checkpoint — see [augmentation.md](AUGMENTATION.md).

## Verify

```bash
pip install -e ".[dev]"
pytest -m "not gpu"        # runs the fast + integration suite (skips GPU-only tests)
```

A green run confirms the package imports, the deterministic pipeline reproduces its
golden manifests, and the web layer responds. See [contributing.md](contributing.md) for
the full developer workflow.
