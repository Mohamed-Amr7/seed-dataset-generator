# Enabling the GPU for MultiSeedGen

There are **two independent ways** to put the RTX 3060 to work. You only need
the host-pip path for day-to-day use; the Docker path is for a reproducible,
deployable container (and it also fixes seed-bank's prod GPU worker).

---

## Path 1 — Host (no root, no Docker)   ← already wired up

`rembg` runs on the GPU through `onnxruntime-gpu`. The CUDA runtime ships as pip
wheels, so **nothing system-level is required** — the kernel driver (already
present: `nvidia-smi` → driver 580 / CUDA 13) is enough.

```bash
pip uninstall -y onnxruntime
# IMPORTANT: pin 1.26.0 (links CUDA 12). onnxruntime-gpu 1.27 links CUDA 13
# (libcudart.so.13) and there are no real CUDA-13 pip runtime wheels yet (only
# 0.0.1 stubs), so 1.27 fails to import. 1.26.0 + the cu12 wheels works.
pip install "onnxruntime-gpu==1.26.0" \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
    nvidia-curand-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

# verify the CUDA execution provider is visible:
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# -> [... 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Verified on this box: rembg on the RTX 3060 is **~4.2x faster** than CPU
(40 Test cut-outs: 29.2s CPU vs 6.9s GPU, jobs=1) and produces **identical
masks** (IoU 1.0). Peak VRAM ~0.7 GB.

Then just run normally — `--rembg-provider auto` (default) uses the GPU when it
is available:

```bash
python -m multiseedgen --sources seeds/Test/images --segment rembg \
    --num-images 0 --check-seeds 24 --out /tmp/chk          # logs the rembg EP
```

`--rembg-provider {auto,cuda,cpu}` forces the choice. The cut-out cache is
namespaced by device, so CPU and GPU masks never collide.

---

## Path 2 — Docker with GPU  (one-time root step)

GPU passthrough into containers needs the **NVIDIA Container Toolkit**, which is
NOT installed on this host yet (verified: `docker run --gpus all ...` fails with
*"could not select device driver"*; `/etc/cdi` and `/etc/docker/daemon.json` are
absent). Install it once, as root (Fedora):

```bash
# 1) add the NVIDIA Container Toolkit repo (REQUIRED on Fedora - the package is NOT
#    in the default repos; the rpmfusion *driver* repo does not provide it). Without
#    this, `dnf install nvidia-container-toolkit` fails with "No match for argument".
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# 2) install the toolkit
sudo dnf install -y nvidia-container-toolkit

# 3) wire it into Docker. This host's Docker uses native CDI (it lists /etc/cdi as a
#    spec dir), so GENERATE THE CDI SPEC - it is required here, not optional:
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker   # also register the nvidia runtime
sudo systemctl restart docker

# 4) verify the GPU is now visible inside a container
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

After that, build and run MultiSeedGen on the GPU:

```bash
docker compose -f docker/compose.gpu.yaml up --build      # Web UI -> http://localhost:8000
# or a batch CLI run:
docker compose -f docker/compose.gpu.yaml run --rm multiseedgen \
    python -m multiseedgen --sources /data/seeds --out /data/out --num-images 2000
```

> Installing the toolkit **also unblocks seed-bank's prod GPU worker**
> (`compose.prod.yaml`, target `runtime-gpu`), which currently can't acquire a
> GPU on this box and falls back to the CPU image.

---

## Path 3 — opt-in SAM image (`--segment sam`)

SAM needs `torch` + `segment-anything`, which are heavy, so they live in a
**separate, opt-in image** (`docker/Dockerfile.sam`, `python:3.12-slim` — PyTorch
has no Python 3.14 wheels, so it can't go on the default GPU image). The default
image stays light and torch-free; you only build this when you want SAM.

```bash
# 1) download a checkpoint once (vit_b ~358 MB fits a 6 GB GPU / runs on CPU):
mkdir -p models && curl -fL -o models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 2) build + run the opt-in SAM image (CPU torch by default — no toolkit needed):
docker compose -f docker/compose.gpu.yaml --profile sam up --build multiseedgen-sam
#    -> Web UI on http://localhost:8001  (the default service still uses :8000)

# or a batch CLI run with SAM:
docker compose -f docker/compose.gpu.yaml --profile sam run --rm multiseedgen-sam \
    python -m multiseedgen --sources /data/seeds --out /data/out --num-images 2000 \
    --segment sam --sam-checkpoint models/sam_vit_b_01ec64.pth --sam-model-type vit_b
```

The `sam` compose **profile** is the opt-in switch: a plain
`docker compose -f docker/compose.gpu.yaml up` ignores `multiseedgen-sam`
entirely, so the default path never builds torch.

The `sam` compose **profile** is the opt-in switch: a plain
`docker compose -f docker/compose.gpu.yaml up` ignores both SAM services entirely,
so the default path never builds torch.

Notes:
- **Checkpoint path.** `--sam-checkpoint` is resolved against the container WORKDIR
  `/app`; the compose service mounts the repo `models/` at `/app/models`, so the
  relative `models/sam_vit_b_01ec64.pth` above works as-is.
- The CPU SAM image is ~2.6 GB vs the ~light default — which is exactly why it is
  kept opt-in. CPU SAM is fine for spot-checks (~100 s/image for vit_b) but slow for
  full datasets; use the GPU service below for throughput.

### SAM on the GPU (`multiseedgen-sam-gpu`)

A second opt-in service builds the same image with **CUDA torch** and reserves the
GPU. vit_b fits the 6 GB RTX 3060.

**Prerequisite — install the NVIDIA Container Toolkit (Path 2 above), as root.**
Without it the GPU reservation fails with *"could not select device driver"*. This
is the one step that cannot be done unprivileged:
```bash
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi   # verify
```

Then build + run the GPU SAM service (its own `sam-gpu` profile, port 8002):
```bash
# Web UI on the GPU:
docker compose -f docker/compose.gpu.yaml --profile sam-gpu up --build multiseedgen-sam-gpu
#   -> http://localhost:8002

# or a batch CLI run on the GPU:
docker compose -f docker/compose.gpu.yaml --profile sam-gpu run --rm multiseedgen-sam-gpu \
    python -m multiseedgen --sources /data/seeds --out /data/out --num-images 2000 \
    --segment sam --sam-checkpoint models/sam_vit_b_01ec64.pth --sam-model-type vit_b
# expect the log line:  [sam] loaded vit_b on cuda
```

Notes:
- The CUDA torch wheel is **cu126** (`docker/compose.gpu.yaml` build arg
  `TORCH_INDEX`). Driver 580 is CUDA-13-capable and backward-compatible with the
  CUDA-12.6 runtime that the torch wheel bundles; to match CUDA 13 exactly instead,
  set `TORCH_INDEX=https://download.pytorch.org/whl/cu130`.
- The GPU image is larger (~6 GB; CUDA torch bundles its own CUDA runtime). It is
  unrelated to the rembg cu12 stack — torch ships self-contained CUDA libs.
- Cap concurrent SAM workers with `--gpu-seg-jobs` if VRAM is tight (vit_b is light,
  but rembg + SAM together can pressure 6 GB).
