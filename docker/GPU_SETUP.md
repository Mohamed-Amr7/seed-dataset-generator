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
# 1) install the toolkit
sudo dnf install -y nvidia-container-toolkit

# 2) wire it into Docker (writes /etc/docker/daemon.json + CDI specs)
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # optional, CDI mode
sudo systemctl restart docker

# 3) verify the GPU is now visible inside a container
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
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
