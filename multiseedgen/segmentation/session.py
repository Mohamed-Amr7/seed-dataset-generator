"""rembg / SAM session state + onnxruntime device resolution.

The session singletons (_REMBG_SESSION / _SAM_SEGMENTER / _RESOLVED_DEVICE / ...) are
MODULE-level on purpose: loky/joblib workers are fresh processes that re-import this
module and lazily rebuild their OWN (unpicklable) session, so the state "survives the
fork". A live onnxruntime/torch session must never be stored on an object that crosses
the worker boundary."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

import cv2
import numpy as np

from ..models import Config

logger = logging.getLogger(__name__)


_REMBG_SESSION = None
_REMBG_PROVIDER = "auto"  # auto|cuda|cpu - set per process from cfg
_ORT_PROVIDERS_CACHE = None  # memoized onnxruntime.get_available_providers()
_AUTO_GPU_SEG_JOBS = None  # memoized VRAM-scaled session count (per process)


def _free_vram_gb() -> float | None:
    """Best-effort free VRAM (GiB) of the first CUDA device via ``nvidia-smi``; None if
    it can't be probed. No new dependency: ``nvidia-smi`` ships with the CUDA driver
    that onnxruntime-gpu already requires, so it's present wherever GPU rembg can run."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        first = out.strip().splitlines()[0].strip()
        return int(first) / 1024.0  # MiB -> GiB
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _auto_gpu_seg_jobs() -> int:
    """Concurrent GPU rembg sessions scaled to free VRAM (used when ``gpu_seg_jobs`` is
    the ``0`` = auto sentinel). Each CUDA rembg session needs ~0.9 GB; use ~70% of free
    VRAM and never exceed the CPU count. Falls back to 4 (the documented 6 GB baseline)
    when VRAM can't be probed, so behavior is unchanged where the probe is unavailable.
    Memoized per process so every worker computes it at most once."""
    global _AUTO_GPU_SEG_JOBS
    if _AUTO_GPU_SEG_JOBS is not None:
        return _AUTO_GPU_SEG_JOBS
    free = _free_vram_gb()
    if free is None:
        _AUTO_GPU_SEG_JOBS = 4
    else:
        n = int(round(free * 0.7 / 0.9))
        _AUTO_GPU_SEG_JOBS = max(1, min(n, os.cpu_count() or 1))
    return _AUTO_GPU_SEG_JOBS


def _is_cuda_oom(exc: Exception) -> bool:
    """True for a CUDA out-of-memory / hard CUDA failure from onnxruntime (which raises
    a generic RuntimeException/Fail, so we match on the message). Deliberately tight so
    ordinary errors (bad image, shape mismatch) still propagate instead of silently
    routing to CPU."""
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "out of memory",
            "bfc_arena",
            "cudaerrormemoryalloc",  # cudaErrorMemoryAllocation
            "cuda_error_out_of_memory",
            "cuda failure",
            "cuda error",
        )
    )


def _set_rembg_provider(pref: str) -> None:
    """Set the desired execution provider for rembg in THIS process. Called at the
    top of the parent loader and every segmentation worker, so the choice survives
    the loky fork. Changing it resets any built session."""
    global _REMBG_PROVIDER, _REMBG_SESSION
    pref = (pref or "auto").lower()
    if pref != _REMBG_PROVIDER:
        _REMBG_PROVIDER, _REMBG_SESSION = pref, None


def _ort_available() -> list:
    global _ORT_PROVIDERS_CACHE
    if _ORT_PROVIDERS_CACHE is None:
        try:
            import onnxruntime as ort

            _ORT_PROVIDERS_CACHE = list(ort.get_available_providers())
        except Exception:
            _ORT_PROVIDERS_CACHE = []
    return _ORT_PROVIDERS_CACHE


_RESOLVED_DEVICE: dict = {}  # pref -> "cuda"/"cpu" (REAL probe result, per process)


def _resolve_rembg_device(pref: str) -> str:
    """Probe the device rembg can ACTUALLY use (not the build-time provider list):
    build a real rembg session and read the providers ORT *granted*. Memoized.
    If the GPU build is present but CUDA can't initialise (no driver / missing or
    incompatible runtime libs), this correctly returns 'cpu' so the cache key and
    the worker cap match the masks actually produced. 'cpu' short-circuits (no load)."""
    pref = (pref or "auto").lower()
    if pref in _RESOLVED_DEVICE:
        return _RESOLVED_DEVICE[pref]
    if pref == "cpu" or "CUDAExecutionProvider" not in _ort_available():
        _RESOLVED_DEVICE[pref] = "cpu"
        return "cpu"
    dev = "cpu"
    global _REMBG_SESSION
    try:
        import onnxruntime as ort
        from rembg import new_session

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        s = new_session(
            providers=[
                ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
                "CPUExecutionProvider",
            ]
        )
        granted = s.inner_session.get_providers()
        dev = "cuda" if "CUDAExecutionProvider" in granted else "cpu"
        _REMBG_SESSION = s  # reuse the probe session; don't waste the load
        if pref == "cuda" and dev == "cpu":
            logger.warning(
                "  [rembg][warn] CUDA requested but ORT granted CPU; check "
                "onnxruntime-gpu / CUDA runtime libs"
            )
    except Exception:
        dev = "cpu"
    _RESOLVED_DEVICE[pref] = dev
    return dev


def _effective_rembg_provider(pref: str) -> str:
    """Device the cache key + worker cap should assume. Concrete 'cpu'/'cuda' are
    taken as already-resolved -- the parent loader rewrites cfg.rembg_provider to a
    PROBED concrete device before dispatch, so workers never re-probe. 'auto' probes."""
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if "CUDAExecutionProvider" in _ort_available() else "cpu"
    return _resolve_rembg_device("auto")


def _rembg_ort_providers(pref: str) -> list:
    """ORT provider list (priority order) for a desired preference. On CUDA we cap
    arena growth (kSameAsRequested) so several concurrent sessions share the 6GB
    card without one greedily grabbing it all."""
    if _effective_rembg_provider(pref) == "cuda":
        return [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def _mask_rembg(bgr: np.ndarray) -> np.ndarray | None:
    global _REMBG_SESSION
    try:
        from rembg import new_session, remove
    except ImportError:
        logger.warning("  [warn] rembg not installed; falling back to threshold")
        return None
    if _REMBG_SESSION is None:
        want_cuda = _effective_rembg_provider(_REMBG_PROVIDER) == "cuda"
        providers = _rembg_ort_providers(_REMBG_PROVIDER)
        if want_cuda:
            try:  # make pip-installed CUDA libs loadable
                import onnxruntime as ort

                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
            except Exception:
                pass
        try:
            _REMBG_SESSION = new_session(providers=providers)
        except TypeError:  # very old rembg without providers kw
            _REMBG_SESSION = new_session()
        try:  # report what ORT actually granted
            actual = _REMBG_SESSION.inner_session.get_providers()
        except Exception:
            actual = providers
        logger.info(f"  [rembg] execution providers: {actual}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        out = remove(rgb, session=_REMBG_SESSION)  # RGBA
    except Exception as exc:
        # A mid-run CUDA OOM (e.g. too many concurrent sessions, or an oversized source)
        # would otherwise crash the whole worker after long segmentation work. If we're
        # on CUDA, drop THIS worker to CPU permanently and retry the image once; any
        # non-OOM error still propagates. Caveat: the parent rewrites cfg.rembg_provider
        # to a concrete 'cuda' before forking and the seg-cache key encodes that device,
        # so a CPU-fallback mask is cached under the cuda key (acceptable on this rare
        # crash-avoidance path; see RECOMMENDED_SETTINGS.md caveats).
        if not (_is_cuda_oom(exc) and _effective_rembg_provider(_REMBG_PROVIDER) == "cuda"):
            raise
        logger.warning(
            f"  [rembg][warn] CUDA failure during inference ({type(exc).__name__}); "
            f"falling back to CPU for the rest of this worker"
        )
        _set_rembg_provider("cpu")  # resets _REMBG_SESSION -> None
        _REMBG_SESSION = new_session(providers=_rembg_ort_providers("cpu"))
        out = remove(rgb, session=_REMBG_SESSION)
    return out[..., 3]


def _crop_to_alpha(bgr: np.ndarray, alpha: np.ndarray, pad: int = 2) -> np.ndarray | None:
    """Tight-crop BGR+alpha to the alpha extent and return a BGRA patch."""
    ys, xs = np.where(alpha > 8)
    if ys.size == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + 1 + pad)
    bgr_c = bgr[y0:y1, x0:x1]
    a_c = alpha[y0:y1, x0:x1]
    return np.dstack([bgr_c, a_c]).astype(np.uint8)


_SAM_SEGMENTER = None  # process-global SAMSegmenter (survives the loky fork)


def _mask_sam(bgr: np.ndarray, cfg: Config) -> np.ndarray | None:
    """Optional SAM path. Import-guarded: returns None (callers then fall back) unless
    a checkpoint is configured AND torch + segment_anything are installed. Returns a
    0/255 mask for the seed, matching the rembg alpha contract. The segmenter is built
    once per process and reused. Prompt mode is cfg.sam_prompt_type."""
    global _SAM_SEGMENTER
    if getattr(cfg, "sam_checkpoint", None) is None:
        return None
    if _SAM_SEGMENTER is None:
        try:
            from ..sam_segment import SAMSegmenter, sam_available
        except Exception:
            return None
        if not sam_available():
            logger.warning(
                "  [warn] --segment sam needs torch + segment-anything "
                "(pip install segment-anything torch); skipping"
            )
            return None
        try:
            _SAM_SEGMENTER = SAMSegmenter(
                checkpoint=cfg.sam_checkpoint,
                model_type=cfg.sam_model_type,
                points_per_side=cfg.sam_points_per_side,
                pred_iou_thresh=cfg.sam_pred_iou_thresh,
                stability_score_thresh=cfg.sam_stability_score_thresh,
            )
            logger.info(f"  [sam] loaded {cfg.sam_model_type} on {_SAM_SEGMENTER.device}")
        except Exception as e:  # bad checkpoint / OOM / etc.
            logger.warning(f"  [warn] SAM load failed ({e}); skipping")
            return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    pt = (cfg.sam_prompt_type or "automatic").lower()
    try:
        if pt == "box":
            inset = max(2, int(float(cfg.grabcut_rect_margin) * min(h, w)))
            mask, _ = _SAM_SEGMENTER.segment_with_box(
                rgb, [inset, inset, max(1, w - 2 * inset), max(1, h - 2 * inset)]
            )
        elif pt == "points":
            mask, _ = _SAM_SEGMENTER.segment_with_points(rgb, [(w // 2, h // 2)], [1])
        else:  # automatic (default)
            mask, _ = _SAM_SEGMENTER.segment_automatic(rgb)
    except Exception as e:  # pragma: no cover
        logger.warning(f"  [warn] SAM inference failed ({e}); skipping")
        return None
    return mask if mask is not None and int((mask > 0).sum()) else None
