"""rembg / SAM session state + onnxruntime device resolution.

The session singletons (_REMBG_SESSION / _SAM_SEGMENTER / _RESOLVED_DEVICE / ...) are
MODULE-level on purpose: loky/joblib workers are fresh processes that re-import this
module and lazily rebuild their OWN (unpicklable) session, so the state "survives the
fork". A live onnxruntime/torch session must never be stored on an object that crosses
the worker boundary."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..models import Config

logger = logging.getLogger(__name__)


_REMBG_SESSION = None
_REMBG_PROVIDER = "auto"  # auto|cuda|cpu - set per process from cfg
_ORT_PROVIDERS_CACHE = None  # memoized onnxruntime.get_available_providers()


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
    out = remove(rgb, session=_REMBG_SESSION)  # RGBA
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
