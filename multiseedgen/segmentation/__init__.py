"""Confidence-scored segmentation cascade, rembg/SAM fallback, cut-out cache, loader.

Split from the former single ``segmentation.py``. The public surface is unchanged:
``load_seeds`` / ``write_seg_report`` (the driver pair the pipeline calls) and
``segment_seed`` / ``extract_background`` (the per-source primitives). A few private
helpers are re-exported too because the cut-out-cache contract test pins them
(``_seg_cache_key`` / ``_seg_params_signature`` / ``_effective_rembg_provider``) and
``SEG_VERSION`` is part of that wire format.
"""

from __future__ import annotations

from ..constants import SEG_VERSION
from .cache import _seg_cache_key, _seg_params_signature
from .loader import load_seeds, write_seg_report
from .segment import extract_background, segment_seed
from .session import _effective_rembg_provider

__all__ = [
    "SEG_VERSION",
    "load_seeds",
    "write_seg_report",
    "segment_seed",
    "extract_background",
    "_seg_cache_key",
    "_seg_params_signature",
    "_effective_rembg_provider",
]
