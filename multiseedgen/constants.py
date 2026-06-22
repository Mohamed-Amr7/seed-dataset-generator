from __future__ import annotations
"""Version + cache-version constants and shared literals."""

from typing import Any, Dict, List, Optional, Tuple




__version__ = "3.0.1"

# Bump SEG_VERSION whenever the segmentation algorithm or scoring changes; it is
# part of the cut-out cache key, so a bump transparently invalidates stale caches.
# v5: rembg can run on GPU (CUDAExecutionProvider); GPU vs CPU masks differ by a
# few pixels, so the effective provider is also folded into the cache signature.
# v6: configurable per-method params (grabcut/otsu/morph) + real SAM path folded
# into the cache signature.
SEG_VERSION = 6


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
