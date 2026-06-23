"""Seed transforms, backgrounds, placement, and scene composition.

Split from the former single ``compositing.py``. The public surface is unchanged:
``generate_scene`` (the per-scene entry point the pipeline calls) plus the
``transform_seed`` / ``make_background`` / ``place_seed`` building blocks and
``_resize_to_long_side`` (used by segmentation's ``--max-seed-px`` downscale).
"""

from __future__ import annotations

from .backgrounds import make_background
from .blend import place_seed
from .scene import generate_scene
from .transforms import _resize_to_long_side, transform_seed

__all__ = [
    "generate_scene",
    "transform_seed",
    "make_background",
    "place_seed",
    "_resize_to_long_side",
]
