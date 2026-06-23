"""MultiSeedGen - synthetic multi-seed detection dataset generator."""

from __future__ import annotations

from .cliconfig import build_parser, parse_args
from .constants import SEG_VERSION, __version__
from .models import Config, Seed, SegResult
from .pipeline import run

__all__ = [
    "__version__",
    "SEG_VERSION",
    "Config",
    "Seed",
    "SegResult",
    "parse_args",
    "build_parser",
    "run",
]
