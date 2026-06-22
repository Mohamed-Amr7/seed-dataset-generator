"""MultiSeedGen - synthetic multi-seed detection dataset generator."""
from __future__ import annotations
from .constants import __version__, SEG_VERSION
from .models import Config, Seed, SegResult
from .cliconfig import parse_args, build_parser
from .pipeline import run

__all__ = ["__version__", "SEG_VERSION", "Config", "Seed", "SegResult",
           "parse_args", "build_parser", "run"]
