#!/usr/bin/env python3
"""
seedgen.py - backward-compatible entry point for MultiSeedGen.

The implementation now lives in the ``multiseedgen`` package (split out of this
file for maintainability). This shim preserves the original interface so that

    python seedgen.py --sources ... --out ...

keeps working unchanged. Equivalent to ``python -m multiseedgen``. The public
API is also re-exported, so ``import seedgen; seedgen.run(seedgen.parse_args())``
and ``from seedgen import Config`` continue to work.

See the package modules for the code:
    multiseedgen/segmentation.py   segmentation cascade + rembg fallback + cache
    multiseedgen/compositing.py    transforms, backgrounds, placement, scenes
    multiseedgen/degrade.py        camera-matched photometric degradation
    multiseedgen/export.py         YOLO / YOLO-seg / COCO / data.yaml / manifest
    multiseedgen/qa.py             QA charts, montages, previews, seed checker
    multiseedgen/pipeline.py       deterministic scene workers + run() driver
    multiseedgen/cliconfig.py      argparse + --config loading
    multiseedgen/models.py         Config / Seed / SegResult dataclasses
"""
from __future__ import annotations

from multiseedgen import (  # noqa: F401  (re-exported for backwards compat)
    Config,
    Seed,
    SegResult,
    __version__,
    build_parser,
    parse_args,
    run,
)

if __name__ == "__main__":
    run(parse_args())
