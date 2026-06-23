"""Domain dataclasses: Seed, SegResult, and the QA-stats accumulator.

The validated run ``Config`` now lives in ``config.py``; it is re-exported here so the
historical ``from multiseedgen.models import Config`` keeps working. ``Seed`` /
``SegResult`` stay plain dataclasses on purpose: they carry raw numpy arrays that are
``np.save``'d / memory-mapped and pickled into loky workers - they are hot data
carriers, not configuration, and must not pay validation/copy overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

__all__ = ["Config", "Seed", "SegResult"]


# --------------------------------------------------------------------------- #
# Source loading                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Seed:
    rgba: np.ndarray  # BGRA uint8
    class_id: int
    source: str = ""  # source-file rel path (provenance / extra_annotations)
    method: str = ""  # segmentation method that produced this cut-out
    split: str = "both"  # "train" | "val" | "both" (seed-level hold-out)


@dataclass
class SegResult:
    patch: np.ndarray | None
    report: dict
    source: str
    class_id: int
    background: np.ndarray | None = None
    cache_hit: bool = False


# --------------------------------------------------------------------------- #
# Dataset QA accumulation (charts/report written by write_qa_report; see M6)  #
# --------------------------------------------------------------------------- #
class _QAStats:
    def __init__(self):
        self.boxes_per_image: list[int] = []
        self.sizes: list[float] = []
        self.aspects: list[float] = []
        self.class_counts: dict[int, int] = {}
        self.occlusion: list[float] = []
        self.negatives = 0

    def add(self, split: str, result) -> None:
        labels = result.labels
        self.boxes_per_image.append(len(labels))
        if not labels:
            self.negatives += 1
        for cls, x0, y0, x1, y1 in labels:
            w, h = (x1 - x0), (y1 - y0)
            self.sizes.append(float(max(w, h)))
            self.aspects.append(float(w) / float(max(h, 1)))
            self.class_counts[int(cls)] = self.class_counts.get(int(cls), 0) + 1
