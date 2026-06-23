"""Exception hierarchy for MultiSeedGen.

These replace the library's former ``sys.exit()`` calls. ``sys.exit`` deep inside the
pipeline made the code unusable as a library and forced the web layer to catch
``SystemExit``. Raising typed exceptions instead lets callers (tests, the web service)
handle failures normally; the CLI entry points catch :class:`MultiSeedGenError` and
translate it back into the old user-facing behavior (a clean stderr message + non-zero
exit), so nothing changes for command-line users.
"""

from __future__ import annotations


class MultiSeedGenError(Exception):
    """Base class for expected, user-facing MultiSeedGen errors.

    Caught at the CLI entry points and re-raised as ``SystemExit(message)``.
    """


class ConfigError(MultiSeedGenError):
    """Invalid configuration: bad CLI/file values, unknown keys, or missing inputs."""


class SplitError(MultiSeedGenError):
    """The requested train/val split is unsatisfiable (e.g. no seeds for a split)."""


class SegmentationError(MultiSeedGenError):
    """Segmentation produced no usable cut-outs from the sources."""
