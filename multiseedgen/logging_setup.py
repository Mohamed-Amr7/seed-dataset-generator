"""Logging configuration for the CLI / web entry points.

Library modules only ever call ``logging.getLogger(__name__)`` and never touch the
handler/level config, so importing the package is side-effect-free and parallel (loky)
workers never double-configure logging. The CLI entry point calls
:func:`configure_logging` exactly once to route records to **stdout** - the old code
printed progress to stdout and the web UI streams the subprocess's stdout, so stdout is
preserved. The level can be overridden with the ``MULTISEEDGEN_LOG_LEVEL`` env var
(e.g. ``DEBUG`` / ``WARNING``).
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_LEVEL = "INFO"
_HANDLER_TAG = "_multiseedgen_handler"


def configure_logging(level: str | int | None = None) -> None:
    """Attach a single stdout handler to the ``multiseedgen`` logger. Idempotent.

    Records are emitted as the bare message (no level/time prefix) to match the exact
    text the pipeline used to ``print()`` - the web UI parses the ``[1/3]`` progress
    lines, so their wording is preserved.
    """
    if level is None:
        level = os.environ.get("MULTISEEDGEN_LOG_LEVEL", _DEFAULT_LEVEL)
    logger = logging.getLogger("multiseedgen")
    logger.setLevel(level)
    if any(getattr(h, _HANDLER_TAG, False) for h in logger.handlers):
        return  # already configured
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _HANDLER_TAG, True)
    logger.addHandler(handler)
    logger.propagate = False  # don't double-emit through the root logger
