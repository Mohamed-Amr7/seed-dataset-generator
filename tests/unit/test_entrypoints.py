"""Unit tests for the CLI entry point and logging setup.

These cover the thin glue that turns the library into a command-line tool: stdout log
configuration (idempotent, library-modules-never-configure) and ``main()`` translating
a typed :class:`MultiSeedGenError` into a clean ``SystemExit`` instead of a traceback.
"""

from __future__ import annotations

import logging
import sys

import pytest

from multiseedgen.__main__ import main
from multiseedgen.logging_setup import configure_logging


@pytest.fixture
def restore_msg_logger():
    """Snapshot/restore the ``multiseedgen`` logger so configuring it in a test does not
    leak handlers into the rest of the suite."""
    logger = logging.getLogger("multiseedgen")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers = []
    try:
        yield logger
    finally:
        logger.handlers, logger.level, logger.propagate = saved


def test_configure_logging_is_idempotent(restore_msg_logger):
    """Calling it twice attaches exactly one tagged stdout handler, never stacks."""
    configure_logging("INFO")
    configure_logging("DEBUG")

    tagged = [h for h in restore_msg_logger.handlers if getattr(h, "_multiseedgen_handler", False)]
    assert len(tagged) == 1
    assert restore_msg_logger.propagate is False


def test_configure_logging_respects_env_level(monkeypatch, restore_msg_logger):
    monkeypatch.setenv("MULTISEEDGEN_LOG_LEVEL", "WARNING")
    configure_logging()
    assert restore_msg_logger.level == logging.WARNING


def test_main_translates_config_error_to_systemexit(monkeypatch, restore_msg_logger):
    """A missing required arg surfaces as a clean SystemExit (non-zero), not a stack
    trace - the historical CLI contract."""
    monkeypatch.setattr(sys, "argv", ["seedgen"])  # no --sources / --out

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code  # truthy -> non-zero / message string
