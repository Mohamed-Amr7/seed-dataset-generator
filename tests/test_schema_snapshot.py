#!/usr/bin/env python3
"""Web ``/api/schema`` contract snapshot.

The form-driving schema endpoint is currently built by introspecting argparse
internals + ``dataclass_fields(Config)``. The Pydantic migration rebuilds it from the
model, and the web split moves it into a router - either can silently drop a field,
mislabel a widget, reorder groups, or break a preset. This pins the *actual emitted*
JSON (minus the two environment-specific absolute-path keys) so the static SPA keeps
getting byte-identical schema across the rework.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")
msg = pytest.importorskip("multiseedgen")

_GOLDEN = Path(__file__).resolve().parent / "golden" / "api_schema.json"
# Absolute, host-specific paths - not part of the form contract.
_ENV_KEYS = ("data_root", "project_root")


def _get_schema() -> dict:
    # Import the submodule lazily via import_module - `multiseedgen.web` re-exports the
    # FastAPI instance as `.app`, which shadows the submodule under attribute access.
    import importlib

    webapp = importlib.import_module("multiseedgen.web.app")
    schema = webapp.schema()  # the route handler returns the dict
    return {k: v for k, v in schema.items() if k not in _ENV_KEYS}


def test_api_schema_snapshot():
    got = _get_schema()
    # Sanity: the shape the SPA depends on is present.
    assert "groups" in got and got["groups"], "schema lost its field groups"
    assert "recommended" in got and "presets" in got

    if os.environ.get("UPDATE_GOLDEN") or not _GOLDEN.exists():
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"wrote schema snapshot {_GOLDEN.name}")

    expected = json.loads(_GOLDEN.read_text())
    assert got == expected, (
        "/api/schema output changed (group order, field widgets/defaults/choices, "
        "recommended, or presets). If intentional, regenerate with UPDATE_GOLDEN=1."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
