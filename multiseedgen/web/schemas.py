"""Pydantic request models for the web API.

These replace the former untyped ``payload: dict`` endpoint signatures, so FastAPI
validates the body shape up front and returns a 422 with field detail instead of the
handler blowing up with a 500 on a malformed body. The ``config`` field stays a free
``dict`` on purpose: it is round-tripped to the CLI (`python -m multiseedgen --config`),
which is the single source of truth for validating individual config keys.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartRunRequest(BaseModel):
    """Body for POST /api/runs."""

    config: dict = Field(default_factory=dict)


class SegCheckRequest(BaseModel):
    """Body for POST /api/seg/check (the segmentation tuner)."""

    config: dict = Field(default_factory=dict)
    n: int = 24  # number of checkerboard cut-outs to render
    out: str | None = None  # optional output dir (sandboxed; defaults to a temp dir)


class ConfigSaveRequest(BaseModel):
    """Body for POST /api/config/save."""

    path: str | None = None
    config: dict = Field(default_factory=dict)
