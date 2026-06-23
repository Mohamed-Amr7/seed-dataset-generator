"""Web service helpers: build the form schema by introspecting the CLI parser + the
Config model, and load the named presets. Kept out of ``app.py`` so the route handlers
stay thin and the form-building logic is testable on its own (see test_schema_snapshot).
"""

from __future__ import annotations

from pathlib import Path

from ..cliconfig import _config_defaults, _load_config_file, build_parser
from ..models import Config

ROOT = Path(__file__).resolve().parents[2]  # the MultiSeedGen project dir
_PARSER = build_parser()

# Path fields that should render a "Browse…" picker in the form, and whether the
# picker selects a directory (sources/out/...) or a file (segment_map/sam_checkpoint).
# 'config' is intentionally absent — the schema never emits it as a form field.
_PATH_KIND = {
    "sources": "dir",
    "out": "dir",
    "textures": "dir",
    "cache_dir": "dir",
    "segment_map": "file",
    "sam_checkpoint": "file",
}

# Production-quality overlay on top of the bare Config defaults; mirrors the
# "Global recommendations" in docs/RECOMMENDED_SETTINGS.md so a fresh form is ready to
# generate a quality dataset after only setting `sources` and `out`.
_RECOMMENDED = {
    "min_seeds": 8,
    "max_seeds": 30,
    "scale_min": 0.05,
    "scale_max": 0.16,
    "bg_from_sources": True,
    "neg_frac": 0.10,
    "segment": "auto",
    "rembg_provider": "auto",
    "fmt": "both",
    "masks": True,
    "degrade": True,
    "noise_sigma": 3.0,
    "jpeg_quality_min": 80,
    "jpeg_quality_max": 95,
    "blur_max_sigma": 0.5,
    "shadow_dir_mode": "scene",
    "shadow_opacity": 0.25,
    "edge_feather": 0.5,
    "qa_report": True,
    "visualize_count": 12,
    "workers": 0,
    "parallel_backend": "auto",
    "seed": 7,
}

# Named recipes, sourced from configs/*.yaml (single source of truth with the CLI).
_PRESET_FILES = [
    ("Root — clean, 27 species", "root.yaml"),
    ("Wheat — tiny, low-contrast", "wheat.yaml"),
    ("Test — hardest (polygons)", "test.yaml"),
    ("Soybean — Healthy (single class)", "soybean_healthy.yaml"),
]


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    return v


def _action_widget(action) -> dict:
    cls = type(action).__name__
    if cls in ("_StoreTrueAction", "_StoreFalseAction"):
        widget = "bool"
    elif action.choices:
        widget = "choice"
    elif action.type is int:
        widget = "int"
    elif action.type is float:
        widget = "float"
    else:
        widget = "text"
    return {
        "widget": widget,
        "choices": list(action.choices) if action.choices else None,
        "help": action.help or "",
        "flag": action.option_strings[0] if action.option_strings else "",
    }


def _load_presets() -> dict:
    """{"Recommended (general)": {...}, <dataset recipes from configs/*.yaml>}.
    Any missing/broken config is skipped so this can never 500."""
    presets = {"Recommended (general)": dict(_RECOMMENDED)}
    for label, fname in _PRESET_FILES:
        p = ROOT / "configs" / fname
        try:
            if p.exists():
                presets[label] = {k: _jsonable(v) for k, v in _load_config_file(p).items()}
        except Exception:
            continue
    return presets


def build_schema() -> dict:
    """Build the form descriptor: ordered field groups + recommended overlay + presets.
    The route adds the environment-specific ``data_root`` / ``project_root``."""
    defaults = _config_defaults()
    field_names = set(Config.model_fields)
    # map dest -> argparse action metadata
    meta: dict = {}
    for grp in _PARSER._action_groups:
        for act in grp._group_actions:
            if act.dest in ("help", "config"):
                continue
            if act.dest in field_names:
                meta.setdefault(act.dest, (grp.title, _action_widget(act)))

    groups: dict = {}
    for dest, (title, info) in meta.items():
        groups.setdefault(title, []).append(
            {
                "name": dest,
                "default": _jsonable(defaults.get(dest)),
                "path_kind": _PATH_KIND.get(dest),  # 'dir'|'file' -> render Browse…; else None
                **info,
            }
        )
    # any Config field with no CLI flag -> a generic group
    for name in Config.model_fields:
        if name not in meta and name != "config":
            groups.setdefault("other", []).append(
                {
                    "name": name,
                    "default": _jsonable(defaults.get(name)),
                    "path_kind": _PATH_KIND.get(name),
                    "widget": "text",
                    "choices": None,
                    "help": "",
                    "flag": "",
                }
            )
    order = [
        "I/O",
        "scene composition",
        "per-seed augmentation",
        "drop shadow",
        "backgrounds",
        "segmentation",
        "realism",
        "performance",
        "output",
        "data split",
        "sampling",
        "export",
        "other",
    ]
    out = []
    for title in order:
        if title in groups:
            out.append({"title": title, "fields": groups[title]})
    for title, fields in groups.items():  # any unordered groups
        if title not in order:
            out.append({"title": title, "fields": fields})
    return {"groups": out, "recommended": _RECOMMENDED, "presets": _load_presets()}
