# Contributing

This is a small, single-maintainer project, but it holds a strict reproducibility
contract (see [architecture.md](architecture.md)). The tooling below exists to keep that
contract enforced automatically.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add web/rembg/sam if you touch those paths
pre-commit install               # runs ruff + mypy on every commit
```

## The regression gate

Run this before every commit — it is exactly what CI runs:

```bash
ruff check .                     # lint
ruff format --check .            # formatting
mypy multiseedgen                # type check
pytest -m "not gpu"              # tests (GPU-only tests skipped off a CUDA host)
```

All four must be green. The single most important check is `pytest` staying green on
`tests/test_determinism.py` and `tests/test_golden.py`: those guard the byte-identity
contract. A diff there means your change altered generated pixels.

## Tests

| Location | Scope | Speed |
|----------|-------|-------|
| `tests/unit/` | isolated single-function units | sub-second |
| `tests/` | integration & contract (determinism, golden, config/schema/cache snapshots, web) | seconds |

Conventions for new tests:

- **Arrange–Act–Assert**, one behaviour per test, descriptive `test_<unit>_<behaviour>` names.
- Keep units **isolated and deterministic** — no network, no GPU, fixed PRNG seeds. Use
  the fixtures in `tests/unit/conftest.py` (`make_cfg`, `seed_cutout`, `seed_on_uniform_bg`, `rng`).
- Use `pytest.mark.parametrize` for matrices (methods, kinds, invalid inputs).
- Mark slow/end-to-end tests `@pytest.mark.slow`, GPU tests `@pytest.mark.gpu`,
  host-pinned byte tests `@pytest.mark.golden`.

Markers are strict (`--strict-markers`); a typo'd marker fails collection.

### Regenerating golden manifests

The golden byte manifests are pinned to this host's `numpy`/`opencv` versions. After a
**reviewed, intentional** change to those deps (or a deliberate pixel-affecting change),
regenerate and commit the new manifests:

```bash
UPDATE_GOLDEN=1 pytest tests/test_golden.py
```

Never regenerate to "make the test pass" — a golden diff you didn't expect is a real
regression in the determinism contract.

## Coding conventions

- **Config is the single source of truth.** New options go in `config.py` (with a
  `Literal` for choices and a range check that *raises* if needed), then wired into the
  CLI table and the web schema. Validators must never mutate gated values (see the
  determinism rules in [architecture.md](architecture.md)).
- **No `sys.exit()` / `print()` in library code.** Raise a `MultiSeedGenError` subclass;
  log via `logging.getLogger(__name__)`.
- **Preserve RNG draw order** in any compositing/degradation change, and keep the
  segmentation cache key (`SEG_VERSION`, `_seg_params_signature`) stable unless you
  intend to invalidate caches.
- Line length 100; `ruff format` is authoritative. Public re-exports are the only
  sanctioned unused imports (already allow-listed in `pyproject.toml`).
