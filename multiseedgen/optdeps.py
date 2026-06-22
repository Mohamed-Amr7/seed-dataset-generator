from __future__ import annotations
"""Import-guarded optional dependencies (joblib, pyyaml, tqdm)."""


# ---- optional dependencies (import-guarded; the numpy+opencv path never needs them) ----
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:                       # pragma: no cover - environment dependent
    _HAVE_JOBLIB = False


try:
    import yaml
    _HAVE_YAML = True
except Exception:                       # pragma: no cover
    _HAVE_YAML = False



def _tqdm(iterable, **kwargs):
    """tqdm wrapper that degrades to a no-op identity when tqdm is absent."""
    try:
        import tqdm as _t
        return _t.tqdm(iterable, **kwargs)
    except Exception:                   # pragma: no cover
        return iterable

# Always-defined fallbacks so siblings can import these names
try:
    Parallel
except NameError:
    Parallel = None
try:
    delayed
except NameError:
    delayed = None
try:
    yaml
except NameError:
    yaml = None
