"""Pytest configuration.

The package is imported as an *installed* distribution - run ``pip install -e .[dev]``
once before the test suite (CI does this). The previous ``sys.path`` bootstrap is gone:
relying on an editable install keeps test imports identical to real usage and lets
tox/nox/CI work without per-directory path tricks.
"""
