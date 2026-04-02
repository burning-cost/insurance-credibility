# Changelog

## [0.1.8] - 2026-04-02

### Fixed
- Replaced bare `assert` statements in `DynamicPoissonGammaModel` with proper
  `RuntimeError` and `ValueError` exceptions. Previously, if `p_` or `q_` were
  somehow `None` after fitting, Python raised an opaque `AssertionError`. The
  same applied when `exposures=None` was passed to `_forward_recursion` and
  `_policy_loglik`. These are now explicit errors with actionable messages.


## [0.1.7] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility
  with scipy's use of `numpy.exceptions` (added in numpy 1.25).


## [0.1.6] - 2026-03-22

### Security
- Pinned `pyasn1>=0.6.3` to fix CVE-2026-30922 (HIGH severity DoS via
  unbounded recursion in pyasn1 <= 0.6.2).


## [0.1.5] - 2026-03-21

### Added
- Databricks benchmark notebook (`notebooks/databricks_validation.py`).
- Open in Colab badge and quickstart notebook.
- Shields.io badge row (PyPI, Python, Tests, License).
- Bühlmann-Straub benchmark results added to README.
- Related Libraries section in README.

### Fixed
- P0 recursion bug in `DynamicPoissonGammaModel` (state transition used wrong
  indexing for the initial prior state). Benchmark numbers updated post-fix.
- Removed scipy upper bound `<1.11` — incompatible with Python 3.12.
- `__version__` now sourced from `importlib.metadata` to prevent drift from
  `pyproject.toml`.
- Used plain string `license` field in `pyproject.toml` for universal
  setuptools compatibility.

### Documentation
- Replaced `pip install` with `uv add` in README installation section.
- Added blog post link and community CTA.
- Added MIT license file.
