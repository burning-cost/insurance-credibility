# Changelog

## v0.1.5 (2026-03-22)

- security: pin `pyasn1>=0.6.3` to fix CVE-2026-30922 (HIGH severity DoS via unbounded recursion in pyasn1 <= 0.6.2)

## v0.1.4 (2026-03-22) [unreleased]
- feat: add Databricks benchmark notebook
- fix: use plain string license field for universal setuptools compatibility
- docs: regenerate API reference [skip ci]
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)
- docs: regenerate API reference [skip ci]
- fix: sync __version__ with pyproject.toml (0.1.0 -> 0.1.4)

## v0.1.4 (2026-03-21)
- docs: replace pip install with uv add in README
- Add blog post link and community CTA to README
- Add MIT license
- docs: add Bühlmann-Straub benchmark results to README
- docs: regenerate API reference [skip ci]
- Fix P0 recursion bug in DynamicPoissonGammaModel; QA batch 9 fixes
- Add Colab quickstart notebook and Open in Colab badge
- refresh benchmark numbers post-P0 fixes
- docs: regenerate API reference [skip ci]
- Fix P0 and P1 bugs from code review; bump to 0.1.3
- docs: regenerate API reference [skip ci]
- Add standalone benchmark script
- fix: remove scipy<1.11 upper bound — incompatible with Python 3.12
- Add shields.io badge row (PyPI, Python, Tests, License)
- docs: add Databricks notebook link
- Fix: relax scipy constraint to >=1.9,<1.11 for Databricks serverless compat
- Add Related Libraries section to README
- Merge branch 'master' of https://github.com/burning-cost/insurance-credibility
