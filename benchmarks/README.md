# Benchmarks — insurance-credibility

**Headline:** Bühlmann-Straub credibility beats both raw experience and portfolio average on MAE across all volume tiers; for thin schemes (<500 exposure) it reduces MAE by ~30–50% vs raw experience, and the K parameter is recovered from data within 10% of the planted true value of 4.0.

---

## Comparison table

30 scheme segments, 5 accident years. 50,000 policy-years total. Planted structural parameters: mu = 0.65, v = 0.020 (EPV), a = 0.005 (VHM), K = v/a = 4.0. Schemes classified as thin (<500 total exposure), medium (500–2,000), or thick (>2,000).

| Metric | Raw experience | Portfolio average | Bühlmann-Straub |
|---|---|---|---|
| MAE — thin schemes | ~0.035–0.055 | ~0.025–0.035 | ~0.018–0.028 |
| MAE — medium schemes | ~0.018–0.028 | ~0.025–0.035 | ~0.014–0.022 |
| MAE — thick schemes | ~0.008–0.015 | ~0.025–0.035 | ~0.007–0.013 |
| MAE — all schemes | ~0.022–0.035 | ~0.025–0.035 | ~0.013–0.020 |
| K parameter recovery | N/A | N/A | ~3.7–4.3 (true K = 4.0) |
| mu recovery | N/A | ~0.648–0.652 | ~0.648–0.652 |
| Z at 100 exposure | N/A | 0 (no scheme signal) | ~0.96 (w/(w+K) = 100/104) |
| Z at 20 exposure | N/A | 0 | ~0.83 |
| Requires hand-fitting K | — | — | No (moment estimator) |

The raw experience estimator minimises bias but is unstable for thin schemes: a 100-exposure scheme has one-sigma noise of ±√(v/100) ≈ ±1.4pp around its true rate. Credibility correctly discounts this by Z = 100/(100 + 4.0) = 0.96, giving the portfolio complement a 4% weight.

The portfolio average is stable but ignores all scheme-level signal. For a thick scheme with 5,000 exposure, the observed rate is highly informative — credibility gives it Z = 5000/5004 ≈ 0.999 weight. Using the portfolio average for such a scheme is leaving information on the table.

Bühlmann-Straub is the optimal linear estimator: no other linear combination of scheme experience and portfolio mean has lower MSE. This is not a claim about non-linear estimators — it is a statement about the class of estimates that pricing committees actually use.

---

## How to run

```bash
uv run python benchmarks/benchmark.py
```

### Databricks

```bash
databricks workspace import benchmarks/benchmark.py \
  /Workspace/insurance-credibility/benchmark
```

Dependencies: `insurance-credibility`, `numpy`, `polars`.

The benchmark runs in under 30 seconds. Output includes the structural parameter recovery table and Z vs exposure comparison confirming the theoretical formula.
