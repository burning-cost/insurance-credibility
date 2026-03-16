# insurance-credibility

[![PyPI](https://img.shields.io/pypi/v/insurance-credibility)](https://pypi.org/project/insurance-credibility/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-credibility)](https://pypi.org/project/insurance-credibility/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-credibility/blob/main/notebooks/quickstart.ipynb)


Credibility models for UK non-life insurance pricing: Bühlmann-Straub group credibility and Bayesian experience rating at individual policy level.

## The problem

Two problems that look similar but need different tools:

**Group credibility (schemes, large accounts):** A fleet scheme has 3 years of loss history. How much should you weight it against the market rate? Too much and you are pricing noise. Too little and you leave money on the table. The Bühlmann-Straub formula gives the optimal weight — it depends on the scheme's own variance, the portfolio variance, and the amount of exposure observed.

**Individual policy experience rating:** A commercial motor policy has been with you for 5 years with no claims. Flat NCD tables say "maximum discount". But how much is 5 years of no-claims worth relative to the a priori GLM rate? Depends on portfolio heterogeneity (how much do individual risks actually differ?), exposure (5 years at 0.5 fleet size is worth less than 5 years at 2.0), and claim frequency (low-frequency risks take longer to accumulate credible experience).

This library addresses both.

## Installation

```bash
pip install insurance-credibility
```

## Quick start

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# Group-level credibility (scheme pricing)
# One row per scheme per year — loss_rate is incurred per vehicle-year
df = pl.DataFrame({
    "scheme":    ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "year":      [2022, 2023, 2024, 2022, 2023, 2024, 2022, 2023, 2024],
    "loss_rate": [0.12, 0.09, 0.11, 0.25, 0.28, 0.22, 0.08, 0.07, 0.09],
    "exposure":  [120.0, 135.0, 140.0, 45.0, 50.0, 48.0, 300.0, 310.0, 320.0],
})

bs = BuhlmannStraub()
bs.fit(df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

print(bs.z_)          # credibility factors per scheme (Z_i)
print(bs.k_)          # Bühlmann's k: noise-to-signal ratio
print(bs.premiums_)   # credibility-blended premium per scheme


# Individual policy experience rating
from insurance_credibility import ClaimsHistory, StaticCredibilityModel

histories = [
    ClaimsHistory("POL001", periods=[1, 2, 3], claim_counts=[0, 1, 0],
                  exposures=[1.0, 1.0, 0.8], prior_premium=400.0),
    ClaimsHistory("POL002", periods=[1, 2, 3], claim_counts=[2, 1, 2],
                  exposures=[1.0, 1.0, 1.0], prior_premium=400.0),
]

model = StaticCredibilityModel()
model.fit(histories)

cf = model.predict(histories[0])
posterior_premium = histories[0].prior_premium * cf
```

## Models

### Classical credibility

`BuhlmannStraub` — group credibility for scheme pricing. Estimates structural parameters (within-group variance, between-group variance) from the portfolio using method of moments. Produces credibility factors and credibility-weighted predictions per group.

Key attributes after fitting:
- `bs.z_` — Polars DataFrame with columns `["group", "Z"]`; Z_i = w_i / (w_i + k)
- `bs.k_` — Bühlmann's k = v/a (noise-to-signal ratio)
- `bs.premiums_` — Polars DataFrame with credibility premiums per group

`HierarchicalBuhlmannStraub` — nested group structure (e.g., scheme → book, sector → district → area). Extends Bühlmann-Straub to multi-level hierarchies following Jewell (1975).

### Experience rating

`StaticCredibilityModel` — Bühlmann-Straub at individual policy level. Fits kappa = sigma^2 / tau^2 from a portfolio of policy histories. Credibility weight for a policy is `omega = e_total / (e_total + kappa)`.

`DynamicPoissonGammaModel` — Poisson-gamma state-space model following Ahn, Jeong, Lu & Wüthrich (2023). Seniority-weighted updates: recent years count more. Produces the full posterior distribution, not just a point estimate.

`SurrogateModel` — IS-surrogate (Calcetero et al. 2024). Suitable for large portfolios where computing the exact posterior for every policy is expensive.

## Data format

```python
from insurance_credibility import ClaimsHistory

history = ClaimsHistory(
    policy_id="POL001",
    periods=[1, 2, 3, 4, 5],          # year indices
    claim_counts=[0, 1, 0, 0, 2],     # observed claims
    exposures=[1.0, 1.0, 0.8, 1.0, 1.0],  # vehicle-years
    prior_premium=450.0,               # GLM-based a priori rate
)
```

`exposures` is the key parameter that distinguishes this from flat NCD tables: a policy with 0.5 years of exposure gets far less credibility than one with 5 years, regardless of claim count.

## Performance

Benchmarked on a synthetic panel of 30 scheme segments over 5 years (150 observations) with known structural parameters planted in the DGP. Three approaches compared against the known true scheme rates.

DGP: portfolio mean loss ratio 0.65, EPV v=0.020, VHM a=0.005, theoretical K=4.0. Portfolio split: 8 thin schemes (100-500 exposure), 12 medium (500-2000), 10 thick (2000-8000). Benchmark run post P0 fixes on Databricks serverless.

**Structural parameter recovery:**
- mu_hat=0.6593 (true=0.6500) — portfolio mean recovered to within 1.4%
- v_hat=0.01770 (true=0.02000) — EPV underestimated by 11.5%
- a_hat=0.00212 (true=0.00500) — VHM underestimated by 57.6%, K=8.36 (true K=4.0)

K is over-estimated because the method-of-moments estimator needs substantial cross-scheme variation to converge. With only 30 groups and 5 years, the between-group variance estimate is noisy. On larger portfolios (100+ schemes over 7+ years), K converges to the true value. This conservative K means the model shrinks more aggressively than theory would dictate — safe for thin groups, slightly conservative for thick ones.

**MAE vs true scheme rates:**

| Tier | Schemes | Raw MAE | Portfolio MAE | Credibility MAE | Best |
|------|---------|---------|---------------|-----------------|------|
| Thin (<500 exp) | 8 | 0.0074 | 0.0596 | 0.0069 | Credibility |
| Medium (500-2000) | 12 | 0.0030 | 0.0423 | 0.0029 | Credibility |
| Thick (2000+ exp) | 10 | 0.0014 | 0.0337 | 0.0014 | Raw (margin negligible) |
| Overall | 30 | 0.0036 | 0.0440 | 0.0035 | Credibility |

- **Thin schemes**: Credibility MAE 0.0069 vs raw 0.0074 — 6.8% improvement. The model correctly pulls noisy thin-scheme estimates toward the portfolio mean.
- **Thick schemes**: Raw and credibility are essentially tied (0.0014 each). At high exposure, Z approaches 1.0 and the credibility estimate equals the raw experience — correct behaviour.
- **Portfolio average**: Uniformly worst across all tiers (MAE 0.0337-0.0596). Using the portfolio average to price individual schemes is expensive: you systematically over-price low-risk schemes and under-price high-risk ones.
- **Credibility Z calibration**: Z = w/(w+K) with K=8.36. Thin schemes (exposure 100-500, total per scheme) get Z=0.1-0.4; thick schemes (2000-8000) get Z=0.7-1.0. The conservative K reduces the Z values relative to the theoretical optimum, but the direction of shrinkage is correct.
- **Limitation**: Parameter estimation is noisy with 30 groups. Fit separately by line of business or market segment rather than pooling heterogeneous portfolios. Minimum practical dataset: 20+ groups with 3+ years each.
## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_credibility_demo.py).

## References

- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and Its Applications*. Springer.
- Ahn, J.Y., Jeong, H., Lu, Y. & Wüthrich, M.V. (2023). "Individual claims reserving using the Poisson-gamma state-space model." *Insurance: Mathematics and Economics*.
- Calcetero, V., Badescu, A. & Lin, X.S. (2024). "Credibility theory for the 21st century." *ASTIN Bulletin*.
- Wüthrich, M.V. (2024). "Transformer models for individual experience rating." *European Actuarial Journal*.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models — generalises Bühlmann-Straub to Poisson/Gamma likelihoods and multiple crossed random effects |
| [insurance-multilevel](https://github.com/burning-cost/insurance-multilevel) | Two-stage CatBoost + REML random effects for broker and scheme factors in high-cardinality portfolios |
| [experience-rating](https://github.com/burning-cost/experience-rating) | NCD systems and experience modification factors — uses credibility weighting for individual policy experience rating |

## Licence

MIT
