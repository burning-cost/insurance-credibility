# insurance-credibility

[![PyPI](https://img.shields.io/pypi/v/insurance-credibility)](https://pypi.org/project/insurance-credibility/) [![Python](https://img.shields.io/pypi/pyversions/insurance-credibility)](https://pypi.org/project/insurance-credibility/) [![Tests](https://github.com/burning-cost/insurance-credibility/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-credibility/actions/workflows/tests.yml) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-credibility/blob/main/LICENSE) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-credibility/blob/main/notebooks/quickstart.ipynb) [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-credibility/blob/main/notebooks/quickstart.ipynb)

---

## The problem

Small segments have unstable loss experience. A fleet scheme with 200 vehicle-years has a loss ratio that is mostly noise, but ignoring it entirely prices a segment you have genuine data on. How much should you trust the scheme's own history versus the portfolio average?

The same question arises at individual policy level: a commercial motor policy with 5 years of no-claims history deserves a discount, but how large? Flat NCD tables assign the same maximum discount regardless of policy size or the underlying claim frequency — a 0.5-vehicle-year policy gets the same credit as a 50-vehicle-year fleet.

**Blog post:** [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](https://burning-cost.github.io/2026/02/19/buhlmann-straub-credibility-in-python/)

---

## Why this library?

Bühlmann-Straub is the actuarial standard for this problem — a statistically optimal blend of segment experience with the portfolio mean, weighted by earned exposure. Most existing implementations assume non-insurance data structures: equal group sizes, no exposure weights, no distinction between within-group and between-group variance.

This library is built for insurance: it handles unequal exposures, nested hierarchies (scheme → book, district → area), and individual policy experience rating in a consistent framework.

---

## Compared to alternatives

| | Manual credibility weights | Random effects GLM | Hierarchical Bayes | **insurance-credibility** |
|---|---|---|---|---|
| Statistically optimal blend | No (rule-of-thumb) | Yes | Yes | Yes (B-S formula) |
| No prior specification needed | Yes | Yes | No | Yes |
| Handles unequal exposures | Manual | Yes | Yes | Yes |
| Nested group hierarchies | Manual | Partial | Yes | Yes (`HierarchicalBuhlmannStraub`) |
| Individual policy experience rating | No | No | Partial | Yes |
| Closed-form, < 1 second | Yes (simple) | No | No | Yes |
| Full posterior distribution | No | No | Yes | Yes (`DynamicPoissonGammaModel`) |

---

## Quickstart

```bash
uv add insurance-credibility
```

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# One row per scheme per year
df = pl.DataFrame({
    "scheme":    ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "year":      [2022, 2023, 2024, 2022, 2023, 2024, 2022, 2023, 2024],
    "loss_rate": [0.12, 0.09, 0.11, 0.25, 0.28, 0.22, 0.08, 0.07, 0.09],
    "exposure":  [120.0, 135.0, 140.0, 45.0, 50.0, 48.0, 300.0, 310.0, 320.0],
})

bs = BuhlmannStraub()
bs.fit(df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

print(bs.z_)         # credibility factors per scheme (Z_i = w_i / (w_i + k))
print(bs.k_)         # Bühlmann's k: noise-to-signal ratio
print(bs.premiums_)  # credibility-blended premium per scheme
```

---

## Group credibility: schemes and large accounts

`BuhlmannStraub` fits structural parameters — within-group variance (v) and between-group variance (a) — from the portfolio using method of moments. It then computes the credibility factor Z_i for each group:

```
Z_i = w_i / (w_i + k)    where k = v/a
```

Z approaches 1.0 as exposure grows — thick schemes are trusted almost entirely. Z shrinks toward 0 on thin schemes — the portfolio mean gets most of the weight. On a 30-scheme, 5-year benchmark with known true parameters:

| Tier | Raw MAE | Portfolio avg MAE | Credibility MAE |
|---|---|---|---|
| Thin (< 500 exposure) | 0.0074 | 0.0596 | **0.0069** |
| Medium (500–2000) | 0.0030 | 0.0423 | **0.0029** |
| Thick (2000+) | 0.0014 | 0.0337 | 0.0014 (tie) |

Credibility beats raw experience on thin and medium tiers. On thick tiers, Z approaches 1.0 and the two methods converge — which is correct behaviour.

`HierarchicalBuhlmannStraub` extends this to nested group structures: scheme → book, sector → district → area. Following Jewell (1975).

---

## Individual policy experience rating

For commercial motor and fleet pricing, where you want to move individual policies away from the GLM rate based on their own claims history:

```python
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

`exposures` is the key parameter that distinguishes this from flat NCD tables: a policy with 0.5 years of exposure gets far less credibility than one with 5 years, regardless of claim count.

---

## Model tiers

**`StaticCredibilityModel`** — Bühlmann-Straub at individual policy level. Fits kappa = sigma^2 / tau^2 from a portfolio of policy histories. Credibility weight for a policy is `omega = e_total / (e_total + kappa)`. Closed-form, fast, suitable for production.

**`DynamicPoissonGammaModel`** — Poisson-gamma state-space model following Ahn, Jeong, Lu & Wüthrich (2023). Seniority-weighted updates: recent years count more. Produces the full posterior distribution per policy, not just a point estimate — useful when communicating uncertainty to a pricing committee or reinsurer.

**`SurrogateModel`** — IS-surrogate (Calcetero et al. 2024). For large portfolios where computing the exact posterior for every policy is expensive.

---

## Structural parameter recovery

On a 30-group, 5-year benchmark with known true parameters (mu=0.650, v=0.020, a=0.005, k=4.0):
- mu recovered within 1.4%
- k recovered within factor of 2 (conservative shrinkage direction)

k is over-estimated in small samples — a known property of the method-of-moments estimator. Conservative shrinkage is safe: it means you trust thin segments slightly less than the theory would dictate. On portfolios with 100+ groups over 7+ years, k converges to the true value.

Full validation: `notebooks/databricks_validation.py`.

---

## Bühlmann-Straub vs random effects GLM

The actuarial credibility approach and the random effects GLM (e.g. `statsmodels` MixedLM) estimate the same quantity under a Gaussian approximation. The differences are practical:

- Bühlmann-Straub is closed-form and fits in under a second on a 150-row scheme panel. No iteration, no convergence issues.
- Random effects GLM requires a correctly specified likelihood and converges slowly on unbalanced panels with many groups.
- Bühlmann-Straub exposes the structural parameters (mu, v, a, k) directly, making them easy to inspect and challenge.

For Poisson-Gamma likelihoods and non-Gaussian random effects, use `DynamicPoissonGammaModel`.

---

## Limitations

- Structural parameter estimation (v, a) requires at least 30–50 groups and 3+ years to converge reliably. On the 30-group benchmark, VHM was underestimated by 57.6%. In thin portfolios, treat credibility factors as directional and apply a floor on Z.
- `StaticCredibilityModel` assumes homoscedastic within-policy variance. Segment by policy size tier on portfolios with large fleets alongside small ones.
- Kappa estimation needs at least 50–100 policies with 2+ years of history. Below this, the estimate is unreliable.
- Structural parameters must be refitted as portfolio composition changes. Stale kappa from a different historical book produces miscalibrated experience adjustments.

---

## Part of the Burning Cost stack

Takes segment-level experience data: earned exposure, observed loss ratios, scheme panels. Feeds credibility-weighted estimates into [insurance-gam](https://github.com/burning-cost/insurance-gam) (as adjusted targets for tariff fitting). [See the full stack](https://burning-cost.github.io/stack/)

| Library | Description |
|---|---|
| [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) | Whittaker-Henderson smoothing — smooths the raw experience rates that credibility weighting then blends |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Interpretable GAMs — credibility-adjusted targets as input to tariff fitting |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals — uncertainty quantification for credibility-blended estimates |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — monitors whether credibility parameters remain valid |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — sign-off pack for credibility models |

---

## References

- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and Its Applications*. Springer.
- Jewell, W.S. (1975). "Multidimensional Credibility." *Operations Research*, 23(5), 904–920.
- Ahn, J.Y., Jeong, H., Lu, Y. & Wüthrich, M.V. (2023). "Dynamic Bayesian Credibility." arXiv:2308.16058.
- Calcetero, V., Badescu, A. & Lin, X.S. (2024). "Credibility theory for the 21st century." *ASTIN Bulletin*.

---

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-credibility/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-credibility/issues)
- **Blog and tutorials:** [burning-cost.github.io](https://burning-cost.github.io)
- **Training course:** [Insurance Pricing in Python](https://burning-cost.github.io/course) — Module 6 covers credibility theory. £97 one-time.

## Licence

BSD-3-Clause
