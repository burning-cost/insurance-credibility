# insurance-credibility

**Bühlmann-Straub credibility and Bayesian experience rating for UK insurance pricing teams.**

[![PyPI](https://img.shields.io/pypi/v/insurance-credibility)](https://pypi.org/project/insurance-credibility/) [![Python](https://img.shields.io/pypi/pyversions/insurance-credibility)](https://pypi.org/project/insurance-credibility/) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-credibility/blob/main/LICENSE)

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

## Installation

```bash
uv add insurance-credibility
```

Or with pip:

```bash
pip install insurance-credibility
```

**Dependencies**: `numpy >= 2.0`, `scipy >= 1.10`, `polars >= 1.0`. No pandas required — but pandas DataFrames are accepted as input and converted automatically.

**Optional**: `pandas >= 2.0` for pandas input support. `torch >= 2.0` for the deep attention model.

```bash
uv add "insurance-credibility[pandas]"   # with pandas support
uv add "insurance-credibility[deep]"     # with deep attention model
```

**Python**: 3.10, 3.11, 3.12.

---

## Quickstart

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# One row per (scheme, underwriting year)
df = pl.DataFrame({
    "scheme":    ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "year":      [2022, 2023, 2024, 2022, 2023, 2024, 2022, 2023, 2024],
    "loss_rate": [0.65, 0.59, 0.61, 0.82, 0.78, 0.85, 0.48, 0.44, 0.46],
    "exposure":  [2_200_000, 2_400_000, 2_100_000,   # £ earned premium
                    380_000,   420_000,   405_000,
                  6_100_000, 6_300_000, 6_400_000],
})

bs = BuhlmannStraub()
bs.fit(df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

print(bs.k_)         # Bühlmann's k: noise-to-signal ratio
print(bs.z_)         # credibility factors per scheme
print(bs.premiums_)  # credibility-blended loss ratio per scheme
```

```
k = 1847432.3   (earned premium needed for Z = 0.5)

shape: (3, 2)
┌────────┬──────────┐
│ group  ┆ Z        │
│ ---    ┆ ---      │
│ str    ┆ f64      │
╞════════╪══════════╡
│ A      ┆ 0.794    │
│ B      ┆ 0.406    │
│ C      ┆ 0.929    │
└────────┴──────────┘
```

Scheme B gets only 41% weight on its own experience because its £1.2m total earned premium is below k. Scheme C at £18.8m earned premium gets 93% — the model almost entirely trusts its own history.

---

## Group credibility: schemes and large accounts

`BuhlmannStraub` fits structural parameters — within-group variance (v) and between-group variance (a) — from the portfolio using method of moments. It then computes the credibility factor Z_i for each group:

```
Z_i = w_i / (w_i + k)    where k = v/a
```

Z approaches 1.0 as exposure grows — thick schemes are trusted almost entirely. Z shrinks toward 0 on thin schemes — the portfolio mean gets most of the weight.

**Practical interpretation of k**: a scheme needs earned premium equal to k to be 50% credible. You can read off the pricing committee's question — "how big does a scheme need to be before we take its experience seriously?" — directly from k:

```python
for target_z in [0.50, 0.75, 0.90]:
    required = bs.k_ * target_z / (1.0 - target_z)
    print(f"Z = {target_z:.0%}  →  required exposure = £{required:,.0f}")
```

On a 30-scheme, 5-year benchmark with known true parameters (mu=0.650, v=0.020, a=0.005, k=4.0):

| Tier | Raw MAE | Portfolio avg MAE | Credibility MAE |
|---|---|---|---|
| Thin (< 500 exposure) | 0.0074 | 0.0596 | **0.0069** |
| Medium (500–2000) | 0.0030 | 0.0423 | **0.0029** |
| Thick (2000+) | 0.0014 | 0.0337 | 0.0014 (tie) |

Credibility beats raw experience on thin and medium tiers. On thick tiers, Z approaches 1.0 and the two methods converge — which is correct behaviour.

`HierarchicalBuhlmannStraub` extends this to nested group structures: scheme → book, sector → district → area. Following Jewell (1975). Thin schemes borrow from their book mean; thin books borrow from the portfolio grand mean.

---

## Exact Bayesian credibility: claim counts

`PoissonGammaCredibility` is the exact Bayesian alternative when you have claim counts and exposures (rather than pre-computed loss ratios). The Poisson-Gamma conjugate pair gives a closed-form posterior — no MCMC, no approximation.

```python
from insurance_credibility import PoissonGammaCredibility

df_counts = pl.DataFrame({
    "scheme":   ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "year":     [2022, 2023, 2024] * 3,
    "claims":   [132, 118, 125,   28, 35, 30,   310, 295, 320],
    "exposure": [2200, 2400, 2100, 380, 420, 405, 6100, 6300, 6400],
})

model = PoissonGammaCredibility()
model.fit(df_counts, group_col="scheme",
          claims_col="claims", exposure_col="exposure")

# Exact posterior 95% credibility intervals — no bootstrapping
intervals = model.credibility_intervals(0.95)

# Score a new scheme: 45 claims over 800 exposure
result = model.predict(claims=45, exposure=800)
print(result["credibility_rate"])  # posterior mean
print(result["Z"])                 # credibility factor
print(result["lower"], result["upper"])  # 95% interval
```

The `beta_` parameter is the "effective prior exposure" — equivalent to Bühlmann's k. A scheme needs exposure equal to `beta_` to reach Z = 0.5.

---

## Individual policy experience rating

For commercial motor and fleet pricing, where you want to move individual policies away from the GLM rate based on their own claims history:

```python
from insurance_credibility import ClaimsHistory, StaticCredibilityModel

histories = [
    ClaimsHistory("POL001", periods=[1, 2, 3], claim_counts=[0, 1, 0],
                  exposures=[1.0, 1.0, 0.8], prior_premium=1_800.0),
    ClaimsHistory("POL002", periods=[1, 2, 3], claim_counts=[2, 1, 2],
                  exposures=[1.0, 1.0, 1.0], prior_premium=1_800.0),
]

model = StaticCredibilityModel()
model.fit(histories)

cf = model.predict(histories[0])    # credibility factor
posterior_premium = histories[0].prior_premium * cf
```

`exposures` is the key parameter that distinguishes this from flat NCD tables: a policy with 0.5 years of exposure gets far less credibility than one with 5 years, regardless of claim count.

**Portfolio balance**: experience rating redistributes premium but should not inflate the total. Apply `balance_calibrate` to enforce this:

```python
from insurance_credibility import balance_calibrate

cal = balance_calibrate(model.predict, histories)
print(f"Relative bias before calibration: {cal.relative_bias:+.2%}")
print(f"Calibration factor: {cal.calibration_factor:.4f}")
```

---

## UK motor example: comparing manual calculation to model output

One of the most useful audit steps is verifying that the model matches the formula. For scheme `SCH-007` with £420k total earned premium, a 72% observed loss ratio, and a fitted k of £1.85m:

```python
# Manual
w    = 420_000        # total earned premium
k    = bs.k_          # 1_847_432
x_bar = 0.72          # observed mean loss ratio
mu   = bs.mu_hat_     # collective mean

Z     = w / (w + k)   # 420k / (420k + 1847k) = 0.185
P     = Z * x_bar + (1 - Z) * mu

print(f"Z = {Z:.4f}")   # 0.1853
print(f"P = {P:.4f}")   # 0.6574

# Verify against model
row = bs.premiums_.filter(pl.col("group") == "SCH-007")
assert abs(row["credibility_premium"][0] - P) < 1e-4
```

The formula is closed-form and auditable. No black box.

---

## API reference

### Classical credibility

**`BuhlmannStraub`**

```python
bs = BuhlmannStraub(truncate_a=True)
bs.fit(data, group_col, period_col, loss_col, weight_col)

bs.mu_hat_   # float — collective mean loss rate
bs.v_hat_    # float — EPV (within-group variance)
bs.a_hat_    # float — VHM (between-group variance)
bs.k_        # float — Bühlmann's k = v/a
bs.z_        # pl.DataFrame["group", "Z"]
bs.premiums_ # pl.DataFrame["group", "exposure", "observed_mean",
             #              "Z", "credibility_premium", "complement"]
bs.summary() # prints structural params, returns premiums_ table
```

**`HierarchicalBuhlmannStraub`**

```python
model = HierarchicalBuhlmannStraub(level_cols=["book", "scheme"])
model.fit(data, period_col, loss_col, weight_col)

model.premiums_at("scheme")   # credibility premiums at scheme level
model.premiums_at("book")     # credibility premiums at book level
model.level_results_["book"]  # LevelResult: mu, v, a, k, z, premiums
model.summary()               # structural parameters at each level
```

**`PoissonGammaCredibility`**

```python
model = PoissonGammaCredibility(prior_alpha=None, prior_beta=None)
model.fit(data, group_col, claims_col, exposure_col)

model.alpha_        # float — fitted Gamma prior shape
model.beta_         # float — fitted Gamma prior rate (≡ Bühlmann k)
model.prior_mean_   # float — alpha / beta
model.premiums_     # pl.DataFrame with posterior estimates per group
model.credibility_intervals(0.95)   # exact posterior intervals
model.predict(claims, exposure)     # dict: rate, Z, lower, upper for new group
```

### Experience rating

**`ClaimsHistory`**

```python
h = ClaimsHistory(
    policy_id="POL001",
    periods=[1, 2, 3],
    claim_counts=[0, 1, 0],
    exposures=[1.0, 1.0, 0.8],   # years at risk per period
    prior_premium=1_800.0,        # GLM base rate
)
h.total_exposure   # 2.8
h.total_claims     # 1
h.claim_frequency  # 1 / 2.8 = 0.357
```

**`StaticCredibilityModel`**

```python
model = StaticCredibilityModel(kappa=None, min_kappa=0.1, max_kappa=1000.0)
model.fit(histories)

model.kappa_            # float — fitted kappa = sigma²/tau²
model.portfolio_mean_   # float — grand mean frequency
model.predict(history)              # float — credibility factor CF
model.predict_batch(histories)      # pl.DataFrame
model.credibility_weight(history)   # float — omega = t/(t+kappa)
```

**`DynamicPoissonGammaModel`**

```python
model = DynamicPoissonGammaModel(p0=0.5, q0=0.8)
model.fit(histories)

model.p_   # float — state reversion parameter
model.q_   # float — recency decay parameter
model.predict(history)              # float — credibility factor
model.predict_batch(histories)      # pl.DataFrame (includes posterior params)
model.predict_posterior_params(h)   # (alpha, beta) for uncertainty quantification
```

**Balance calibration**

```python
from insurance_credibility import balance_calibrate, apply_calibration

cal = balance_calibrate(model.predict, histories)
cal.calibration_factor   # multiplicative correction
cal.relative_bias        # (predicted - actual) / actual

posterior = apply_calibration(histories, model.predict, cal.calibration_factor)
```

---

## Model tiers

**`BuhlmannStraub`** — the standard for scheme and territory experience rating. Non-parametric: estimates v and a from the portfolio via method of moments. Closed-form, fits in milliseconds. The right default for most UK motor and home portfolios.

**`PoissonGammaCredibility`** — exact Bayesian credibility for claim count data. Same closed-form speed as Bühlmann-Straub, but with full posterior distributions and exact credibility intervals. Use this when you have claims and exposure separately (not pre-computed ratios) and when exact intervals matter for governance sign-off.

**`HierarchicalBuhlmannStraub`** — nested group structures. Scheme → book, postcode sector → district → area. Following Jewell (1975). Each level borrows strength from the level above.

**`StaticCredibilityModel`** — Bühlmann-Straub at individual policy level. Fits kappa = sigma² / tau² from a portfolio of policy histories. For commercial motor, fleet, and large account renewal pricing. Closed-form, fast, suitable for batch scoring.

**`DynamicPoissonGammaModel`** — Poisson-gamma state-space model following Ahn, Jeong, Lu & Wüthrich (2023). Seniority-weighted: recent years count more than old years. Produces the full posterior distribution per policy — useful when communicating uncertainty to a pricing committee or reinsurer. Requires numerical optimisation; run on Databricks for large portfolios.

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
- Bühlmann-Straub exposes the structural parameters (mu, v, a, k) directly, making them easy to inspect and challenge in peer review or regulatory sign-off.

For Poisson-Gamma likelihoods and non-Gaussian random effects, use `DynamicPoissonGammaModel`.

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
| Exact posterior intervals | No | No | Yes | Yes (`PoissonGammaCredibility`) |

---

## Limitations

- Structural parameter estimation (v, a) requires at least 30–50 groups and 3+ years to converge reliably. On the 30-group benchmark, VHM was underestimated by 57.6%. In thin portfolios, treat credibility factors as directional and apply a floor on Z.
- `StaticCredibilityModel` assumes homoscedastic within-policy variance. Segment by policy size tier on portfolios with large fleets alongside small ones.
- Kappa estimation needs at least 50–100 policies with 2+ years of history. Below this, the estimate is unreliable.
- Structural parameters must be refitted as portfolio composition changes. Stale kappa from a different historical book produces miscalibrated experience adjustments.

---

## Examples

The `examples/` directory contains runnable scripts:

- `examples/scheme_experience_rating.py` — Bühlmann-Straub for a 25-scheme motor portfolio. Shows structural parameters, per-scheme results, manual calculation cross-check, accuracy comparison by tier, and credibility thresholds.
- `examples/policy_experience_rating.py` — `StaticCredibilityModel` and `DynamicPoissonGammaModel` for 200 fleet policies. Shows why exposure matters, manual cross-check, balance calibration.

Run locally (no Databricks required):

```bash
git clone https://github.com/burning-cost/insurance-credibility
cd insurance-credibility
uv run python examples/scheme_experience_rating.py
uv run python examples/policy_experience_rating.py
```

Databricks notebooks in `notebooks/`:

- `notebooks/buhlmann_straub_demo.py` — full UK motor scheme workflow: fit, interpret, audit, hierarchical model, policy experience rating
- `notebooks/poisson_gamma_credibility_demo.py` — exact Bayesian credibility for claim counts with posterior intervals
- `notebooks/fremtpl2_credibility.py` — validation on French motor MTPL open data (22 regions)

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

- Bühlmann, H. & Straub, E. (1970). Glaubwürdigkeit für Schadensätze. *Mitteilungen VSVM*, 70, 111–133.
- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and Its Applications*. Springer.
- Jewell, W.S. (1975). Multidimensional Credibility. *Operations Research*, 23(5), 904–920.
- Ahn, J.Y., Jeong, H., Lu, Y. & Wüthrich, M.V. (2023). Dynamic Bayesian Credibility. arXiv:2308.16058.
- Calcetero, V., Badescu, A. & Lin, X.S. (2024). Credibility theory for the 21st century. *ASTIN Bulletin*.

---

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-credibility/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-credibility/issues)
- **Blog and tutorials:** [burning-cost.github.io](https://burning-cost.github.io)
- **Training course:** [Insurance Pricing in Python](https://burning-cost.github.io/course) — Module 6 covers credibility theory. £97 one-time.

## Licence

MIT
