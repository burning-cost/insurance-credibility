# Databricks notebook source
# MAGIC %md
# MAGIC # Poisson-Gamma Credibility: Exact Bayesian Scheme Pricing
# MAGIC
# MAGIC This notebook demonstrates `PoissonGammaCredibility` — the closed-form
# MAGIC Bayesian credibility model for claim frequency data. No MCMC required.
# MAGIC
# MAGIC **The problem**: You have a portfolio of motor schemes (or territories,
# MAGIC or NCD classes). Each scheme has observed claim counts and exposure.
# MAGIC Some schemes are large (credible). Some are small (thin). How do you
# MAGIC set a frequency loading for each scheme that balances its own experience
# MAGIC against the portfolio mean?
# MAGIC
# MAGIC **The answer**: Place a Gamma prior on each scheme's underlying Poisson
# MAGIC rate. After observing claims, the posterior is another Gamma. The
# MAGIC credibility estimate is the posterior mean — exact, closed-form, no
# MAGIC approximation.
# MAGIC
# MAGIC This is the CAS exam Bayesian credibility model. It is strictly better
# MAGIC than Bühlmann-Straub for Poisson claim counts because:
# MAGIC - It is the *exact* Bayesian posterior mean, not the best linear approximation
# MAGIC - It provides full posterior distributions, not just point estimates
# MAGIC - Credibility intervals come directly from Gamma quantiles — no bootstrapping

# COMMAND ----------

# MAGIC %pip install insurance-credibility --quiet

# COMMAND ----------

import numpy as np
import polars as pl
from insurance_credibility import PoissonGammaCredibility

np.random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic portfolio: 8 motor schemes, 3 years
# MAGIC
# MAGIC We generate a realistic scheme portfolio where:
# MAGIC - True underlying rates vary between schemes (Gamma-distributed)
# MAGIC - Claim counts follow Poisson given the true rate
# MAGIC - Exposures vary significantly (large schemes vs small schemes)

# COMMAND ----------

# True underlying rates — Gamma(shape=6, rate=100) → mean = 0.06
# This is our "true" portfolio distribution
rng = np.random.default_rng(42)
n_schemes = 8
true_rates = rng.gamma(shape=6, scale=1/100, size=n_schemes)
scheme_ids = [f"SCH-{i+1:02d}" for i in range(n_schemes)]

print("True underlying claim rates (per unit exposure):")
for s, r in zip(scheme_ids, true_rates):
    print(f"  {s}: {r:.4f}")
print(f"\nPortfolio mean: {true_rates.mean():.4f}")

# COMMAND ----------

# Generate 3 years of panel data
years = [2021, 2022, 2023]
# Exposures: deliberately unequal — some large schemes, some small
base_exposures = np.array([5000, 500, 200, 8000, 1200, 100, 3000, 750], dtype=float)

rows = []
for i, (scheme, rate) in enumerate(zip(scheme_ids, true_rates)):
    for year in years:
        exposure = base_exposures[i] * rng.uniform(0.9, 1.1)
        claims = rng.poisson(rate * exposure)
        rows.append({
            "scheme": scheme,
            "year": year,
            "claims": int(claims),
            "exposure": round(exposure, 1),
        })

df = pl.DataFrame(rows)
print(f"Dataset: {len(df)} rows ({df['scheme'].n_unique()} schemes, {df['year'].n_unique()} years)")
print(df.head(12))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit the model

# COMMAND ----------

model = PoissonGammaCredibility()
model.fit(df, group_col="scheme", claims_col="claims", exposure_col="exposure")

print(f"Fitted prior: alpha={model.alpha_:.4f}, beta={model.beta_:.4f}")
print(f"Prior mean (portfolio rate): {model.prior_mean_:.4f}")
print(f"\nThis beta={model.beta_:.1f} means a scheme needs ~{model.beta_:.0f} exposure")
print(f"units to achieve Z=0.5 (equal weight on own data and prior).")

# COMMAND ----------

result = model.summary()
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compare credibility estimates to true rates
# MAGIC
# MAGIC The key diagnostic: credibility estimates should be closer to the true
# MAGIC rates than naive observed rates (shrinkage toward the prior helps for
# MAGIC thin schemes, where raw observations are noisy).

# COMMAND ----------

# Aggregate observed rates for comparison
obs = (
    df.group_by("scheme")
    .agg([
        pl.col("claims").sum(),
        pl.col("exposure").sum(),
    ])
    .with_columns((pl.col("claims") / pl.col("exposure")).alias("observed_rate"))
    .sort("scheme")
)

true_df = pl.DataFrame({
    "scheme": scheme_ids,
    "true_rate": true_rates.tolist(),
})

comparison = (
    model.premiums_
    .join(true_df, on="group", how="left")
    .rename({"group": "scheme"})
    .select(["scheme", "total_exposure", "observed_rate", "credibility_rate", "Z", "true_rate"])
    .with_columns([
        (abs(pl.col("observed_rate") - pl.col("true_rate")) / pl.col("true_rate") * 100).alias("obs_error_pct"),
        (abs(pl.col("credibility_rate") - pl.col("true_rate")) / pl.col("true_rate") * 100).alias("cred_error_pct"),
    ])
    .sort("total_exposure", descending=True)
)

display(comparison)

# COMMAND ----------

# Summary: mean absolute percentage error
obs_mape = comparison["obs_error_pct"].mean()
cred_mape = comparison["cred_error_pct"].mean()
print(f"Observed rate MAPE:      {obs_mape:.1f}%")
print(f"Credibility rate MAPE:   {cred_mape:.1f}%")
print(f"Improvement: {obs_mape - cred_mape:.1f} percentage points")
print("\nNote: small schemes (low exposure) drive the improvement.")
print("Large schemes (Z near 1) — credibility rate ≈ observed rate.")
print("Small schemes (Z near 0) — credibility rate ≈ prior mean.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Posterior credibility intervals
# MAGIC
# MAGIC Unlike Bühlmann-Straub, we have full posterior distributions. The
# MAGIC 95% credibility intervals come directly from Gamma quantiles — no
# MAGIC bootstrapping, no asymptotic approximation.
# MAGIC
# MAGIC Small schemes have wide intervals (high uncertainty), large schemes
# MAGIC have narrow intervals. This is the correct actuarial answer.

# COMMAND ----------

intervals = model.credibility_intervals(0.95)
intervals_full = (
    intervals
    .join(true_df, on="group", how="left")
    .rename({"group": "scheme"})
    .with_columns([
        ((pl.col("upper") - pl.col("lower")) / pl.col("credibility_rate") * 100).alias("interval_width_pct"),
        (
            (pl.col("true_rate") >= pl.col("lower")) &
            (pl.col("true_rate") <= pl.col("upper"))
        ).alias("true_rate_in_interval"),
    ])
    .sort("interval_width_pct")
)

display(intervals_full)

# COMMAND ----------

n_covered = intervals_full["true_rate_in_interval"].sum()
print(f"Coverage: {n_covered}/{n_schemes} schemes have true rate inside 95% credibility interval")
print("(Expected: ~7-8 out of 8 with correct model specification)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Scoring a new scheme (predict)
# MAGIC
# MAGIC A broker presents a new scheme: 45 claims over 800 earned car years.
# MAGIC What rate should we price it at?

# COMMAND ----------

new_claims = 45
new_exposure = 800.0
obs_rate = new_claims / new_exposure

result_new = model.predict(claims=new_claims, exposure=new_exposure, credibility_interval=0.95)

print("New scheme: 45 claims / 800 exposure")
print(f"  Observed rate:      {obs_rate:.4f}  ({obs_rate:.2%} per unit exposure)")
print(f"  Portfolio prior:    {model.prior_mean_:.4f}  ({model.prior_mean_:.2%})")
print(f"  Credibility factor: Z = {result_new['Z']:.3f}")
print(f"  Credibility rate:   {result_new['credibility_rate']:.4f}  ({result_new['credibility_rate']:.2%})")
print(f"  95% interval:       [{result_new['lower']:.4f}, {result_new['upper']:.4f}]")
print()
print(f"Interpretation: with Z={result_new['Z']:.2f}, we give this scheme")
print(f"  {result_new['Z']*100:.0f}% weight on its own rate and")
print(f"  {(1-result_new['Z'])*100:.0f}% weight on the portfolio mean.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Using a manually specified prior
# MAGIC
# MAGIC If you have external information about the portfolio (e.g., from industry
# MAGIC benchmarks or a reinsurance submission), you can supply the prior directly
# MAGIC instead of calibrating from data.

# COMMAND ----------

# Industry tells us: claim rate ~0.06, portfolio CV ~25%
# CV = sqrt(alpha/beta^2) / (alpha/beta) = 1/sqrt(alpha) = 0.25
# So alpha = 16, and beta = alpha / 0.06 = 267

model_external = PoissonGammaCredibility(prior_alpha=16.0, prior_beta=267.0)
model_external.fit(df, group_col="scheme", claims_col="claims", exposure_col="exposure")

print(f"External prior: alpha={model_external.alpha_:.1f}, beta={model_external.beta_:.1f}")
print(f"Prior mean: {model_external.prior_mean_:.4f}")
print(f"Prior CV: {1/np.sqrt(model_external.alpha_):.2%}")
print()

comparison_external = model_external.premiums_.select(
    ["group", "Z", "credibility_rate"]
).rename({"group": "scheme", "Z": "Z_external", "credibility_rate": "cred_rate_external"})

comparison_both = model.premiums_.select(
    ["group", "Z", "credibility_rate"]
).rename({"group": "scheme", "Z": "Z_data", "credibility_rate": "cred_rate_data"}).join(
    comparison_external, on="scheme"
)
display(comparison_both)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC `PoissonGammaCredibility` provides:
# MAGIC
# MAGIC 1. **Exact posterior mean** — not an approximation, not the best linear
# MAGIC    estimator. The true Bayesian answer for Poisson data with Gamma mixing.
# MAGIC
# MAGIC 2. **Posterior intervals** — exact Gamma quantiles, no bootstrapping.
# MAGIC    Thin schemes get wide intervals. Large schemes get narrow intervals.
# MAGIC    The intervals correctly reflect what we know.
# MAGIC
# MAGIC 3. **Empirical prior calibration** — method-of-moments on portfolio data
# MAGIC    gives sensible defaults. External priors can override for informed
# MAGIC    Bayesian updates.
# MAGIC
# MAGIC 4. **Zero new dependencies** — scipy only, already in the package.
# MAGIC
# MAGIC When to use it instead of BuhlmannStraub:
# MAGIC - You have claim counts and exposures (not pre-computed loss ratios)
# MAGIC - You want credibility intervals, not just a point estimate
# MAGIC - Your data is clearly Poisson-distributed (count data, small groups)
# MAGIC - Regulatory or audit sign-off requires a defensible Bayesian framework
