# Databricks notebook source
# MAGIC %md
# MAGIC # Bühlmann-Straub Credibility: UK Motor Scheme Pricing
# MAGIC
# MAGIC This notebook demonstrates the full `insurance-credibility` workflow for
# MAGIC scheme experience rating — the most common credibility problem in UK commercial
# MAGIC motor and fleet pricing.
# MAGIC
# MAGIC **The problem**: You have 20–40 motor affinity schemes (trade associations,
# MAGIC fleet managers, leasing companies). Each has between 3 and 7 years of annual
# MAGIC loss experience. Some are large and their history is informative. Most are
# MAGIC small and their observed loss ratios are mostly noise. You need a defensible,
# MAGIC statistically optimal way to blend each scheme's own experience with the
# MAGIC portfolio mean.
# MAGIC
# MAGIC **What this notebook covers**:
# MAGIC 1. Generate a realistic synthetic scheme portfolio
# MAGIC 2. Fit Bühlmann-Straub and interpret the structural parameters
# MAGIC 3. Manual calculation cross-check for audit purposes
# MAGIC 4. Visualise how Z varies with exposure and what that means for pricing
# MAGIC 5. Hierarchical model for multi-tier structures (scheme → book → portfolio)
# MAGIC 6. Extend to individual policy experience rating via `StaticCredibilityModel`

# COMMAND ----------

# MAGIC %pip install insurance-credibility --quiet

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

import numpy as np
import polars as pl
from insurance_credibility import (
    BuhlmannStraub,
    HierarchicalBuhlmannStraub,
    PoissonGammaCredibility,
    ClaimsHistory,
    StaticCredibilityModel,
    balance_calibrate,
)

print(f"insurance-credibility loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic scheme portfolio
# MAGIC
# MAGIC We generate a panel of 30 commercial motor schemes over 5 underwriting years.
# MAGIC The data structure mirrors what you would pull from a reserving system:
# MAGIC one row per (scheme, underwriting year) with earned premium as the exposure
# MAGIC weight and the ultimate loss ratio as the response.
# MAGIC
# MAGIC The generating structure is:
# MAGIC - True portfolio mean: 68%
# MAGIC - Genuine between-scheme dispersion: SD ≈ 8pp (VHM a ≈ 0.0064)
# MAGIC - Within-scheme year-on-year noise: SD ≈ 6pp (EPV v ≈ 0.0036)
# MAGIC - True k = v/a ≈ 0.56

# COMMAND ----------

rng = np.random.default_rng(2024)

N_SCHEMES = 30
YEARS = [2019, 2020, 2021, 2022, 2023]

TRUE_MEAN       = 0.68
TRUE_BETWEEN_SD = 0.08  # a = 0.0064
TRUE_WITHIN_SD  = 0.06  # v = 0.0036

true_rates = np.clip(rng.normal(TRUE_MEAN, TRUE_BETWEEN_SD, N_SCHEMES), 0.30, 1.10)
scheme_ids = [f"SCH-{i+1:03d}" for i in range(N_SCHEMES)]

# Earned premiums: 5 large, 15 medium, 10 thin
base_ep = np.concatenate([
    rng.uniform(3_000, 9_000, 5),    # large £3m–£9m
    rng.uniform(500, 2_500, 15),     # medium
    rng.uniform(80, 450, 10),        # thin
]) * 1_000

rows = []
for i, (sid, true_lr) in enumerate(zip(scheme_ids, true_rates)):
    for year in YEARS:
        ep = base_ep[i] * rng.uniform(0.85, 1.15)
        obs_lr = true_lr + rng.normal(0.0, TRUE_WITHIN_SD)
        obs_lr = max(0.05, obs_lr)
        rows.append({
            "scheme_id":   sid,
            "uwyr":        year,
            "earned_prem": round(ep, 0),
            "loss_ratio":  round(obs_lr, 4),
        })

df = pl.DataFrame(rows)

print(f"Panel: {df['scheme_id'].n_unique()} schemes × {df['uwyr'].n_unique()} years = {len(df)} rows")
print(f"Earned premium range: £{df['earned_prem'].min():,.0f} – £{df['earned_prem'].max():,.0f} per year")
display(df.head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit Bühlmann-Straub
# MAGIC
# MAGIC The three structural parameters are:
# MAGIC
# MAGIC | Parameter | Symbol | Meaning |
# MAGIC |-----------|--------|---------|
# MAGIC | Collective mean | mu | Grand weighted-average loss ratio across all schemes and years |
# MAGIC | EPV | v | Average year-on-year noise *within* a scheme — this is the part you cannot price away |
# MAGIC | VHM | a | True dispersion *between* schemes — this is the genuine pricing signal |
# MAGIC | Bühlmann's k | k=v/a | Noise-to-signal ratio. Schemes need exposure k to reach Z=0.5 |

# COMMAND ----------

bs = BuhlmannStraub()
bs.fit(
    df,
    group_col="scheme_id",
    period_col="uwyr",
    loss_col="loss_ratio",
    weight_col="earned_prem",
)

print("Bühlmann-Straub structural parameters")
print("=" * 50)
print(f"  Collective mean   mu = {bs.mu_hat_:.4f}  (true: {TRUE_MEAN:.4f})")
print(f"  Process variance   v = {bs.v_hat_:.6f}  (true: {TRUE_WITHIN_SD**2:.6f})")
print(f"  Between-group var  a = {bs.a_hat_:.6f}  (true: {TRUE_BETWEEN_SD**2:.6f})")
print(f"  Bühlmann's k       k = {bs.k_:.4f}  (true: {TRUE_WITHIN_SD**2 / TRUE_BETWEEN_SD**2:.4f})")

# COMMAND ----------

results_tbl = bs.summary()
display(results_tbl)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Manual calculation cross-check
# MAGIC
# MAGIC This is the calculation a reviewing actuary or pricing committee should be
# MAGIC able to follow. For any scheme i with total earned premium w_i:
# MAGIC
# MAGIC ```
# MAGIC Z_i = w_i / (w_i + k)
# MAGIC P_i = Z_i × X̄_i + (1 − Z_i) × mu
# MAGIC ```
# MAGIC
# MAGIC where X̄_i is the scheme's own exposure-weighted mean loss ratio and mu is
# MAGIC the collective mean.

# COMMAND ----------

# Pick the thinnest and largest scheme for the cross-check
premiums = bs.premiums_
thin = premiums.sort("exposure").row(0, named=True)
large = premiums.sort("exposure", descending=True).row(0, named=True)

for label, row in [("Thinnest scheme", thin), ("Largest scheme", large)]:
    w = row["exposure"]
    x_bar = row["observed_mean"]
    k = bs.k_
    mu = bs.mu_hat_
    z = w / (w + k)
    p = z * x_bar + (1 - z) * mu
    print(f"\n{label}:  {row['group']}")
    print(f"  Earned premium (w):     £{w:,.0f}")
    print(f"  Observed mean LR (X̄):  {x_bar:.4f}")
    print(f"  Z = {w:,.0f} / ({w:,.0f} + {k:.2f}) = {z:.4f}")
    print(f"  P = {z:.4f} × {x_bar:.4f} + {1-z:.4f} × {mu:.4f} = {p:.4f}")
    print(f"  Model output: {row['credibility_premium']:.4f}",
          "✓" if abs(p - row['credibility_premium']) < 1e-4 else "(discrepancy)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Credibility weight Z by exposure tier
# MAGIC
# MAGIC The central output for a pricing committee: how much do we trust each scheme?
# MAGIC
# MAGIC - Z close to 1.0: scheme experience dominates. The scheme's observed LR
# MAGIC   is used almost directly.
# MAGIC - Z close to 0.0: scheme experience is overridden by the portfolio mean.
# MAGIC   Thin schemes are priced at the collective rate, not their noisy own data.

# COMMAND ----------

true_df = pl.DataFrame({"group": scheme_ids, "true_lr": true_rates.tolist()})
results = (
    bs.premiums_
    .join(true_df, on="group")
    .with_columns([
        pl.when(pl.col("exposure") >= 2_000_000).then(pl.lit("Large"))
          .when(pl.col("exposure") >= 500_000).then(pl.lit("Medium"))
          .otherwise(pl.lit("Thin"))
          .alias("tier"),
        (abs(pl.col("observed_mean")       - pl.col("true_lr"))).alias("raw_ae"),
        (abs(pl.col("credibility_premium") - pl.col("true_lr"))).alias("cred_ae"),
        (abs(pl.lit(bs.mu_hat_)            - pl.col("true_lr"))).alias("mean_ae"),
    ])
)

tier_summary = (
    results.group_by("tier")
    .agg([
        pl.len().alias("schemes"),
        pl.col("Z").mean().alias("mean_Z"),
        pl.col("raw_ae").mean().alias("raw_MAE"),
        pl.col("cred_ae").mean().alias("cred_MAE"),
        pl.col("mean_ae").mean().alias("mean_MAE"),
    ])
    .sort("raw_MAE", descending=True)
)

display(tier_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Credibility thresholds
# MAGIC
# MAGIC A frequently-asked question: "how large does a scheme need to be before
# MAGIC we take its experience seriously?"
# MAGIC
# MAGIC From k, we can answer directly:

# COMMAND ----------

print(f"Bühlmann's k = £{bs.k_:,.0f}")
print()
print("Earned premium required to reach Z =")
for target_z in [0.25, 0.50, 0.75, 0.90]:
    required = bs.k_ * target_z / (1.0 - target_z)
    print(f"  {target_z:.0%}:  £{required:,.0f}")

print()
print("These are the natural scheme-size tiers for your pricing grid.")
print("Schemes below the 50% threshold are essentially priced at the portfolio mean.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Hierarchical model: scheme → book
# MAGIC
# MAGIC When schemes are grouped into books (e.g. by distribution channel or class
# MAGIC of business), the two-level hierarchical model borrows strength at both levels.
# MAGIC Thin schemes borrow from their book mean; thin books borrow from the portfolio.
# MAGIC
# MAGIC This uses `HierarchicalBuhlmannStraub` following Jewell (1975).

# COMMAND ----------

# Assign each scheme to one of 5 books
book_map = {sid: f"BOOK-{(i % 5) + 1}" for i, sid in enumerate(scheme_ids)}
df_hier = df.with_columns(
    pl.col("scheme_id").replace(book_map).alias("book_id")
)

hier = HierarchicalBuhlmannStraub(level_cols=["book_id", "scheme_id"])
hier.fit(df_hier, period_col="uwyr", loss_col="loss_ratio", weight_col="earned_prem")

hier.summary()

# COMMAND ----------

# Scheme-level premiums use book-level mean as complement instead of portfolio mean
scheme_premiums = hier.premiums_at("scheme_id")
print("Hierarchical model — scheme-level premiums")
print("(complement is book mean, not portfolio mean)")
display(scheme_premiums.sort("exposure"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Individual policy experience rating
# MAGIC
# MAGIC For fleet and commercial accounts, the same credibility logic applies at
# MAGIC policy level. `StaticCredibilityModel` estimates kappa from the portfolio
# MAGIC and applies the Bühlmann formula to individual policy histories.
# MAGIC
# MAGIC The key: exposure matters. A fleet with 10 vehicle-years of clean history
# MAGIC earns a real discount. A 0.5-vehicle-year account with the same clean
# MAGIC record earns almost nothing.

# COMMAND ----------

def make_fleet_history(policy_id: str, n_years: int, frequency: float, ep: float) -> ClaimsHistory:
    """Generate a fleet history with Poisson claims."""
    periods = list(range(1, n_years + 1))
    exposures = [float(rng.uniform(0.8, 1.2)) for _ in range(n_years)]
    claims = [int(rng.poisson(frequency * e)) for e in exposures]
    return ClaimsHistory(policy_id=policy_id, periods=periods,
                         claim_counts=claims, exposures=exposures, prior_premium=ep)

# Training portfolio: 150 fleet policies
train_histories = [
    make_fleet_history(
        f"FLT-{i:04d}",
        n_years=int(rng.integers(2, 6)),
        frequency=float(rng.gamma(3, 0.4)),
        ep=float(rng.uniform(1000, 8000)),
    )
    for i in range(150)
]

exp_model = StaticCredibilityModel()
exp_model.fit(train_histories)

print(f"Static credibility model fitted on {len(train_histories)} fleet policies")
print(f"  kappa = {exp_model.kappa_:.3f}  (vehicle-years needed for Z=0.5)")
print()

# Score three illustrative policies
cases = [
    ("Large fleet, clean",  ClaimsHistory("A", [1,2,3,4,5], [0,0,0,0,0], exposures=[10.0]*5, prior_premium=3000.0)),
    ("Small fleet, clean",  ClaimsHistory("B", [1,2,3,4,5], [0,0,0,0,0], exposures=[0.5]*5,  prior_premium=3000.0)),
    ("Large fleet, losses", ClaimsHistory("C", [1,2,3,4,5], [3,2,4,3,5], exposures=[10.0]*5, prior_premium=3000.0)),
]

print(f"  {'Policy':<25} {'Veh-yrs':>8} {'Claims':>7} {'Z':>7} {'CF':>7} {'Posterior £':>12}")
print("-" * 72)
for label, h in cases:
    cf = exp_model.predict(h)
    z = exp_model.credibility_weight(h)
    print(f"  {label:<25} {h.total_exposure:>8.1f} {h.total_claims:>7} {z:>7.3f} {cf:>7.3f} {h.prior_premium*cf:>12.0f}")

print()
print("The small fleet with no claims gets almost no discount (Z≈0, model")
print("cannot distinguish genuine safety from thin data). The large fleet's")
print("no-claims record earns a meaningful credibility adjustment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Balance calibration
# MAGIC
# MAGIC Credibility redistributes premium between good and bad risks. It should
# MAGIC not inflate or deflate the total. Use `balance_calibrate` to verify
# MAGIC this and apply the correction if needed.

# COMMAND ----------

cal = balance_calibrate(exp_model.predict, train_histories)

print(f"Balance calibration on training portfolio")
print(f"  Sum actual (claims):     {cal.sum_actual:,.2f}")
print(f"  Sum predicted (model):   {cal.sum_predicted:,.2f}")
print(f"  Relative bias:           {cal.relative_bias:+.2%}")
print(f"  Calibration factor:      {cal.calibration_factor:.4f}")
print()
if abs(cal.relative_bias) < 0.02:
    print("Model is well-balanced. Calibration factor is near 1.0.")
else:
    print(f"Apply delta={cal.calibration_factor:.4f} to restore portfolio balance.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Task | Class | Key parameter |
# MAGIC |------|-------|---------------|
# MAGIC | Scheme/territory experience rating | `BuhlmannStraub` | k = v/a |
# MAGIC | Nested scheme structures | `HierarchicalBuhlmannStraub` | k at each level |
# MAGIC | Exact Bayesian (count data) | `PoissonGammaCredibility` | beta (prior exposure) |
# MAGIC | Fleet/policy experience rating | `StaticCredibilityModel` | kappa |
# MAGIC | Time-discounted policy rating | `DynamicPoissonGammaModel` | p, q |
# MAGIC
# MAGIC The unifying idea: all models blend the group's own experience with the
# MAGIC portfolio mean. The blend is governed by a single noise-to-signal parameter
# MAGIC (k, kappa, or beta depending on the model). Groups earn credibility
# MAGIC proportional to their exposure, not their claim count.
# MAGIC
# MAGIC See [burning-cost.github.io](https://burning-cost.github.io) for the full
# MAGIC methodology documentation and worked regulatory examples.
