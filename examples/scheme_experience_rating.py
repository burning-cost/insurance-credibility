"""
Bühlmann-Straub experience rating for motor fleet schemes.

This example works through the typical use case for a UK commercial motor
pricing team: you have 20–40 affinity schemes (fleet managers, trade
associations, professional bodies) with between 3 and 7 years of annual
loss ratios and earned premium exposure. Some schemes are large and credible;
most are not. You need to decide how much weight to give each scheme's own
experience versus the portfolio mean.

Bühlmann-Straub answers this optimally. The credibility factor Z_i for
scheme i is:

    Z_i = w_i / (w_i + k)    where k = v/a

where w_i is total earned premium, v is the average within-scheme volatility,
and a is the genuine between-scheme dispersion. Schemes earn credibility
proportional to their exposure.

Running this script:

    uv run python examples/scheme_experience_rating.py

No Databricks required — this runs locally against synthetic data.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from insurance_credibility import BuhlmannStraub

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic portfolio: 25 schemes, 5 underwriting years
# ─────────────────────────────────────────────────────────────────────────────
# The data structure is what you would export from your reserving system or
# data warehouse: one row per (scheme, underwriting year) with earned premium
# as the exposure weight and the loss ratio as the response.
#
# We generate it with a known true structure so we can verify at the end
# that the credibility estimates are better than raw experience.

rng = np.random.default_rng(2024)

N_SCHEMES = 25
YEARS = [2019, 2020, 2021, 2022, 2023]

# True underlying loss ratios: portfolio mean 68%, genuine between-scheme
# dispersion of ±15 percentage points (standard deviation ~8pp).
TRUE_MEAN = 0.68
TRUE_BETWEEN_SD = 0.08  # VHM: a = 0.0064
TRUE_WITHIN_SD = 0.06   # EPV: v = 0.0036 (year-to-year noise within scheme)
# True k = v / a = 0.0036 / 0.0064 ≈ 0.56

true_rates = rng.normal(TRUE_MEAN, TRUE_BETWEEN_SD, N_SCHEMES)
true_rates = np.clip(true_rates, 0.30, 1.10)  # realistic range
scheme_ids = [f"SCH-{i+1:03d}" for i in range(N_SCHEMES)]

# Earned premium exposures: deliberately unequal.
# A few large schemes (£5m+), most medium (£500k–£2m), some thin (<£200k).
base_premiums = np.array([
    8_500, 6_200, 4_800, 3_900, 3_200,   # large
    2_100, 1_800, 1_600, 1_400, 1_200,   # medium-large
    900,   800,   750,   700,   650,      # medium
    500,   450,   380,   320,   280,      # medium-small
    180,   140,   110,    80,    55,      # thin
], dtype=float) * 1_000  # £ thousands → £

rows = []
for i, (scheme, true_lr) in enumerate(zip(scheme_ids, true_rates)):
    for year in YEARS:
        # Vary exposure year-on-year ±15%
        exposure = base_premiums[i] * rng.uniform(0.85, 1.15)
        # Observed loss ratio: true rate + within-scheme noise
        observed_lr = true_lr + rng.normal(0.0, TRUE_WITHIN_SD)
        observed_lr = max(0.05, observed_lr)  # floor at 5%
        rows.append({
            "scheme_id":   scheme,
            "uwyr":        year,
            "earned_prem": round(exposure, 0),
            "loss_ratio":  round(observed_lr, 4),
        })

df = pl.DataFrame(rows)

print("Synthetic scheme panel")
print(f"  {df['scheme_id'].n_unique()} schemes × {df['uwyr'].n_unique()} years = {len(df)} rows")
print(f"  Earned premium range: £{df['earned_prem'].min():,.0f} – £{df['earned_prem'].max():,.0f}")
print(f"  True portfolio mean loss ratio: {TRUE_MEAN:.1%}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fit Bühlmann-Straub
# ─────────────────────────────────────────────────────────────────────────────

bs = BuhlmannStraub()
bs.fit(
    df,
    group_col="scheme_id",
    period_col="uwyr",
    loss_col="loss_ratio",
    weight_col="earned_prem",
)

print("Structural parameters")
print("─" * 42)
print(f"  Collective mean      mu  = {bs.mu_hat_:.4f}  (true: {TRUE_MEAN:.4f})")
print(f"  Process variance     v   = {bs.v_hat_:.6f}  (EPV, within-scheme noise)")
print(f"  Between-scheme var   a   = {bs.a_hat_:.6f}  (VHM, genuine heterogeneity)")
print(f"  Bühlmann's k         k   = {bs.k_:.2f}   (v / a, noise-to-signal ratio)")
print()

# k interpretation: a scheme needs earned_prem = k × (scale factor) to reach Z = 0.5
# Here k is in the same units as earned_prem (£), so a scheme needs ~k exposure
# for Z = 0.5.
# For brevity we just show Z values; in practice £-denominated k is directly
# interpretable: "a scheme needs £X earned premium to be 50% credible."

# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-scheme results table
# ─────────────────────────────────────────────────────────────────────────────

# Add true rates for comparison
true_df = pl.DataFrame({"scheme_id": scheme_ids, "true_lr": true_rates.tolist()})

results = (
    bs.premiums_
    .rename({"group": "scheme_id"})
    .join(true_df, on="scheme_id")
    .select([
        "scheme_id",
        "exposure",
        "observed_mean",
        "Z",
        "credibility_premium",
        "true_lr",
    ])
    .with_columns([
        pl.lit(bs.mu_hat_).alias("portfolio_mean"),
    ])
    .sort("exposure", descending=True)
)

print("Per-scheme results (sorted by exposure, large → small)")
print("─" * 80)
print(f"{'Scheme':<12} {'Exposure £k':>12} {'Observed LR':>12} {'Z':>7} {'Cred. LR':>10} {'True LR':>9}")
print("─" * 80)
for row in results.iter_rows(named=True):
    obs_str = f"{row['observed_mean']:.1%}"
    cred_str = f"{row['credibility_premium']:.1%}"
    true_str = f"{row['true_lr']:.1%}"
    print(
        f"  {row['scheme_id']:<10} "
        f"{row['exposure']/1000:>12,.0f} "
        f"{obs_str:>12} "
        f"{row['Z']:>7.3f} "
        f"{cred_str:>10} "
        f"{true_str:>9}"
    )

print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Manual calculation cross-check for one scheme
# ─────────────────────────────────────────────────────────────────────────────
# A pricing committee will ask "how did you get that number?".
# Walk through the calculation for the thinnest scheme (last row).

thin_row = results.sort("exposure").row(0, named=True)
print("Manual calculation cross-check")
print(f"  Scheme: {thin_row['scheme_id']}")
print(f"  Total earned premium (exposure):  £{thin_row['exposure']:,.0f}")
print(f"  Portfolio mean (complement):      {bs.mu_hat_:.4f}")
print(f"  Observed loss ratio (X̄_i):       {thin_row['observed_mean']:.4f}")
print(f"  Bühlmann's k:                     {bs.k_:.2f}")
z_manual = thin_row['exposure'] / (thin_row['exposure'] + bs.k_)
cred_manual = z_manual * thin_row['observed_mean'] + (1 - z_manual) * bs.mu_hat_
print(f"  Z = w / (w + k) = {thin_row['exposure']:,.0f} / ({thin_row['exposure']:,.0f} + {bs.k_:.2f})")
print(f"    = {z_manual:.4f}")
print(f"  Credibility premium = Z × X̄_i + (1-Z) × mu")
print(f"    = {z_manual:.4f} × {thin_row['observed_mean']:.4f} + {1-z_manual:.4f} × {bs.mu_hat_:.4f}")
print(f"    = {cred_manual:.4f}")
print(f"  Model output:  {thin_row['credibility_premium']:.4f}  ✓" if abs(cred_manual - thin_row['credibility_premium']) < 1e-4 else f"  Model output: {thin_row['credibility_premium']:.4f}  (discrepancy: check rounding)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Accuracy comparison: raw vs credibility vs portfolio average
# ─────────────────────────────────────────────────────────────────────────────
# Measure mean absolute error against true rates, by scheme size tier.

results = results.with_columns([
    (abs(pl.col("observed_mean") - pl.col("true_lr"))).alias("raw_ae"),
    (abs(pl.col("credibility_premium") - pl.col("true_lr"))).alias("cred_ae"),
    (abs(pl.col("portfolio_mean") - pl.col("true_lr"))).alias("mean_ae"),
    pl.when(pl.col("exposure") >= 2_000_000)
      .then(pl.lit("Large (≥£2m)"))
      .when(pl.col("exposure") >= 500_000)
      .then(pl.lit("Medium (£500k–£2m)"))
      .otherwise(pl.lit("Thin (<£500k)"))
      .alias("tier"),
])

tier_summary = (
    results.group_by("tier")
    .agg([
        pl.len().alias("n_schemes"),
        pl.col("raw_ae").mean().alias("raw_mae"),
        pl.col("cred_ae").mean().alias("cred_mae"),
        pl.col("mean_ae").mean().alias("mean_mae"),
    ])
    .sort("raw_mae", descending=True)
)

print("Mean absolute error by tier (loss ratio points)")
print("─" * 65)
print(f"  {'Tier':<22} {'N':>4} {'Raw':>8} {'Port. avg':>10} {'Credibility':>12}")
print("─" * 65)
for row in tier_summary.iter_rows(named=True):
    print(
        f"  {row['tier']:<22} {row['n_schemes']:>4} "
        f"{row['raw_mae']:>8.4f} {row['mean_mae']:>10.4f} {row['cred_mae']:>12.4f}"
    )

print()
print("Credibility dominates on thin schemes. On large schemes, Z approaches 1")
print("and the model correctly defers to the scheme's own experience.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 6. How much exposure is needed for 50% / 75% / 90% credibility?
# ─────────────────────────────────────────────────────────────────────────────
# A practical question for a pricing committee: "at what point do we trust
# a scheme's own experience?"

print("Credibility thresholds")
print(f"  k = {bs.k_:,.0f} (Bühlmann's k, in the same units as earned_prem)")
for target_z in [0.50, 0.75, 0.90]:
    required = bs.k_ * target_z / (1.0 - target_z)
    print(f"  Z = {target_z:.0%}  →  required exposure = £{required:,.0f}")

print()
print("Schemes below the Z=50% threshold should be treated as largely")
print("driven by the portfolio mean, not their own loss ratios.")
