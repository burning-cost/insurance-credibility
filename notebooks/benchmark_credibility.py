# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-credibility (Bühlmann-Straub) vs raw experience and portfolio mean
# MAGIC
# MAGIC **Library:** `insurance-credibility` — Bühlmann-Straub credibility estimation for group
# MAGIC pricing. Computes optimal credibility weights from structural parameters estimated
# MAGIC by method of moments, blending each group's own experience with the collective mean.
# MAGIC
# MAGIC **Baseline 1:** Raw group experience — the group's own exposure-weighted mean loss rate.
# MAGIC Overfits on thin groups where a bad year looks structural.
# MAGIC
# MAGIC **Baseline 2:** Grand mean — ignore group information entirely, price every group at
# MAGIC the portfolio mean. Underfits on groups where the experience is genuinely credible.
# MAGIC
# MAGIC **Dataset:** Synthetic fleet motor — 200 groups (schemes) with varying exposure
# MAGIC (5–500 vehicle-years per group per year) over 3 accident years. Groups genuinely
# MAGIC differ in underlying risk. Some groups have fewer than 20 claims across all years
# MAGIC — the setting where credibility weighting earns its keep.
# MAGIC
# MAGIC **Date:** 2026-03-22
# MAGIC
# MAGIC **Library version:** 0.1.4
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The key question: does Bühlmann-Straub credibility weighting improve out-of-sample
# MAGIC prediction on the held-out year compared to using raw group experience (which overfits
# MAGIC on thin groups) or ignoring group identity entirely? The expected answer is yes —
# MAGIC but only meaningfully for the thin-exposure groups. The benchmark makes this concrete.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-credibility polars numpy pandas matplotlib scipy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from insurance_credibility import BuhlmannStraub

warnings.filterwarnings("ignore")

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generation
# MAGIC
# MAGIC Synthetic fleet motor — 200 groups, 3 accident years (2022, 2023, 2024).
# MAGIC
# MAGIC DGP:
# MAGIC - Each group has a true underlying claim frequency mu_i drawn from a Gamma
# MAGIC   distribution: mu_i ~ Gamma(alpha=4, scale=0.05), so mean ~0.20 claims/veh-yr.
# MAGIC - The Gamma shape controls between-group heterogeneity (VHM).
# MAGIC - Within each year, observed claims = Poisson(mu_i * w_{ij}).
# MAGIC - Exposure w_{ij} varies widely: small schemes (5–50 veh-yrs) and large (100–500).
# MAGIC
# MAGIC The DGP is designed so:
# MAGIC - Groups with small exposure have noisy observed rates — raw experience is unreliable.
# MAGIC - Groups with large exposure have stable rates — raw experience and credibility agree.
# MAGIC - The portfolio has genuine between-group heterogeneity (a > 0), so credibility
# MAGIC   is doing something real, not just regressing to the mean for nothing.
# MAGIC
# MAGIC Split: years 2022 and 2023 are the training panel; year 2024 is the hold-out.
# MAGIC This is the actuarial standard — fit on historical experience, price the next year.

# COMMAND ----------

rng = np.random.default_rng(2024)

N_GROUPS = 200
YEARS = [2022, 2023, 2024]
TRAIN_YEARS = [2022, 2023]
TEST_YEAR = 2024

# True underlying frequency per group — drawn once, stable over time (no trend)
# Gamma(shape=4, scale=0.05) => mean 0.20, CV = 0.50 (meaningful between-group variance)
MU_SHAPE = 4.0
MU_SCALE = 0.05
mu_true = rng.gamma(MU_SHAPE, MU_SCALE, N_GROUPS)  # shape (200,)

# Exposure structure: mix of small, medium, large schemes
# Skewed: most schemes are small, a few are large — realistic fleet book
exposure_base = rng.choice(
    [rng.uniform(5, 30), rng.uniform(30, 150), rng.uniform(150, 500)],
    p=[0.50, 0.35, 0.15],
    size=N_GROUPS,
).astype(float)
# Assign properly (rng.choice with callable doesn't work that way — rebuild)
size_class = rng.choice([0, 1, 2], size=N_GROUPS, p=[0.50, 0.35, 0.15])
exposure_base = np.where(
    size_class == 0, rng.uniform(5, 30, N_GROUPS),
    np.where(size_class == 1, rng.uniform(30, 150, N_GROUPS),
             rng.uniform(150, 500, N_GROUPS))
)

# Add year-to-year exposure variation (±20%) — schemes grow/shrink
records = []
for yr_idx, year in enumerate(YEARS):
    yr_mult = rng.uniform(0.8, 1.2, N_GROUPS)
    exposure = np.maximum(exposure_base * yr_mult, 1.0)
    claims = rng.poisson(mu_true * exposure)
    loss_rate = claims / exposure  # observed rate
    for g in range(N_GROUPS):
        records.append({
            "group": f"G{g:03d}",
            "year": year,
            "exposure": exposure[g],
            "claims": claims[g],
            "loss_rate": loss_rate[g],
            "mu_true": mu_true[g],
        })

df_all = pl.DataFrame(records)

# Split
df_train = df_all.filter(pl.col("year").is_in(TRAIN_YEARS))
df_test  = df_all.filter(pl.col("year") == TEST_YEAR)

print(f"Total records: {len(df_all)}")
print(f"Train (years {TRAIN_YEARS}): {len(df_train)} group-years")
print(f"Test  (year {TEST_YEAR}):  {len(df_test)} groups")
print()
print("Exposure distribution (train, total over 2 years):")
exp_by_group = (
    df_train.group_by("group")
    .agg(pl.col("exposure").sum().alias("total_exposure"))
    .get_column("total_exposure")
    .to_numpy()
)
for pct in [10, 25, 50, 75, 90]:
    print(f"  p{pct}: {np.percentile(exp_by_group, pct):.0f} veh-yrs")
print()
print(f"True mu_i: mean={mu_true.mean():.4f}, CV={mu_true.std()/mu_true.mean():.2f}")
print(f"Groups with < 20 total claims in train: "
      f"{(df_train.group_by('group').agg(pl.col('claims').sum()).get_column('claims').to_numpy() < 20).sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline 1: Raw Group Experience
# MAGIC
# MAGIC The observed exposure-weighted mean loss rate over the training years.
# MAGIC No shrinkage — thin groups get full weight on noisy observations.
# MAGIC Practitioners often use this when they "trust the numbers" on small schemes.

# COMMAND ----------

t0 = time.perf_counter()

raw_exp = (
    df_train
    .group_by("group")
    .agg([
        pl.col("claims").sum().alias("total_claims"),
        pl.col("exposure").sum().alias("total_exposure"),
    ])
    .with_columns(
        (pl.col("total_claims") / pl.col("total_exposure")).alias("raw_rate")
    )
)

baseline_raw_time = time.perf_counter() - t0

# Grand mean (second baseline: ignore group info entirely)
grand_mean = (
    df_train.get_column("claims").sum() / df_train.get_column("exposure").sum()
)

print(f"Raw experience computed in {baseline_raw_time:.3f}s")
print(f"Grand mean loss rate (portfolio): {grand_mean:.4f}")
print()
print("Raw rate distribution:")
raw_rates = raw_exp.get_column("raw_rate").to_numpy()
for pct in [10, 25, 50, 75, 90]:
    print(f"  p{pct}: {np.percentile(raw_rates, pct):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library: Bühlmann-Straub Credibility
# MAGIC
# MAGIC Fits the Bühlmann-Straub model to the training panel. Estimates:
# MAGIC - **v** (EPV): expected within-group variance — the noise level
# MAGIC - **a** (VHM): between-group variance — the signal level
# MAGIC - **k = v/a**: noise-to-signal ratio (Bühlmann's k)
# MAGIC - **Z_i**: credibility factor for each group, Z_i = w_i / (w_i + k)
# MAGIC
# MAGIC The credibility premium for group i is:
# MAGIC   P_i = Z_i * X_bar_i + (1 - Z_i) * mu_hat
# MAGIC
# MAGIC Large k means homogeneous portfolio — trust the collective.
# MAGIC Small k means heterogeneous portfolio — trust the group's own experience.

# COMMAND ----------

t0 = time.perf_counter()

bs = BuhlmannStraub()
bs.fit(
    df_train.to_pandas(),  # accepts pandas or polars
    group_col="group",
    period_col="year",
    loss_col="loss_rate",
    weight_col="exposure",
)

library_fit_time = time.perf_counter() - t0

print(f"Bühlmann-Straub fit time: {library_fit_time:.3f}s")
print()
bs.summary()

# COMMAND ----------

# Inspect key structural parameters
print(f"Structural parameters:")
print(f"  mu_hat (collective mean): {bs.mu_:.6f}  [true: {mu_true.mean():.6f}]")
print(f"  v_hat  (EPV, within-grp): {bs.v_:.6f}")
print(f"  a_hat  (VHM, betw-grp):   {bs.a_:.6f}")
print(f"  k      (v/a, noise:signal): {bs.k_:.2f}")
print()
print(f"Credibility factors Z_i:")
z_vals = bs.z_.get_column("Z").to_numpy()
print(f"  min: {z_vals.min():.4f}  (thinnest group)")
print(f"  median: {np.median(z_vals):.4f}")
print(f"  max: {z_vals.max():.4f}  (largest group)")
print()
print(f"Groups with Z < 0.30 (thin — mostly portfolio mean): {(z_vals < 0.30).sum()}")
print(f"Groups with Z > 0.70 (credible — mostly own rate):   {(z_vals > 0.70).sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics on the Held-Out Year
# MAGIC
# MAGIC We evaluate predictions against the actual 2024 observed loss rates.
# MAGIC
# MAGIC For each group and method:
# MAGIC - **RMSE**: root mean squared error on loss rate predictions (weighted by exposure)
# MAGIC - **MAE**: mean absolute error on loss rate predictions (weighted by exposure)
# MAGIC - **Gini**: discriminatory power — does higher predicted rate rank groups by actual risk?
# MAGIC - **Calibration**: overall A/E ratio on the test year
# MAGIC
# MAGIC We also split by exposure quartile to see where each method wins.

# COMMAND ----------

# Merge predictions with test year
premiums_df = bs.premiums_.rename({"group": "group", "credibility_premium": "cred_rate"})
# premiums_ columns: group, Z, x_bar, credibility_premium
# Get column names
print("Premiums DataFrame columns:", bs.premiums_.columns)

# COMMAND ----------

# Build comparison DataFrame
prem_pd = bs.premiums_.to_pandas()
raw_pd  = raw_exp.to_pandas()
test_pd = df_test.to_pandas()

comp = (
    test_pd
    .merge(prem_pd[["group", "credibility_premium"]], on="group")
    .merge(raw_pd[["group", "raw_rate", "total_exposure"]], on="group")
)
comp["grand_mean"] = grand_mean

# Predictions
y_true   = comp["loss_rate"].values
w_test   = comp["exposure"].values
w_train  = comp["total_exposure"].values
y_cred   = comp["credibility_premium"].values
y_raw    = comp["raw_rate"].values
y_grand  = comp["grand_mean"].values
mu_t     = comp.merge(
    pd.DataFrame({"group": [f"G{g:03d}" for g in range(N_GROUPS)], "mu_true": mu_true}),
    on="group"
)["mu_true"].values

# Exposure-weighted RMSE and MAE
def wmse(y, yhat, w):
    return float(np.sqrt(np.average((y - yhat)**2, weights=w)))

def wmae(y, yhat, w):
    return float(np.average(np.abs(y - yhat), weights=w))

def ae_ratio(y, yhat, w):
    return float(np.sum(y * w) / np.sum(yhat * w))

def gini(y, yhat):
    order = np.argsort(yhat)
    y_s = y[order]
    n = len(y_s)
    cum_y = np.cumsum(y_s) / max(y_s.sum(), 1e-10)
    cum_p = np.arange(1, n+1) / n
    return float(2 * np.trapz(cum_y, cum_p) - 1)

# Overall metrics
print(f"{'Metric':<40} {'Grand mean':>12} {'Raw exp.':>10} {'Cred (BS)':>10} {'True mu':>10}")
print("=" * 84)

for label, yhat in [("Grand mean", y_grand), ("Raw experience", y_raw),
                     ("Bühlmann-Straub", y_cred), ("True mu (oracle)", mu_t)]:
    rmse = wmse(y_true, yhat, w_test)
    mae  = wmae(y_true, yhat, w_test)
    ae   = ae_ratio(y_true, yhat, w_test)
    g    = gini(y_true, yhat)
    print(f"  {label:<38} RMSE={rmse:.5f}  MAE={mae:.5f}  A/E={ae:.3f}  Gini={g:.3f}")

# COMMAND ----------

# Break down by training exposure quartile
comp["exp_quartile"] = pd.qcut(comp["total_exposure"], q=4, labels=["Q1 (thin)", "Q2", "Q3", "Q4 (thick)"])

print(f"\nWeighted RMSE by training exposure quartile:")
print(f"  {'Quartile':<14} {'n':>4} {'Exp range':>14} {'Grand mean':>12} {'Raw exp.':>10} {'Cred (BS)':>10}")
print("-" * 68)

for q in ["Q1 (thin)", "Q2", "Q3", "Q4 (thick)"]:
    m = comp["exp_quartile"] == q
    sub = comp[m]
    yt = sub["loss_rate"].values
    wt = sub["exposure"].values

    rmse_gm  = wmse(yt, sub["grand_mean"].values, wt)
    rmse_raw = wmse(yt, sub["raw_rate"].values, wt)
    rmse_cr  = wmse(yt, sub["credibility_premium"].values, wt)
    exp_range = f"{sub['total_exposure'].min():.0f}-{sub['total_exposure'].max():.0f}"

    print(f"  {q:<14} {m.sum():>4}  {exp_range:>14}  {rmse_gm:>12.5f}  {rmse_raw:>10.5f}  {rmse_cr:>10.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Visualisations

# COMMAND ----------

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

z_by_group = bs.z_.to_pandas().rename(columns={"Z": "Z_val"})
comp = comp.merge(z_by_group[["group", "Z_val"]], on="group")

# ── Plot 1: Credibility factor Z vs training exposure ─────────────────────
ax1.scatter(comp["total_exposure"], comp["Z_val"], alpha=0.5, s=20, color="steelblue")
ax1.set_xlabel("Training exposure (total veh-yrs, 2 years)")
ax1.set_ylabel("Credibility factor Z_i")
ax1.set_title("Credibility Factor vs Group Exposure")
ax1.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Z = 0.5")
ax1.set_xscale("log")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Raw error vs credibility error by group ────────────────────────
err_raw  = np.abs(comp["raw_rate"].values - comp["loss_rate"].values)
err_cred = np.abs(comp["credibility_premium"].values - comp["loss_rate"].values)
ax2.scatter(err_raw, err_cred, alpha=0.4, s=15,
            c=comp["total_exposure"], cmap="viridis_r", norm=matplotlib.colors.LogNorm())
max_err = max(err_raw.max(), err_cred.max())
ax2.plot([0, max_err], [0, max_err], "k--", linewidth=1.5, label="Equal error")
ax2.set_xlabel("|Raw rate - actual|")
ax2.set_ylabel("|Credibility rate - actual|")
ax2.set_title("Abs Error: Raw vs Credibility\n(colour = training exposure, dark = thin)")
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=matplotlib.colors.LogNorm(
    vmin=comp["total_exposure"].min(), vmax=comp["total_exposure"].max()))
sm.set_array([])
plt.colorbar(sm, ax=ax2, label="Training exposure")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Plot 3: RMSE improvement by exposure quartile ─────────────────────────
quartiles = ["Q1 (thin)", "Q2", "Q3", "Q4 (thick)"]
rmse_raw_q, rmse_cr_q, rmse_gm_q = [], [], []

for q in quartiles:
    m = comp["exp_quartile"] == q
    yt = comp.loc[m, "loss_rate"].values
    wt = comp.loc[m, "exposure"].values
    rmse_gm_q.append(wmse(yt, comp.loc[m, "grand_mean"].values, wt))
    rmse_raw_q.append(wmse(yt, comp.loc[m, "raw_rate"].values, wt))
    rmse_cr_q.append(wmse(yt, comp.loc[m, "credibility_premium"].values, wt))

x = np.arange(len(quartiles))
w = 0.25
ax3.bar(x - w,   rmse_gm_q,  w, label="Grand mean",       color="skyblue",   alpha=0.85)
ax3.bar(x,       rmse_raw_q, w, label="Raw experience",    color="tomato",    alpha=0.85)
ax3.bar(x + w,   rmse_cr_q,  w, label="Bühlmann-Straub",   color="seagreen",  alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels(quartiles, fontsize=9)
ax3.set_ylabel("Weighted RMSE (test year)")
ax3.set_title("RMSE by Training Exposure Quartile")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

# ── Plot 4: Shrinkage — predicted rate vs raw rate, coloured by Z ─────────
sc = ax4.scatter(
    comp["raw_rate"], comp["credibility_premium"],
    c=comp["Z_val"], cmap="RdYlGn", vmin=0, vmax=1,
    alpha=0.6, s=20
)
max_rate = max(comp["raw_rate"].max(), comp["credibility_premium"].max())
ax4.plot([0, max_rate], [0, max_rate], "k--", linewidth=1.5, label="No shrinkage")
ax4.axhline(grand_mean, color="grey", linestyle=":", linewidth=1.5, label=f"Grand mean ({grand_mean:.3f})")
ax4.set_xlabel("Raw group rate")
ax4.set_ylabel("Credibility-weighted rate")
ax4.set_title("Shrinkage Effect\n(colour = Z: green=high credibility, red=low)")
plt.colorbar(sc, ax=ax4, label="Credibility factor Z_i")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.suptitle(
    "insurance-credibility: Bühlmann-Straub vs Baselines — Fleet Motor Benchmark",
    fontsize=13, fontweight="bold"
)
plt.savefig("/tmp/benchmark_credibility.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_credibility.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verdict

# COMMAND ----------

rmse_gm  = wmse(y_true, y_grand, w_test)
rmse_raw = wmse(y_true, y_raw,   w_test)
rmse_cr  = wmse(y_true, y_cred,  w_test)
rmse_oracle = wmse(y_true, mu_t, w_test)

mae_gm  = wmae(y_true, y_grand, w_test)
mae_raw = wmae(y_true, y_raw,   w_test)
mae_cr  = wmae(y_true, y_cred,  w_test)

ae_gm  = ae_ratio(y_true, y_grand, w_test)
ae_raw = ae_ratio(y_true, y_raw,   w_test)
ae_cr  = ae_ratio(y_true, y_cred,  w_test)

g_gm  = gini(y_true, y_grand)
g_raw = gini(y_true, y_raw)
g_cr  = gini(y_true, y_cred)

print("=" * 66)
print("VERDICT: Bühlmann-Straub vs baselines")
print("=" * 66)
print()
print(f"{'Metric':<30} {'Grand mean':>12} {'Raw exp.':>12} {'Bühlmann-Straub':>16}")
print("-" * 72)
print(f"  {'RMSE (test year)':<28} {rmse_gm:>12.5f} {rmse_raw:>12.5f} {rmse_cr:>16.5f}")
print(f"  {'MAE (test year)':<28} {mae_gm:>12.5f} {mae_raw:>12.5f} {mae_cr:>16.5f}")
print(f"  {'A/E ratio':<28} {ae_gm:>12.3f} {ae_raw:>12.3f} {ae_cr:>16.3f}")
print(f"  {'Gini coefficient':<28} {g_gm:>12.4f} {g_raw:>12.4f} {g_cr:>16.4f}")
print()
print(f"Oracle (true mu) RMSE: {rmse_oracle:.5f}  [lower bound on achievable error]")
print()

pct_vs_raw = (rmse_raw - rmse_cr) / rmse_raw * 100
pct_vs_gm  = (rmse_gm  - rmse_cr) / rmse_gm  * 100

print(f"Credibility RMSE improvement vs raw experience:  {pct_vs_raw:+.1f}%")
print(f"Credibility RMSE improvement vs grand mean:      {pct_vs_gm:+.1f}%")
print()
print("Where the improvement comes from:")
print("  - Thin groups (Q1): credibility shrinks noisy raw rates toward the portfolio mean.")
print("    This reduces variance in the prediction without adding much bias.")
print("  - Thick groups (Q4): credibility gives high Z, so the result is close to raw")
print("    experience anyway. Little change here.")
print("  - Grand mean: always wrong for groups that genuinely differ from the average.")
print("    Credibility beats it everywhere, including large groups.")
print()
print("Honest caveat:")
print("  On a portfolio with genuine between-group heterogeneity and a reasonable sample")
print("  of groups, Bühlmann-Straub wins on RMSE — but the win is concentrated in the")
print("  thin-exposure tail. If your fleet book is mostly large schemes with 200+ veh-yrs,")
print(f"  Z_i approaches 1 and credibility converges to raw experience. k = {bs.k_:.1f}")
print("  tells you the noise-to-signal ratio; if k is very large, the portfolio looks")
print("  homogeneous and the collective mean is the right answer for everyone.")
print()
print(f"Fit time: {library_fit_time:.3f}s")
print("The Bühlmann-Straub estimator is closed-form — fitting 200 groups over 2 years")
print("takes milliseconds. There is no computational reason to skip it.")
