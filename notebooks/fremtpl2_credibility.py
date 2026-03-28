# Databricks notebook source
# DISCLAIMER: freMTPL2 is French motor data (OpenML dataset 41214).
# Used here for methodology validation only — to demonstrate that
# Bühlmann-Straub credibility generalises to real insurance data.
# Not UK market data.
#
# Dataset: 677K French MTPL policies.
# Source: Noll, Salzmann, Wüthrich (2018).
# Group factor: Region (22 French administrative regions).
# Response: annual claim frequency (ClaimNb / Exposure).
#
# Date: 2026-03-28
# Library version: 0.1.8

# COMMAND ----------

# MAGIC %md
# MAGIC # freMTPL2: Bühlmann-Straub Credibility on Real French Motor Data
# MAGIC
# MAGIC French motor MTPL has 22 regions with genuine risk heterogeneity. We fit
# MAGIC Bühlmann-Straub to see what the structural parameters reveal, and confirm
# MAGIC that low-exposure regions are shrunk toward the grand mean.

# COMMAND ----------

%pip install insurance-credibility scikit-learn polars numpy --quiet

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from sklearn.datasets import fetch_openml

from insurance_credibility import BuhlmannStraub

print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load freMTPL2freq

# COMMAND ----------

print("Fetching freMTPL2freq from OpenML (dataset ID 41214, ~677K rows)...")
raw = fetch_openml(data_id=41214, as_frame=True, parser="auto")
df_pd = raw.frame
print(f"Rows: {len(df_pd):,}  |  Columns: {list(df_pd.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build a Panel
# MAGIC
# MAGIC freMTPL2 is cross-sectional — no year column. We split by VehAge quartile
# MAGIC as a proxy period so Bühlmann-Straub has multiple observations per region
# MAGIC to estimate within-region variance. Each (Region, pseudo-period) cell becomes
# MAGIC one panel row: total claims / total exposure = observed frequency.

# COMMAND ----------

df = (
    pl.from_pandas(df_pd)
    .select([
        pl.col("Region").cast(pl.Utf8),
        pl.col("ClaimNb").cast(pl.Float64),
        pl.col("Exposure").cast(pl.Float64),
        pl.col("VehAge").cast(pl.Float64),
    ])
    .filter(pl.col("Exposure") > 0)
)

# Assign VehAge quartile as pseudo-period (1=youngest, 4=oldest vehicles)
va = df.get_column("VehAge").to_numpy()
cuts = np.percentile(va, [25, 50, 75])
period = np.digitize(va, bins=cuts, right=True) + 1  # 1..4
df = df.with_columns(pl.Series("period", period.astype(int)))

# Aggregate to (Region, period) panel
panel = (
    df.group_by(["Region", "period"])
    .agg([
        pl.col("ClaimNb").sum().alias("claims"),
        pl.col("Exposure").sum().alias("exposure"),
    ])
    .with_columns(
        (pl.col("claims") / pl.col("exposure")).alias("freq")
    )
    .sort(["Region", "period"])
)

grand_mean = panel.get_column("claims").sum() / panel.get_column("exposure").sum()

print(f"Panel: {len(panel)} region-period cells "
      f"({panel.get_column('Region').n_unique()} regions x 4 pseudo-periods)")
print(f"Grand mean frequency: {grand_mean:.4f} claims/exposure-unit")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit Bühlmann-Straub

# COMMAND ----------

bs = BuhlmannStraub()
bs.fit(
    panel.to_pandas(),
    group_col="Region",
    period_col="period",
    loss_col="freq",
    weight_col="exposure",
)

print("Structural parameters:")
print(f"  mu_hat  (collective mean):    {bs.mu_:.6f}")
print(f"  v_hat   (EPV, within-region): {bs.v_:.8f}")
print(f"  a_hat   (VHM, betw-region):   {bs.a_:.8f}")
print(f"  k = v/a (noise:signal ratio): {bs.k_:.2f}")
print()
z_vals = bs.z_.get_column("Z").to_numpy()
print(f"Z across {len(z_vals)} regions: "
      f"min={z_vals.min():.4f}  median={np.median(z_vals):.4f}  max={z_vals.max():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Results Table

# COMMAND ----------

raw_by_region = (
    panel.group_by("Region")
    .agg([
        pl.col("claims").sum().alias("total_claims"),
        pl.col("exposure").sum().alias("total_exposure"),
    ])
    .with_columns(
        (pl.col("total_claims") / pl.col("total_exposure")).alias("raw_mean")
    )
)

prem = bs.premiums_.rename({"group": "Region"})

results = (
    raw_by_region
    .join(prem.select(["Region", "credibility_premium", "Z"]), on="Region")
    .with_columns(pl.lit(grand_mean).alias("grand_mean"))
    .sort("total_exposure")
)

print(f"{'Region':<12} {'Exposure':>12} {'Raw mean':>10} {'Cred. est.':>12} {'Z':>8}")
print("-" * 58)
for row in results.iter_rows(named=True):
    print(
        f"  {row['Region']:<10} {row['total_exposure']:>12,.0f} "
        f"{row['raw_mean']:>10.5f} {row['credibility_premium']:>12.5f} "
        f"{row['Z']:>8.4f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verdict

# COMMAND ----------

exp = results.get_column("total_exposure").to_numpy()
z   = results.get_column("Z").to_numpy()
raw = results.get_column("raw_mean").to_numpy()
cred = results.get_column("credibility_premium").to_numpy()

thin_mask  = exp <= np.percentile(exp, 25)
thick_mask = exp >= np.percentile(exp, 75)

raw_spread  = raw.max()  - raw.min()
cred_spread = cred.max() - cred.min()

print("=" * 58)
print("VERDICT")
print("=" * 58)
print(f"  k = {bs.k_:.1f} — noise-to-signal ratio.")
print(f"  Between-region heterogeneity: a_hat = {bs.a_:.6f} "
      f"({'positive — real signal' if bs.a_ > 0 else 'zero — homogeneous portfolio'})")
print()
print(f"  Thin regions (bottom exposure quartile):  mean Z = {z[thin_mask].mean():.3f}")
print(f"  Thick regions (top exposure quartile):    mean Z = {z[thick_mask].mean():.3f}")
print()
print(f"  Raw mean spread across regions:   {raw_spread:.5f}")
print(f"  Credibility estimate spread:      {cred_spread:.5f}  "
      f"({cred_spread/raw_spread:.2f}x of raw spread)")
print()
print("Low-exposure regions are visibly shrunk toward the grand mean —")
print("the defining behaviour of Bühlmann-Straub credibility.")
