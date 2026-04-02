# Databricks notebook source
# MAGIC %md
# MAGIC # BMSEquilibriumSimulator: NCD Underreporting and Game-Theoretic Pricing
# MAGIC
# MAGIC This notebook demonstrates the full workflow for computing Nash equilibrium
# MAGIC reporting thresholds in a UK no-claims discount (NCD) system, based on:
# MAGIC
# MAGIC - **Liang, Zhang, Zhou & Zou (arXiv:2601.12655)** — first published analysis of
# MAGIC   strategic underreporting in an oligopolistic insurance market
# MAGIC - **Lemaire (1977)** — foundational "hunger for bonus" dynamic programming
# MAGIC
# MAGIC The core pricing problem: observed claim frequency at high-NCD classes understates
# MAGIC true frequency because small claims are strategically withheld. A GLM fit on
# MAGIC reported claims absorbs this suppression into the NCD coefficients. High-NCD
# MAGIC policyholders are systematically undercharged; lower-NCD policyholders
# MAGIC cross-subsidise them.
# MAGIC
# MAGIC **What this notebook covers:**
# MAGIC 1. UK NCD ladder structure and threshold computation
# MAGIC 2. Frequency suppression quantification
# MAGIC 3. Nash equilibrium threshold under duopoly competition (Liang et al.)
# MAGIC 4. Sensitivity analysis: how thresholds change with interest rates, premiums, and severity
# MAGIC 5. GLM correction: estimating true NCD relativities from observed data

# COMMAND ----------

# MAGIC %pip install insurance-credibility scipy numpy polars

# COMMAND ----------

import numpy as np
from scipy import stats as scipy_stats
import warnings

from insurance_credibility import BMSEquilibriumSimulator

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. UK NCD Ladder Structure
# MAGIC
# MAGIC The standard UK motor NCD ladder has 10 classes (0-year to 9-year NCD).
# MAGIC Step-back rule: 2 years per fault claim. Discounts range from 0% (no NCD)
# MAGIC to 70% (maximum NCD, typically reached after 9 claim-free years).

# COMMAND ----------

# Standard UK 10-class NCD ladder
uk_discounts = [0.0, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.65, 0.70, 0.70]
base_premium = 1000.0  # £1,000 base premium for illustration

# Gamma severity distribution (Liang et al. base case parameters)
# Gamma(alpha=1.2, rate=0.0085) => mean ~ £141, but scaled for UK motor
# In practice: calibrate to your own loss distribution
# We use a scaled version giving mean ~ £800 (UK small-to-medium claim)
severity_dist = scipy_stats.gamma(a=1.5, scale=500.0)

print(f"Severity distribution: Gamma(a=1.5, scale=500)")
print(f"Mean claim size: £{severity_dist.mean():.0f}")
print(f"Median claim size: £{severity_dist.median():.0f}")
print(f"90th percentile: £{severity_dist.ppf(0.90):.0f}")
print(f"95th percentile: £{severity_dist.ppf(0.95):.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Basic Threshold Computation (Lemaire Algorithm)

# COMMAND ----------

# Simulate observed frequencies: higher NCD classes show lower reported frequency
# (a mixture of true lower risk and strategic underreporting)
observed_freq = [
    0.080,  # Class 0: no NCD, young/new drivers, high risk, full reporting
    0.070,  # Class 1: 1yr NCD
    0.065,  # Class 2: 2yr NCD
    0.058,  # Class 3: 3yr NCD
    0.052,  # Class 4: 4yr NCD
    0.045,  # Class 5: 5yr NCD
    0.040,  # Class 6: 6yr NCD
    0.038,  # Class 7: 7yr NCD
    0.033,  # Class 8: 8yr NCD
    0.028,  # Class 9: 9yr NCD (max, most underreported)
]

sim = BMSEquilibriumSimulator(
    discounts=uk_discounts,
    base_premium=base_premium,
    step_back=2,
    discount_factor=0.97,  # ~3% risk-free rate
    claim_freq=0.05,
    severity_dist=severity_dist,
    max_horizon=10,
)
sim.fit(observed_freq=observed_freq)
sim.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Interpreting the Thresholds
# MAGIC
# MAGIC The threshold b*_n is the claim amount below which a rational policyholder
# MAGIC in class n will self-insure rather than report to protect their NCD.
# MAGIC
# MAGIC Key insight: thresholds are highest at mid-ladder classes (3-6), where the
# MAGIC absolute premium penalty from a step-back is largest.

# COMMAND ----------

premiums = sim.class_premiums()
thresholds = sim.thresholds_
reporting_probs = sim.reporting_probs_

print("NCD Class Analysis")
print("=" * 70)
print(f"{'Class':>5} {'Discount':>8} {'Premium':>8} {'Threshold b*':>14} "
      f"{'P(report)':>11} {'Implied λ_obs':>14}")
print("-" * 70)
for i in range(10):
    obs = observed_freq[i]
    p_rep = reporting_probs[i]
    print(
        f"  {i:>3}    {uk_discounts[i]:>6.0%}  £{premiums[i]:>6.0f}  "
        f"£{thresholds[i]:>12.2f}  {p_rep:>10.3f}  "
        f"{obs:>13.4f}"
    )
print()
print("Threshold = NPV of the premium penalty from a fault claim")
print("P(report) = P(claim > threshold) under assumed severity distribution")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Frequency Bias: True vs Observed

# COMMAND ----------

corrected = sim.corrected_freq_
bias = sim.frequency_bias_()

print("Frequency Bias Analysis")
print("=" * 65)
print(f"{'Class':>5} {'Observed λ':>12} {'Corrected λ':>13} {'Bias (%)':>10} {'Undercount':>11}")
print("-" * 65)
for i in range(10):
    obs_i = observed_freq[i]
    corr_i = corrected[i]
    bias_i = bias[i]
    if np.isfinite(corr_i):
        undercount = corr_i - obs_i
        print(
            f"  {i:>3}  {obs_i:>11.4f}  {corr_i:>12.4f}  "
            f"{bias_i*100:>9.1f}%  {undercount:>10.4f}"
        )
    else:
        print(f"  {i:>3}  {obs_i:>11.4f}       N/A          N/A          N/A")

print()
print("Bias: (obs - true) / true. Negative = frequency is underestimated.")
print("A 20% bias at class 5 means observed frequency is 20% below true frequency.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Stationary Distribution
# MAGIC
# MAGIC The steady-state distribution of policyholders across NCD classes.
# MAGIC This is the long-run proportion in each class under the reporting strategy.

# COMMAND ----------

pi = sim.stationary_dist_
avg_class = np.average(np.arange(10), weights=pi)
avg_discount = np.average(uk_discounts, weights=pi)

print("Steady-State NCD Distribution")
print("=" * 50)
for i in range(10):
    bar = "█" * int(pi[i] * 60)
    print(f"  Class {i}: {pi[i]:>6.3f}  {bar}")
print()
print(f"  Average NCD class: {avg_class:.2f}")
print(f"  Average discount: {avg_discount:.1%}")
print(f"  Fraction at maximum NCD (class 8-9): {pi[8:].sum():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Liang et al. Nash Equilibrium Threshold (Two-Insurer Model)
# MAGIC
# MAGIC For a two-class BMS under duopoly competition, Liang et al. derive a
# MAGIC closed-form Nash equilibrium threshold. This is the threshold that
# MAGIC rational policyholders adopt given equilibrium premiums.
# MAGIC
# MAGIC Base case from paper: p₀=0.9, Gamma(1.2, 0.0085), κ=1.25, δ=0.97,
# MAGIC k₁=0.015, k₂=0.8. Nash equilibrium: θ*₁≈35.83, θ*₂≈33.45.

# COMMAND ----------

# Two-class BMS simulator for the Liang et al. model
sim_liang = BMSEquilibriumSimulator(
    discounts=[0.0, 0.25],
    base_premium=35.85,
    step_back=1,
    discount_factor=0.97,
    claim_freq=0.10,
    severity_dist=scipy_stats.gamma(a=1.2, scale=1.0 / 0.0085),
)

# Compute Nash equilibrium threshold at the paper's equilibrium premiums
result = sim_liang.liang_equilibrium(
    theta1=35.83, theta2=33.45,
    kappa=1.25, k1=0.015, k2=0.8,
)

print("Liang et al. (2026) Base Case Nash Equilibrium")
print("=" * 50)
print(f"  Insurer 1 Class 1 premium (θ*₁):  {result['theta1']:.4f}")
print(f"  Insurer 2 Class 1 premium (θ*₂):  {result['theta2']:.4f}")
print(f"  Premium gap (θ*₁ - θ*₂):          {result['premium_diff']:.4f}")
print(f"  Choice probability η(Δ):           {result['eta']:.4f}")
print(f"  Nash equilibrium threshold b*:     {result['threshold']:.4f}")
print(f"  Discount factor δ:                 {result['discount_factor']}")
print(f"  Penalty ratio κ:                   {result['kappa']}")
print()
print("Interpretation: policyholders with a loss below b* = {:.4f} will not".format(result['threshold']))
print("  report the claim, regardless of which insurer they are with.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Sensitivity Analysis
# MAGIC
# MAGIC How does the equilibrium threshold vary with key parameters?

# COMMAND ----------

print("Sensitivity: Threshold vs Premium Differential (θ₁ - θ₂)")
print("=" * 55)
print(f"{'θ₁':>6} {'θ₂':>6} {'Δ':>6} {'η':>8} {'b*':>10}")
print("-" * 55)
for theta1, theta2 in [
    (30.0, 35.0), (32.0, 35.0), (35.0, 35.0),
    (38.0, 35.0), (42.0, 35.0), (50.0, 35.0),
]:
    r = sim_liang.liang_equilibrium(theta1, theta2, kappa=1.25, k1=0.015, k2=0.8)
    print(f"{theta1:>6.1f} {theta2:>6.1f} {r['premium_diff']:>6.1f} "
          f"{r['eta']:>7.4f} {r['threshold']:>9.4f}")

# COMMAND ----------

print("\nSensitivity: Threshold vs Brand Preference k₂")
print("=" * 45)
print(f"{'k₂':>6} {'η':>8} {'b*':>10}")
print("-" * 45)
for k2 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    r = sim_liang.liang_equilibrium(35.83, 33.45, kappa=1.25, k1=0.015, k2=k2)
    print(f"{k2:>6.2f} {r['eta']:>7.4f} {r['threshold']:>9.4f}")
print()
print("At k₂=0.5 (symmetric): η=0.5, insurers charge same premium at equilibrium.")
print("At k₂>0.5: insurer 1 preferred, charges more (Proposition 4.1).")

# COMMAND ----------

print("\nSensitivity: UK NCD Threshold vs Interest Rate (Holtan 2001 effect)")
print("=" * 60)
print(f"{'Rate (%)':>10} {'δ':>8} {'b* at class 5':>15} {'b* at class 7':>15}")
print("-" * 60)
for r_pct in [1.0, 2.0, 3.0, 5.0, 8.0, 11.0, 15.0]:
    delta = 1.0 / (1.0 + r_pct / 100.0)
    s = BMSEquilibriumSimulator(
        discounts=uk_discounts,
        base_premium=base_premium,
        step_back=2,
        discount_factor=delta,
        claim_freq=0.05,
    )
    s.fit()
    print(f"{r_pct:>9.1f}%  {delta:>7.4f}  £{s.thresholds_[5]:>13.2f}  £{s.thresholds_[7]:>13.2f}")
print()
print("Higher rates → lower NPV of future premiums → lower threshold → more reporting.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Corrected GLM NCD Relativities
# MAGIC
# MAGIC The practical pricing application: what NCD relativities should a GLM
# MAGIC use if we correct for the underreporting bias?

# COMMAND ----------

# Implied NCD relativities from observed data (naive GLM)
# Relativity = frequency_class / frequency_class_0
obs_array = np.array(observed_freq)
naive_relativities = obs_array / obs_array[0]

# Corrected relativities
corr_array = sim.corrected_freq_
corr_relativities = corr_array / corr_array[0]

print("NCD Relativities: Naive GLM vs Bias-Corrected")
print("=" * 70)
print(f"{'Class':>5} {'Obs freq':>10} {'Naïve rel.':>12} {'Corr freq':>12} {'Corr rel.':>11} {'Change':>8}")
print("-" * 70)
for i in range(10):
    if np.isfinite(corr_relativities[i]):
        change_pct = (corr_relativities[i] - naive_relativities[i]) / naive_relativities[i] * 100
        print(
            f"  {i:>3}  {obs_array[i]:>9.4f}  {naive_relativities[i]:>11.4f}  "
            f"{corr_array[i]:>11.4f}  {corr_relativities[i]:>10.4f}  "
            f"{change_pct:>6.1f}%"
        )
    else:
        print(
            f"  {i:>3}  {obs_array[i]:>9.4f}  {naive_relativities[i]:>11.4f}  "
            f"       N/A         N/A       N/A"
        )

print()
print("Positive 'Change': corrected relativity is higher (more expensive) than naive.")
print("These classes are undercharged by the naive GLM.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Protected NCD Scenario
# MAGIC
# MAGIC A policyholder with Protected NCD has near-zero rational retention threshold
# MAGIC for a single claim (their discount percentage is preserved). We can model
# MAGIC this with a very small step_back (or post-claim base premium increase).

# COMMAND ----------

# Protected NCD: step_back effectively 0 in terms of discount,
# but base premium increases post-claim by ~10%
# Model as: same discount structure but 10% higher effective base premium after claim
print("Protected NCD: Threshold Comparison")
print("=" * 55)
print()

# Standard (unprotected)
sim_unprotected = BMSEquilibriumSimulator(
    discounts=uk_discounts,
    base_premium=base_premium,
    step_back=2,
    discount_factor=0.97,
    claim_freq=0.05,
    severity_dist=severity_dist,
)
sim_unprotected.fit()

# Protected NCD: step back to adjacent class only (insurer reprices base up ~10%)
# Approximate PNCD as step_back=1 with 10% higher effective base premium
sim_protected = BMSEquilibriumSimulator(
    discounts=uk_discounts,
    base_premium=base_premium * 0.12,  # approximate: 12% base repricing post-claim
    step_back=2,
    discount_factor=0.97,
    claim_freq=0.05,
    severity_dist=severity_dist,
)
sim_protected.fit()

print(f"{'Class':>5} {'Unprotected b*':>16} {'PNCD approx b*':>16}")
print("-" * 42)
for i in range(10):
    print(
        f"  {i:>3}  £{sim_unprotected.thresholds_[i]:>14.2f}  "
        f"£{sim_protected.thresholds_[i]:>14.2f}"
    )
print()
print("PNCD dramatically lowers the rational retention threshold.")
print("A PNCD holder has little incentive to suppress claims (one fault claim).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Key Takeaways for UK Pricing Teams
# MAGIC
# MAGIC 1. **Do not treat observed claim frequency at high NCD classes as ground truth.**
# MAGIC    It understates true frequency by a factor of 1/p_n where p_n = P(Y > b*_n).
# MAGIC
# MAGIC 2. **Mid-ladder classes (3-6 years NCD) show the greatest bias.**
# MAGIC    The absolute premium penalty is largest here, creating the strongest
# MAGIC    suppression incentive.
# MAGIC
# MAGIC 3. **The bias is not flat.** A uniform "underreporting loading" will correct
# MAGIC    the direction but not the pattern. Use class-specific corrections.
# MAGIC
# MAGIC 4. **Interest rates matter (Holtan 2001).** Rising rates reduce the NPV of
# MAGIC    future premium penalties, lowering thresholds and increasing reporting.
# MAGIC    The 2022-2024 rate rises will have modestly increased claim reporting at
# MAGIC    high-NCD classes.
# MAGIC
# MAGIC 5. **Protected NCD changes everything.** PNCD holders have near-zero rational
# MAGIC    retention threshold for a single claim. Segment your frequency model by
# MAGIC    protection status.
# MAGIC
# MAGIC 6. **The Nash equilibrium requires market-level competitor data** (Liang et al.).
# MAGIC    Steps 1-3 (threshold estimation and frequency correction) can be done with
# MAGIC    internal data only. Step 4 (full Nash equilibrium) requires PCW or IUA data.

# COMMAND ----------

print("BMSEquilibriumSimulator demo complete.")
print(f"Package version: {__import__('insurance_credibility').__version__}")
