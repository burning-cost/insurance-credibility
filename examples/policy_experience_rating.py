"""
Individual commercial motor policy experience rating.

For fleets and large accounts, the GLM base rate is a starting point — not
the final price. A fleet with 5 years of no-claims history, or one with
a string of large losses, deserves to have that history reflected in the
renewal price.

This example shows how `StaticCredibilityModel` works for individual policy
experience rating:

- Fit kappa from a portfolio of policy histories (the structural parameter
  linking within-policy noise to between-policy dispersion)
- Apply the credibility formula to individual policies
- See how the same claim count produces very different credibility factors
  depending on how much exposure backs it

Running this script:

    uv run python examples/policy_experience_rating.py

This runs locally against synthetic data. For a portfolio-scale demonstration
with model comparison and calibration, see the Databricks notebook at:
    notebooks/buhlmann_straub_demo.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from insurance_credibility import ClaimsHistory, StaticCredibilityModel
from insurance_credibility import DynamicPoissonGammaModel
from insurance_credibility import balance_calibrate

# ─────────────────────────────────────────────────────────────────────────────
# 1. Construct a portfolio of commercial motor policy histories
# ─────────────────────────────────────────────────────────────────────────────
# Each policy has:
#   - A prior_premium: the GLM base rate (what we would charge ignoring history)
#   - claim_counts: claims per policy year
#   - exposures: years at risk per period (partial years for mid-term starts)
#
# In practice this comes from your policy admin system: filter to accounts
# with at least 2 years of history, join to the GLM score.

rng = np.random.default_rng(42)

# Portfolio: 200 commercial motor fleets, 3–6 years of history
N_POLICIES = 200

def make_history(policy_id: str, n_years: int, true_lambda: float, prior_prem: float) -> ClaimsHistory:
    """Generate a synthetic policy history."""
    periods = list(range(1, n_years + 1))
    # Most years full exposure; some mid-term (0.5–1.0)
    exposures = [1.0 if rng.random() > 0.15 else rng.uniform(0.5, 1.0) for _ in range(n_years)]
    claim_counts = [int(rng.poisson(true_lambda * e)) for e in exposures]
    return ClaimsHistory(
        policy_id=policy_id,
        periods=periods,
        claim_counts=claim_counts,
        exposures=exposures,
        prior_premium=prior_prem,
    )

# True underlying frequencies: most policies close to the base rate,
# a long tail of high-risk accounts.
true_lambdas = rng.gamma(shape=4.0, scale=0.25, size=N_POLICIES)  # mean=1.0, CV=50%
prior_premiums = rng.uniform(800, 4000, N_POLICIES)  # base GLM premium £800–£4000
n_years_each = rng.integers(2, 7, size=N_POLICIES)   # 2–6 years of history

histories = [
    make_history(f"POL-{i+1:04d}", int(n_years_each[i]), true_lambdas[i], float(prior_premiums[i]))
    for i in range(N_POLICIES)
]

print(f"Portfolio: {len(histories)} policies")
print(f"  History lengths: {min(h.n_periods for h in histories)}–{max(h.n_periods for h in histories)} years")
print(f"  Total exposure: {sum(h.total_exposure for h in histories):,.1f} vehicle-years")
print(f"  Total claims: {sum(h.total_claims for h in histories):,}")
print(f"  Portfolio frequency: {sum(h.total_claims for h in histories) / sum(h.total_exposure for h in histories):.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fit the StaticCredibilityModel
# ─────────────────────────────────────────────────────────────────────────────

model = StaticCredibilityModel()
model.fit(histories)

print("Static credibility model")
print(f"  kappa              = {model.kappa_:.3f}")
print(f"  within variance    = {model.within_variance_:.4f}")
print(f"  between variance   = {model.between_variance_:.4f}")
print(f"  portfolio mean     = {model.portfolio_mean_:.3f}")
print()
print(f"  kappa = {model.kappa_:.2f} means a policy needs {model.kappa_:.1f} vehicle-years")
print(f"  of exposure to reach Z=0.5 (half weight on its own experience).")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Score the portfolio in batch
# ─────────────────────────────────────────────────────────────────────────────

results_df = model.predict_batch(histories)
print(f"Scored {len(results_df)} policies.")
print(results_df.describe())
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Detailed case studies: three contrasting policies
# ─────────────────────────────────────────────────────────────────────────────
# These illustrate how the model behaves differently depending on exposure —
# the critical distinction from a flat NCD table.

cases = {
    "Large fleet, no claims":
        ClaimsHistory("CASE-A", [1,2,3,4,5], [0,0,0,0,0],
                      exposures=[10.0,10.0,10.0,9.5,10.0], prior_premium=1800.0),
    "Small account, no claims":
        ClaimsHistory("CASE-B", [1,2,3,4,5], [0,0,0,0,0],
                      exposures=[0.5, 0.5, 0.5, 0.5, 0.5], prior_premium=1800.0),
    "Large fleet, high claims":
        ClaimsHistory("CASE-C", [1,2,3,4,5], [3,4,2,5,3],
                      exposures=[10.0,10.0,10.0,9.5,10.0], prior_premium=1800.0),
}

print("Case studies: how credibility varies with exposure")
print("─" * 70)
print(f"  {'Policy':<30} {'Exposure':>9} {'Claims':>7} {'Z':>7} {'CF':>7} {'Post. £':>9}")
print("─" * 70)
for label, h in cases.items():
    cf = model.predict(h)
    omega = model.credibility_weight(h)
    print(
        f"  {label:<30} {h.total_exposure:>9.1f} {h.total_claims:>7} "
        f"{omega:>7.3f} {cf:>7.3f} {h.prior_premium * cf:>9.0f}"
    )

print()
print("  The large fleet with no claims gets a meaningful discount (Z=high,")
print("  experience trusted). The small account with identical no-claims")
print("  history gets far less: its Z is low, so the model barely moves from")
print("  the prior. This is correct — 2.5 vehicle-years of no claims tells")
print("  us almost nothing.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Manual calculation cross-check
# ─────────────────────────────────────────────────────────────────────────────

print("Manual calculation cross-check — large fleet, no claims (CASE-A)")
h = cases["Large fleet, no claims"]
t = h.total_exposure
kappa = model.kappa_
mu = h.prior_premium
y_bar = h.claim_frequency

omega = t / (t + kappa)
cf_manual = omega * (y_bar / mu) + (1 - omega)

print(f"  Total exposure (t):     {t:.1f}")
print(f"  Portfolio kappa:        {kappa:.3f}")
print(f"  Z (omega) = t/(t+kappa) = {t:.1f}/({t:.1f}+{kappa:.3f}) = {omega:.4f}")
print(f"  Empirical frequency Y_bar = {y_bar:.4f}")
print(f"  A priori rate mu         = {mu:.1f}")
print(f"  CF = omega × (Y_bar/mu) + (1-omega)")
print(f"     = {omega:.4f} × ({y_bar:.4f}/{mu:.1f}) + {1-omega:.4f}")
print(f"     = {cf_manual:.4f}")
print(f"  Model output: {model.predict(h):.4f}  ✓" if abs(cf_manual - model.predict(h)) < 1e-4 else f"  Model output: {model.predict(h):.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Balance calibration
# ─────────────────────────────────────────────────────────────────────────────
# Experience rating redistributes premium between good and bad risks — it
# should not increase or decrease the total. The balance calibration step
# applies a small multiplicative correction to ensure sum(posterior) = sum(prior × observed).

cal = balance_calibrate(model.predict, histories)

print("Balance calibration")
print(f"  Sum of actual frequency:     {cal.sum_actual:,.2f}")
print(f"  Sum of predicted frequency:  {cal.sum_predicted:,.2f}")
print(f"  Relative bias before cal:    {cal.relative_bias:+.2%}")
print(f"  Calibration factor delta:    {cal.calibration_factor:.4f}")
print()
if abs(cal.relative_bias) < 0.02:
    print("  Model is well-balanced (<2% bias). Calibration factor near 1.0.")
else:
    print(f"  Apply delta={cal.calibration_factor:.4f} to all posterior premiums to restore balance.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Compare static vs dynamic model
# ─────────────────────────────────────────────────────────────────────────────
# The dynamic Poisson-gamma model (Ahn et al. 2023) additionally down-weights
# older claims — a sensible extension when fleet composition changes year on year.
# It requires MLE optimisation; for large portfolios run this on Databricks.

print("Dynamic vs Static credibility comparison")
print("─" * 50)
print("Fitting DynamicPoissonGammaModel (may take a few seconds)...")

dynamic = DynamicPoissonGammaModel()
dynamic.fit(histories, verbose=False)

print(f"  Fitted p (state reversion) = {dynamic.p_:.3f}")
print(f"  Fitted q (decay)           = {dynamic.q_:.3f}")
print()

print(f"  {'Policy':<30} {'Static CF':>10} {'Dynamic CF':>12}")
print("─" * 56)
for label, h in cases.items():
    static_cf = model.predict(h)
    dyn_cf = dynamic.predict(h)
    print(f"  {label:<30} {static_cf:>10.3f} {dyn_cf:>12.3f}")

print()
print(f"  q={dynamic.q_:.2f}: discount on year 1 vs year 5 data =",
      f"{dynamic.q_**4:.2f}x")
print("  Lower q means older years have less influence on the credibility factor.")
print("  Static model weights all years equally.")
print()
print("For most commercial motor portfolios, the static model is sufficient.")
print("Use the dynamic model when you have evidence that risk composition is")
print("changing within accounts (e.g. fleet turnover >40% annually).")
