"""
Classical credibility models for non-life insurance pricing.

Implements the Bühlmann-Straub (1970) credibility model, its hierarchical
extension (Jewell, 1975), the Poisson-Gamma conjugate Bayesian credibility
model, and the BMSEquilibriumSimulator for NCD underreporting analysis.

Quick start — Bühlmann-Straub::

    import polars as pl
    from insurance_credibility.classical import BuhlmannStraub

    bs = BuhlmannStraub()
    bs.fit(df, group_col="scheme", period_col="year",
           loss_col="loss_rate", weight_col="exposure")
    bs.summary()
    bs.z_        # credibility factors by scheme
    bs.premiums_ # full results DataFrame

For hierarchical multi-level structures::

    from insurance_credibility.classical import HierarchicalBuhlmannStraub

    model = HierarchicalBuhlmannStraub(level_cols=["region", "district", "sector"])
    model.fit(df, period_col="year", loss_col="loss_rate", weight_col="exposure")
    model.premiums_at("sector")

For Poisson claim count data (exact Bayesian credibility)::

    from insurance_credibility.classical import PoissonGammaCredibility

    model = PoissonGammaCredibility()
    model.fit(df, group_col="scheme", claims_col="claims", exposure_col="exposure")
    model.summary()
    model.credibility_intervals(0.95)   # exact posterior intervals
    model.predict(claims=45, exposure=1000)  # score a new group

For NCD underreporting / bonus-malus equilibrium analysis::

    import numpy as np
    from scipy import stats
    from insurance_credibility.classical import BMSEquilibriumSimulator

    uk_discounts = [0.0, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.65, 0.70, 0.70]
    sim = BMSEquilibriumSimulator(
        discounts=uk_discounts,
        base_premium=1000.0,
        severity_dist=stats.gamma(a=1.2, scale=1/0.0085),
    )
    sim.fit(observed_freq=[0.08, 0.07, 0.06, 0.055, 0.05, 0.045, 0.04, 0.04, 0.035, 0.03])
    sim.thresholds_          # reporting thresholds b*_n by class
    sim.reporting_probs_     # P(Y > b*_n) by class
    sim.corrected_freq_      # bias-corrected true frequencies
    sim.summary()
"""

from .bms import BMSEquilibriumSimulator
from .buhlmann_straub import BuhlmannStraub
from .conjugate import PoissonGammaCredibility
from .hierarchical import HierarchicalBuhlmannStraub, LevelResult

__all__ = [
    "BMSEquilibriumSimulator",
    "BuhlmannStraub",
    "HierarchicalBuhlmannStraub",
    "LevelResult",
    "PoissonGammaCredibility",
]
