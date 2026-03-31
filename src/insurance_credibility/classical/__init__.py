"""
Classical credibility models for non-life insurance pricing.

Implements the Bühlmann-Straub (1970) credibility model, its hierarchical
extension (Jewell, 1975), and the Poisson-Gamma conjugate Bayesian credibility
model.

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
"""

from .buhlmann_straub import BuhlmannStraub
from .conjugate import PoissonGammaCredibility
from .hierarchical import HierarchicalBuhlmannStraub, LevelResult

__all__ = [
    "BuhlmannStraub",
    "HierarchicalBuhlmannStraub",
    "LevelResult",
    "PoissonGammaCredibility",
]
