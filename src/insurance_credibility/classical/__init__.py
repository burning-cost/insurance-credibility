"""
Classical Bühlmann-Straub credibility models for non-life insurance pricing.

Implements the Bühlmann-Straub (1970) credibility model and its hierarchical
extension (Jewell, 1975). The standard tool for blending group-level loss
experience with the portfolio mean, weighted by exposure.

Quick start::

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
"""

from .buhlmann_straub import BuhlmannStraub
from .hierarchical import HierarchicalBuhlmannStraub, LevelResult

__all__ = [
    "BuhlmannStraub",
    "HierarchicalBuhlmannStraub",
    "LevelResult",
]
