"""
insurance-credibility: Credibility models for UK non-life insurance pricing.

Two subpackages covering the full credibility toolkit:

classical
    Bühlmann-Straub (1970) group credibility and its hierarchical extension
    (Jewell 1975) for nested structures (scheme → book, sector → district → area).

experience
    Individual policy-level Bayesian experience rating. Four model tiers:
    static Bühlmann-Straub, dynamic Poisson-gamma state-space, IS-surrogate,
    and deep attention (Wüthrich 2024).

Quick start::

    # Group-level credibility (scheme pricing)
    from insurance_credibility import BuhlmannStraub
    bs = BuhlmannStraub()
    bs.fit(df, group_col="scheme", period_col="year",
           loss_col="loss_rate", weight_col="exposure")

    # Individual policy experience rating
    from insurance_credibility import ClaimsHistory, StaticCredibilityModel
    model = StaticCredibilityModel()
    model.fit(histories)
    cf = model.predict(history)
"""

# Classical credibility
from .classical import BuhlmannStraub, HierarchicalBuhlmannStraub, LevelResult

# Experience rating data types
from .experience import CalibrationResult, ClaimsHistory

# Experience rating models
from .experience import (
    DynamicPoissonGammaModel,
    StaticCredibilityModel,
    SurrogateModel,
)

# Experience rating calibration
from .experience import (
    apply_calibration,
    balance_calibrate,
    balance_report,
    calibrated_predict_fn,
)

# Experience rating utilities
from .experience import (
    credibility_factor,
    exposure_weighted_mean,
    history_sufficient_stat,
    posterior_premium,
    seniority_weights,
)

__version__ = "0.1.4"

__all__ = [
    # Classical
    "BuhlmannStraub",
    "HierarchicalBuhlmannStraub",
    "LevelResult",
    # Experience — data types
    "ClaimsHistory",
    "CalibrationResult",
    # Experience — models
    "StaticCredibilityModel",
    "DynamicPoissonGammaModel",
    "SurrogateModel",
    "DeepAttentionModel",
    # Experience — calibration
    "balance_calibrate",
    "apply_calibration",
    "calibrated_predict_fn",
    "balance_report",
    # Experience — utilities
    "credibility_factor",
    "posterior_premium",
    "seniority_weights",
    "exposure_weighted_mean",
    "history_sufficient_stat",
    # Meta
    "__version__",
]


def __getattr__(name: str):
    """Lazy import for optional torch-dependent classes."""
    if name == "DeepAttentionModel":
        from .experience.attention import DeepAttentionModel

        return DeepAttentionModel
    raise AttributeError(f"module 'insurance_credibility' has no attribute {name!r}")
