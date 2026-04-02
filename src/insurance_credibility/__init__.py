"""
insurance-credibility: Credibility models for UK non-life insurance pricing.

Two subpackages covering the full credibility toolkit:

classical
    Bühlmann-Straub (1970) group credibility and its hierarchical extension
    (Jewell 1975) for nested structures (scheme → book, sector → district → area).
    Also includes PoissonGammaCredibility: the exact Bayesian credibility model
    for claim count data (closed-form, no MCMC).

    BMSEquilibriumSimulator: game-theoretic NCD underreporting analysis based
    on Liang et al. (arXiv:2601.12655) and Lemaire (1977). Computes Nash
    equilibrium reporting thresholds, corrects observed frequencies for the
    hunger-for-bonus bias, and quantifies the cross-subsidy from lower-NCD
    to higher-NCD policyholders.

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

    # Exact Bayesian credibility for claim counts
    from insurance_credibility import PoissonGammaCredibility
    model = PoissonGammaCredibility()
    model.fit(df, group_col="scheme", claims_col="claims", exposure_col="exposure")
    model.credibility_intervals(0.95)  # exact posterior intervals

    # NCD underreporting / hunger-for-bonus equilibrium
    from scipy import stats
    from insurance_credibility import BMSEquilibriumSimulator
    sim = BMSEquilibriumSimulator(
        discounts=[0.0, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.65, 0.70, 0.70],
        base_premium=1000.0,
        severity_dist=stats.gamma(a=1.2, scale=1/0.0085),
    )
    sim.fit(observed_freq=[0.08, 0.07, 0.06, 0.055, 0.05, 0.045, 0.04, 0.04, 0.035, 0.03])
    sim.summary()

    # Individual policy experience rating
    from insurance_credibility import ClaimsHistory, StaticCredibilityModel
    model = StaticCredibilityModel()
    model.fit(histories)
    cf = model.predict(history)
"""

# Classical credibility
from .classical import (
    BMSEquilibriumSimulator,
    BuhlmannStraub,
    HierarchicalBuhlmannStraub,
    LevelResult,
    PoissonGammaCredibility,
)

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

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-credibility")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    # Classical
    "BMSEquilibriumSimulator",
    "BuhlmannStraub",
    "HierarchicalBuhlmannStraub",
    "LevelResult",
    "PoissonGammaCredibility",
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
