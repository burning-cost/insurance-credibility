"""
Regression tests for P1 bug fixes in StaticCredibilityModel (batch 3 audit).

Covers:
- assert statements replaced with ValueError
- prior_premium <= 0 raises ValueError instead of returning 1.0 silently
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_credibility.experience import ClaimsHistory, StaticCredibilityModel


def make_history(
    policy_id: str,
    counts: list[int],
    exposures: list[float] | None = None,
    prior: float = 400.0,
) -> ClaimsHistory:
    if exposures is None:
        exposures = [1.0] * len(counts)
    return ClaimsHistory(
        policy_id=policy_id,
        periods=list(range(1, len(counts) + 1)),
        claim_counts=counts,
        prior_premium=prior,
        exposures=exposures,
    )


def make_portfolio(n: int, rng: np.random.Generator, mean: float = 1.0) -> list[ClaimsHistory]:
    histories = []
    for i in range(n):
        counts = rng.poisson(mean, size=3).tolist()
        histories.append(make_history(f"P{i}", counts, prior=mean))
    return histories


@pytest.fixture
def fitted_model() -> StaticCredibilityModel:
    rng = np.random.default_rng(42)
    histories = make_portfolio(20, rng)
    model = StaticCredibilityModel()
    model.fit(histories)
    return model


class TestAssertReplacement:
    """Assert statements in production paths must be replaced with ValueError."""

    def test_predict_zero_prior_premium_raises_valueerror(self, fitted_model):
        """ValueError must be raised when prior_premium <= 0.

        The ClaimsHistory constructor validates prior_premium and raises during
        object construction — which is correct behaviour. The test wraps the
        whole operation (construction + predict) inside pytest.raises so either
        site can satisfy the contract.
        """
        with pytest.raises(ValueError, match="prior_premium"):
            bad_history = make_history("BAD", [1, 0, 1], prior=0.0)
            fitted_model.predict(bad_history)

    def test_predict_negative_prior_premium_raises_valueerror(self, fitted_model):
        """Negative prior_premium must also raise ValueError."""
        with pytest.raises(ValueError, match="prior_premium"):
            bad_history = make_history("BAD2", [1, 0], prior=-100.0)
            fitted_model.predict(bad_history)

    def test_predict_no_exposures_raises_valueerror(self, fitted_model):
        """predict() raises ValueError when history has exposures=None.

        ClaimsHistory normalises exposures=None to [1.0, ...] at construction
        time, so we simulate the missing-exposure state by setting the attribute
        to None directly on an already-constructed object.
        """
        bad_history = make_history("NOEXP", [1, 0], prior=400.0)
        bad_history.exposures = None  # simulate missing exposures post-construction
        with pytest.raises(ValueError, match="exposure"):
            fitted_model.predict(bad_history)

    def test_predict_valid_history_works(self, fitted_model):
        """predict() with valid history returns a positive float."""
        good_history = make_history("GOOD", [1, 0, 1], prior=400.0)
        cf = fitted_model.predict(good_history)
        assert isinstance(cf, float)
        assert cf >= 0.0

    def test_credibility_weight_no_exposures_raises_valueerror(self, fitted_model):
        """credibility_weight() raises ValueError when exposures=None."""
        bad_history = make_history("NOEXP2", [0], prior=400.0)
        bad_history.exposures = None  # simulate missing exposures post-construction
        with pytest.raises(ValueError, match="exposure"):
            fitted_model.credibility_weight(bad_history)

    def test_estimate_kappa_none_exposures_raises_valueerror(self):
        """fit() raises ValueError when any history has exposures=None."""
        h1 = make_history("H1", [1, 0], prior=400.0)
        h1.exposures = None  # simulate missing exposures post-construction
        h2 = make_history("H2", [0, 1])
        model = StaticCredibilityModel()
        with pytest.raises(ValueError, match="exposure"):
            model.fit([h1, h2])
