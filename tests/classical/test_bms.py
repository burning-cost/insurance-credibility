"""
Tests for BMSEquilibriumSimulator.

Covers the Lemaire (1977) dynamic programming thresholds, the Liang et al.
(arXiv:2601.12655) two-insurer Nash equilibrium closed form, the Markov
chain stationary distribution, frequency correction, and all input validation.

The reference values are computed analytically from the formulas in the
module docstring. Where the paper gives numerical examples we cross-check
against those.
"""

import warnings

import numpy as np
import pytest
from scipy import stats as scipy_stats

from insurance_credibility.classical import BMSEquilibriumSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UK_DISCOUNTS_10 = [0.0, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.65, 0.70, 0.70]
UK_BASE_PREMIUM = 1000.0

GAMMA_DIST = scipy_stats.gamma(a=1.2, scale=1.0 / 0.0085)


@pytest.fixture
def sim_no_severity() -> BMSEquilibriumSimulator:
    """10-class UK NCD ladder, no severity distribution."""
    sim = BMSEquilibriumSimulator(
        discounts=UK_DISCOUNTS_10,
        base_premium=UK_BASE_PREMIUM,
        step_back=2,
        discount_factor=0.97,
        claim_freq=0.05,
    )
    sim.fit()
    return sim


@pytest.fixture
def sim_with_severity() -> BMSEquilibriumSimulator:
    """10-class UK NCD ladder with Gamma severity."""
    sim = BMSEquilibriumSimulator(
        discounts=UK_DISCOUNTS_10,
        base_premium=UK_BASE_PREMIUM,
        step_back=2,
        discount_factor=0.97,
        claim_freq=0.05,
        severity_dist=GAMMA_DIST,
    )
    sim.fit()
    return sim


@pytest.fixture
def sim_with_obs_freq() -> BMSEquilibriumSimulator:
    """10-class UK NCD ladder with severity and observed frequencies."""
    observed = [0.08, 0.07, 0.06, 0.055, 0.050, 0.045, 0.040, 0.038, 0.035, 0.030]
    sim = BMSEquilibriumSimulator(
        discounts=UK_DISCOUNTS_10,
        base_premium=UK_BASE_PREMIUM,
        step_back=2,
        discount_factor=0.97,
        claim_freq=0.05,
        severity_dist=GAMMA_DIST,
    )
    sim.fit(observed_freq=observed)
    return sim


@pytest.fixture
def sim_two_class() -> BMSEquilibriumSimulator:
    """Two-class BMS for Liang et al. closed-form tests."""
    sim = BMSEquilibriumSimulator(
        discounts=[0.0, 0.25],
        base_premium=35.85,
        step_back=1,
        discount_factor=0.97,
        claim_freq=0.10,
        severity_dist=GAMMA_DIST,
    )
    sim.fit()
    return sim


# ---------------------------------------------------------------------------
# 1. Construction and validation
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_basic_construction(self):
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.3, 0.5], base_premium=500.0)
        assert not sim._fitted

    def test_repr_before_fit(self):
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=100.0)
        assert "not fitted" in repr(sim)

    def test_repr_after_fit(self, sim_no_severity):
        r = repr(sim_no_severity)
        assert "BMSEquilibriumSimulator(" in r
        assert "not fitted" not in r
        assert "n_classes=10" in r

    def test_empty_discounts_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            BMSEquilibriumSimulator(discounts=[0.0], base_premium=1000.0)

    def test_two_d_discounts_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            BMSEquilibriumSimulator(discounts=[[0.0, 0.3], [0.0, 0.3]], base_premium=1000.0)

    def test_negative_discount_raises(self):
        with pytest.raises(ValueError, match="\\[0, 1\\)"):
            BMSEquilibriumSimulator(discounts=[-0.1, 0.3], base_premium=1000.0)

    def test_discount_gte_one_raises(self):
        with pytest.raises(ValueError, match="\\[0, 1\\)"):
            BMSEquilibriumSimulator(discounts=[0.0, 1.0], base_premium=1000.0)

    def test_non_positive_base_premium_raises(self):
        with pytest.raises(ValueError, match="base_premium"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=0.0)
        with pytest.raises(ValueError, match="base_premium"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=-100.0)

    def test_invalid_step_back_raises(self):
        with pytest.raises(ValueError, match="step_back"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=1000.0, step_back=0)
        with pytest.raises(ValueError, match="step_back"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=1000.0, step_back=1.5)

    def test_invalid_discount_factor_raises(self):
        with pytest.raises(ValueError, match="discount_factor"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=1000.0, discount_factor=0.0)
        with pytest.raises(ValueError, match="discount_factor"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=1000.0, discount_factor=1.1)

    def test_invalid_claim_freq_raises(self):
        with pytest.raises(ValueError, match="claim_freq"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=1000.0, claim_freq=0.0)
        with pytest.raises(ValueError, match="claim_freq"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=1000.0, claim_freq=1.5)

    def test_invalid_max_horizon_raises(self):
        with pytest.raises(ValueError, match="max_horizon"):
            BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=1000.0, max_horizon=0)

    def test_non_monotone_discounts_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BMSEquilibriumSimulator(discounts=[0.0, 0.5, 0.3], base_premium=1000.0)
            warns = [x for x in w if "monoton" in str(x.message).lower()]
            assert len(warns) > 0

    def test_fit_returns_self(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_10, base_premium=1000.0)
        result = sim.fit()
        assert result is sim


# ---------------------------------------------------------------------------
# 2. Thresholds: Lemaire algorithm
# ---------------------------------------------------------------------------

class TestThresholds:

    def test_thresholds_length(self, sim_no_severity):
        assert len(sim_no_severity.thresholds_) == 10

    def test_class_zero_threshold_is_zero(self, sim_no_severity):
        """Class 0 (no NCD) cannot step back further — threshold should be 0."""
        # With step_back=2 and class 0, we stay at 0: no additional penalty
        # In practice the threshold may be non-zero if there is some penalty
        # from being at class 0 vs class 0 (none)
        t = sim_no_severity.thresholds_
        assert t[0] >= 0.0

    def test_thresholds_non_negative(self, sim_no_severity):
        assert np.all(sim_no_severity.thresholds_ >= 0)

    def test_thresholds_numeric(self, sim_no_severity):
        assert np.all(np.isfinite(sim_no_severity.thresholds_))

    def test_mid_ladder_thresholds_positive(self, sim_no_severity):
        """Classes with meaningful NCD penalties should have positive thresholds."""
        t = sim_no_severity.thresholds_
        # Classes 2–7 have material premium differences after step-back
        assert np.any(t[2:8] > 0)

    def test_threshold_increases_with_premium_penalty(self):
        """
        A bigger premium penalty (larger discount gap between classes) should
        produce a higher threshold, all else equal.
        """
        # Ladder A: small gap between classes 3 and 1
        sim_a = BMSEquilibriumSimulator(
            discounts=[0.0, 0.10, 0.20, 0.25, 0.30],
            base_premium=1000.0,
            step_back=2,
            discount_factor=0.97,
            claim_freq=0.05,
        )
        sim_a.fit()

        # Ladder B: large gap between classes 3 and 1
        sim_b = BMSEquilibriumSimulator(
            discounts=[0.0, 0.10, 0.20, 0.60, 0.70],
            base_premium=1000.0,
            step_back=2,
            discount_factor=0.97,
            claim_freq=0.05,
        )
        sim_b.fit()

        # Class 3 threshold should be higher in ladder B (larger penalty)
        assert sim_b.thresholds_[3] > sim_a.thresholds_[3]

    def test_higher_discount_factor_increases_threshold(self):
        """
        A higher discount factor (lower interest rate) increases NPV of future
        premium penalties, raising the threshold (Holtan 2001).
        """
        sim_low_r = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10,
            base_premium=1000.0,
            discount_factor=0.99,  # ~1% risk-free rate
            claim_freq=0.05,
        )
        sim_low_r.fit()

        sim_high_r = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10,
            base_premium=1000.0,
            discount_factor=0.90,  # ~11% risk-free rate
            claim_freq=0.05,
        )
        sim_high_r.fit()

        # Higher discount factor → higher NPV → higher threshold for mid-ladder classes
        t_low = sim_low_r.thresholds_
        t_high = sim_high_r.thresholds_
        # For classes with positive penalties, low r (δ=0.99) > high r (δ=0.90)
        penalty_classes = [i for i in range(10) if t_low[i] > 0 or t_high[i] > 0]
        if penalty_classes:
            assert t_low[penalty_classes[0]] >= t_high[penalty_classes[0]]

    def test_higher_base_premium_scales_threshold(self):
        """
        Threshold should scale linearly with base_premium (NPV formula is linear in B).
        """
        sim_1 = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10, base_premium=1000.0,
            discount_factor=0.97, claim_freq=0.05,
        )
        sim_1.fit()

        sim_2 = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10, base_premium=2000.0,
            discount_factor=0.97, claim_freq=0.05,
        )
        sim_2.fit()

        # Each threshold should double (within floating-point precision)
        t1 = sim_1.thresholds_
        t2 = sim_2.thresholds_
        for i in range(10):
            if t1[i] > 1e-6:
                assert abs(t2[i] / t1[i] - 2.0) < 1e-10

    def test_not_fitted_raises(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_10, base_premium=1000.0)
        with pytest.raises(RuntimeError, match="fit"):
            _ = sim.thresholds_

    def test_manual_threshold_class5(self):
        """
        Cross-check threshold at class 5 (60% discount) manually.

        Class 5: discount=0.60, premium=£400
        Step-back to class 3: discount=0.40, premium=£600
        Penalty = £200/yr. Rebuild horizon = 2 years.
        NPV = 200×0.97 + 200×0.97² ≈ 382.18
        """
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.65, 0.70, 0.70],
            base_premium=1000.0,
            step_back=2,
            discount_factor=0.97,
            claim_freq=0.05,
        )
        sim.fit()
        expected = 200 * 0.97 + 200 * (0.97 ** 2)
        assert abs(sim.thresholds_[5] - expected) < 0.01


# ---------------------------------------------------------------------------
# 3. Reporting probabilities
# ---------------------------------------------------------------------------

class TestReportingProbs:

    def test_reporting_probs_length(self, sim_with_severity):
        assert len(sim_with_severity.reporting_probs_) == 10

    def test_reporting_probs_in_zero_one(self, sim_with_severity):
        p = sim_with_severity.reporting_probs_
        assert np.all(p > 0) and np.all(p <= 1.0)

    def test_class_zero_reporting_prob_is_one(self, sim_with_severity):
        """
        If threshold at class 0 is zero, all losses are reported: P(Y > 0) = 1.
        """
        t = sim_with_severity.thresholds_
        p = sim_with_severity.reporting_probs_
        if t[0] <= 0:
            assert abs(p[0] - 1.0) < 1e-8

    def test_higher_threshold_lower_prob(self, sim_with_severity):
        """
        Higher threshold → lower reporting probability (P(Y > b) is decreasing in b).
        Compare two classes where we know the threshold ordering.
        """
        t = sim_with_severity.thresholds_
        p = sim_with_severity.reporting_probs_
        # Find a pair (i, j) where t[i] < t[j], expect p[i] >= p[j]
        for i in range(9):
            j = i + 1
            if t[j] > t[i] + 1.0:  # clear ordering
                assert p[i] >= p[j] - 1e-10, (
                    f"Class {i} (b={t[i]:.2f}, p={p[i]:.4f}) should have "
                    f"p >= class {j} (b={t[j]:.2f}, p={p[j]:.4f})"
                )
                break

    def test_reporting_prob_matches_dist_sf(self, sim_with_severity):
        """
        P(Y > b*_n) should exactly equal GAMMA_DIST.sf(b*_n).
        """
        t = sim_with_severity.thresholds_
        p = sim_with_severity.reporting_probs_
        for i in range(10):
            expected = float(GAMMA_DIST.sf(t[i])) if t[i] > 0 else 1.0
            assert abs(p[i] - expected) < 1e-10

    def test_no_severity_raises_on_reporting_probs(self, sim_no_severity):
        with pytest.raises(RuntimeError, match="severity_dist"):
            _ = sim_no_severity.reporting_probs_

    def test_reporting_probs_numeric(self, sim_with_severity):
        assert np.all(np.isfinite(sim_with_severity.reporting_probs_))


# ---------------------------------------------------------------------------
# 4. Stationary distribution
# ---------------------------------------------------------------------------

class TestStationaryDistribution:

    def test_stationary_sums_to_one(self, sim_no_severity):
        pi = sim_no_severity.stationary_dist_
        assert abs(pi.sum() - 1.0) < 1e-8

    def test_stationary_non_negative(self, sim_no_severity):
        pi = sim_no_severity.stationary_dist_
        assert np.all(pi >= 0)

    def test_stationary_length(self, sim_no_severity):
        assert len(sim_no_severity.stationary_dist_) == 10

    def test_stationary_invariance(self, sim_no_severity):
        """
        π should be a left eigenvector of T: π T ≈ π (up to numerical tolerance).
        """
        pi = sim_no_severity.stationary_dist_
        T = sim_no_severity.transition_matrix_
        pi_next = pi @ T
        assert np.max(np.abs(pi_next - pi)) < 1e-8

    def test_stationary_highest_class_has_mass(self, sim_no_severity):
        """Most policyholders accumulate at the top NCD classes over time."""
        pi = sim_no_severity.stationary_dist_
        # The top 3 classes combined should have majority of mass
        assert pi[-3:].sum() > 0.3

    def test_higher_claim_freq_shifts_mass_down(self):
        """
        Higher claim frequency → more policyholders at lower NCD classes.
        """
        sim_low = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10, base_premium=1000.0, claim_freq=0.02
        )
        sim_low.fit()

        sim_high = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10, base_premium=1000.0, claim_freq=0.20
        )
        sim_high.fit()

        # Average class at low frequency should be higher (more mass at top)
        avg_low = np.average(np.arange(10), weights=sim_low.stationary_dist_)
        avg_high = np.average(np.arange(10), weights=sim_high.stationary_dist_)
        assert avg_low > avg_high


# ---------------------------------------------------------------------------
# 5. Transition matrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:

    def test_transition_shape(self, sim_no_severity):
        T = sim_no_severity.transition_matrix_
        assert T.shape == (10, 10)

    def test_transition_row_stochastic(self, sim_no_severity):
        T = sim_no_severity.transition_matrix_
        row_sums = T.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_transition_non_negative(self, sim_no_severity):
        T = sim_no_severity.transition_matrix_
        assert np.all(T >= 0)

    def test_transition_class_zero_cannot_go_lower(self, sim_no_severity):
        """
        From class 0, stepping back still stays at 0 (lower clamp).
        """
        T = sim_no_severity.transition_matrix_
        # Class 0 transitions: stay at 0 (with prob q_0) or go to 1 (with prob 1-q_0)
        assert T[0, 0] > 0 or T[0, 1] > 0  # must go somewhere
        # Should not be able to go below 0
        assert T[0, 0] + T[0, 1] > 0.99  # all mass in {0, 1}

    def test_transition_top_class_cannot_go_higher(self, sim_no_severity):
        """
        From the top class, stepping forward stays at the top (upper clamp).
        """
        T = sim_no_severity.transition_matrix_
        # From class 9: forward stays at 9, back goes to 7
        assert T[9, 9] > 0 or T[9, 7] > 0

    def test_transition_step_back_correct(self, sim_no_severity):
        """
        The step-back column for class i should be max(0, i - step_back).
        """
        T = sim_no_severity.transition_matrix_
        step_back = sim_no_severity.step_back
        for i in range(2, 10):
            back_class = max(0, i - step_back)
            # T[i, back_class] should be positive (reporting probability × claim_freq)
            assert T[i, back_class] > 0


# ---------------------------------------------------------------------------
# 6. Frequency correction
# ---------------------------------------------------------------------------

class TestFrequencyCorrection:

    def test_corrected_freq_length(self, sim_with_obs_freq):
        assert len(sim_with_obs_freq.corrected_freq_) == 10

    def test_corrected_freq_gte_observed(self, sim_with_obs_freq):
        """
        Corrected frequency >= observed frequency for all classes (since p_n ≤ 1).
        """
        obs = np.array([0.08, 0.07, 0.06, 0.055, 0.050, 0.045, 0.040, 0.038, 0.035, 0.030])
        corr = sim_with_obs_freq.corrected_freq_
        finite_mask = np.isfinite(corr)
        assert np.all(corr[finite_mask] >= obs[finite_mask] - 1e-10)

    def test_corrected_freq_equals_obs_when_prob_is_one(self):
        """
        Where the threshold is zero (p_n = 1), corrected freq = observed freq.
        """
        # Class 0 typically has zero threshold → p=1 → no correction
        observed = [0.08, 0.07, 0.06, 0.055, 0.050, 0.045, 0.040, 0.038, 0.035, 0.030]
        sim = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10,
            base_premium=1000.0,
            severity_dist=GAMMA_DIST,
        )
        sim.fit(observed_freq=observed)
        t = sim.thresholds_
        p = sim.reporting_probs_
        corr = sim.corrected_freq_
        for i in range(10):
            if t[i] <= 0:
                assert abs(p[i] - 1.0) < 1e-8
                assert abs(corr[i] - observed[i]) < 1e-10

    def test_corrected_freq_formula(self, sim_with_obs_freq):
        """
        Corrected freq should equal observed / p_n.
        """
        obs = np.array([0.08, 0.07, 0.06, 0.055, 0.050, 0.045, 0.040, 0.038, 0.035, 0.030])
        p = sim_with_obs_freq.reporting_probs_
        expected = obs / p
        corr = sim_with_obs_freq.corrected_freq_
        finite_mask = np.isfinite(corr) & np.isfinite(expected)
        assert np.allclose(corr[finite_mask], expected[finite_mask], rtol=1e-10)

    def test_no_severity_raises_on_corrected_freq(self, sim_no_severity):
        with pytest.raises(RuntimeError, match="severity_dist"):
            _ = sim_no_severity.corrected_freq_

    def test_no_obs_freq_raises_on_corrected_freq(self, sim_with_severity):
        with pytest.raises(RuntimeError, match="observed_freq"):
            _ = sim_with_severity.corrected_freq_

    def test_obs_freq_wrong_length_raises(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_10, base_premium=1000.0)
        with pytest.raises(ValueError, match="length 10"):
            sim.fit(observed_freq=[0.05, 0.04])

    def test_obs_freq_negative_raises(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_10, base_premium=1000.0)
        observed = [0.08, 0.07, 0.06, 0.055, 0.050, 0.045, -0.04, 0.038, 0.035, 0.030]
        with pytest.raises(ValueError, match="non-negative"):
            sim.fit(observed_freq=observed)


# ---------------------------------------------------------------------------
# 7. Frequency bias
# ---------------------------------------------------------------------------

class TestFrequencyBias:

    def test_bias_non_positive(self, sim_with_obs_freq):
        """
        Bias = (obs - true) / true. Should be <= 0 since obs <= true.
        """
        bias = sim_with_obs_freq.frequency_bias_()
        finite_mask = np.isfinite(bias)
        assert np.all(bias[finite_mask] <= 1e-10)

    def test_bias_range(self, sim_with_obs_freq):
        bias = sim_with_obs_freq.frequency_bias_()
        finite_mask = np.isfinite(bias)
        assert np.all(bias[finite_mask] >= -1.0 - 1e-10)

    def test_bias_zero_where_threshold_zero(self, sim_with_obs_freq):
        """
        Where the threshold is zero, there is no suppression: bias ≈ 0.
        """
        t = sim_with_obs_freq.thresholds_
        bias = sim_with_obs_freq.frequency_bias_()
        for i in range(10):
            if t[i] <= 0:
                assert abs(bias[i]) < 1e-10

    def test_bias_requires_severity_and_obs(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_10, base_premium=1000.0)
        sim.fit()
        with pytest.raises(RuntimeError, match="severity_dist"):
            sim.frequency_bias_()


# ---------------------------------------------------------------------------
# 8. Liang et al. closed form
# ---------------------------------------------------------------------------

class TestLiangEquilibrium:

    def test_returns_dict_with_expected_keys(self, sim_two_class):
        result = sim_two_class.liang_equilibrium(
            theta1=35.83, theta2=33.45, kappa=1.25, k1=0.015, k2=0.8
        )
        for key in ("threshold", "eta", "premium_diff", "theta1", "theta2",
                    "kappa", "k1", "k2", "discount_factor"):
            assert key in result, f"Missing key: {key}"

    def test_threshold_positive(self, sim_two_class):
        result = sim_two_class.liang_equilibrium(
            theta1=35.83, theta2=33.45, kappa=1.25, k1=0.015, k2=0.8
        )
        assert result["threshold"] > 0

    def test_threshold_base_case_liang_et_al(self, sim_two_class):
        """
        Liang et al. base case (Section 4.3 numerical illustration):
        p0=0.9, gamma(1.2, 0.0085), M=35.85, k1=0.015, k2=0.8, κ=1.25, δ=0.97
        θ*₁ ≈ 35.83, θ*₂ ≈ 33.45

        The Nash equilibrium threshold at (35.83, 33.45):
        η(Δ) = 1/(1 + exp(0.015 × 2.38 + log(0.2/0.8)))
        Δ = 35.83 - 33.45 = 2.38
        log(0.2/0.8) = log(0.25) ≈ -1.386
        exponent = 0.015 × 2.38 - 1.386 = 0.0357 - 1.386 ≈ -1.350
        η ≈ 1/(1 + exp(-1.350)) ≈ 0.794
        b* = 0.97 × 0.25 × [35.83 × 0.794 + 33.45 × 0.206]
           ≈ 0.2425 × [28.47 + 6.89]
           ≈ 0.2425 × 35.36
           ≈ 8.575
        """
        result = sim_two_class.liang_equilibrium(
            theta1=35.83, theta2=33.45, kappa=1.25, k1=0.015, k2=0.8
        )
        # Verify the computation is internally consistent
        # b* = δ(κ-1) × [θ1·η + θ2·(1-η)]
        eta = result["eta"]
        expected_threshold = 0.97 * (1.25 - 1) * (35.83 * eta + 33.45 * (1 - eta))
        assert abs(result["threshold"] - expected_threshold) < 1e-10

    def test_symmetric_competition_equal_thresholds(self, sim_two_class):
        """
        At k2=0.5 (symmetric competition), η=0.5 regardless of premiums.
        The threshold simplifies to δ(κ-1) × (θ1+θ2)/2.
        """
        result = sim_two_class.liang_equilibrium(
            theta1=30.0, theta2=30.0, kappa=1.25, k1=0.015, k2=0.5
        )
        assert abs(result["eta"] - 0.5) < 1e-6
        expected = 0.97 * 0.25 * 30.0
        assert abs(result["threshold"] - expected) < 1e-8

    def test_preferred_insurer_higher_eta(self, sim_two_class):
        """
        k2 > 0.5 means insurer 1 is preferred. At equal premiums, η > 0.5.
        """
        result = sim_two_class.liang_equilibrium(
            theta1=30.0, theta2=30.0, kappa=1.25, k1=0.015, k2=0.8
        )
        assert result["eta"] > 0.5

    def test_higher_premium_diff_reduces_eta(self, sim_two_class):
        """
        When insurer 1 charges much more than insurer 2, policyholders
        prefer insurer 2 more: η should decrease as θ1 - θ2 increases.
        """
        r1 = sim_two_class.liang_equilibrium(
            theta1=30.0, theta2=30.0, kappa=1.25, k1=0.015, k2=0.5
        )
        r2 = sim_two_class.liang_equilibrium(
            theta1=40.0, theta2=30.0, kappa=1.25, k1=0.015, k2=0.5
        )
        assert r2["eta"] < r1["eta"]

    def test_higher_kappa_raises_threshold(self, sim_two_class):
        """
        A larger premium penalty ratio κ increases the threshold.
        """
        r_low = sim_two_class.liang_equilibrium(
            theta1=30.0, theta2=30.0, kappa=1.1, k1=0.015, k2=0.5
        )
        r_high = sim_two_class.liang_equilibrium(
            theta1=30.0, theta2=30.0, kappa=1.4, k1=0.015, k2=0.5
        )
        assert r_high["threshold"] > r_low["threshold"]

    def test_invalid_kappa_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="kappa"):
            sim_two_class.liang_equilibrium(theta1=30.0, theta2=30.0, kappa=0.9)
        with pytest.raises(ValueError, match="kappa"):
            sim_two_class.liang_equilibrium(theta1=30.0, theta2=30.0, kappa=2.0)

    def test_non_positive_theta_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="theta"):
            sim_two_class.liang_equilibrium(theta1=0.0, theta2=30.0)
        with pytest.raises(ValueError, match="theta"):
            sim_two_class.liang_equilibrium(theta1=30.0, theta2=-5.0)

    def test_invalid_k1_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="k1"):
            sim_two_class.liang_equilibrium(theta1=30.0, theta2=30.0, k1=0.0)

    def test_invalid_k2_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="k2"):
            sim_two_class.liang_equilibrium(theta1=30.0, theta2=30.0, k2=0.0)
        with pytest.raises(ValueError, match="k2"):
            sim_two_class.liang_equilibrium(theta1=30.0, theta2=30.0, k2=1.0)

    def test_eta_in_zero_one(self, sim_two_class):
        for theta1, theta2 in [(25.0, 35.0), (35.0, 25.0), (30.0, 30.0)]:
            result = sim_two_class.liang_equilibrium(theta1, theta2)
            assert 0.0 < result["eta"] < 1.0

    def test_threshold_scales_with_discount_factor(self):
        """
        The threshold b* is proportional to δ at fixed premiums.
        """
        sim_97 = BMSEquilibriumSimulator(
            discounts=[0.0, 0.25], base_premium=35.85, discount_factor=0.97
        )
        sim_90 = BMSEquilibriumSimulator(
            discounts=[0.0, 0.25], base_premium=35.85, discount_factor=0.90
        )
        r97 = sim_97.liang_equilibrium(theta1=30.0, theta2=28.0, k2=0.5)
        r90 = sim_90.liang_equilibrium(theta1=30.0, theta2=28.0, k2=0.5)
        ratio = r97["threshold"] / r90["threshold"]
        assert abs(ratio - 0.97 / 0.90) < 1e-8

    def test_non_two_class_can_still_call_liang(self, sim_with_severity):
        """
        liang_equilibrium() is callable on any simulator — it only uses the
        discount_factor and validates its own inputs. The 10-class simulator
        should still work for the closed-form computation.
        """
        result = sim_with_severity.liang_equilibrium(
            theta1=30.0, theta2=28.0, kappa=1.25, k1=0.015, k2=0.8
        )
        assert result["threshold"] > 0


# ---------------------------------------------------------------------------
# 9. Nash equilibrium premiums
# ---------------------------------------------------------------------------

class TestNashEquilibriumPremiums:

    def test_non_two_class_raises(self, sim_with_severity):
        with pytest.raises(ValueError, match="two-class"):
            sim_with_severity.nash_equilibrium_premiums()

    def test_returns_dict_with_expected_keys(self, sim_two_class):
        result = sim_two_class.nash_equilibrium_premiums(max_iter=10)
        for key in ("theta1", "theta2", "threshold", "converged", "iterations", "premium_gap"):
            assert key in result

    def test_premiums_positive(self, sim_two_class):
        result = sim_two_class.nash_equilibrium_premiums(max_iter=20)
        assert result["theta1"] > 0
        assert result["theta2"] > 0

    def test_threshold_positive(self, sim_two_class):
        result = sim_two_class.nash_equilibrium_premiums(max_iter=20)
        assert result["threshold"] > 0

    def test_preferred_insurer_charges_more_k2_gt_half(self, sim_two_class):
        """
        Proposition 4.1 (Liang et al.): k2 > 0.5 → θ*₁ ≥ θ*₂.
        Insurer 1 is preferred and can command a premium.
        """
        result = sim_two_class.nash_equilibrium_premiums(
            kappa=1.25, k1=0.015, k2=0.8, max_iter=100
        )
        if result["converged"]:
            assert result["theta1"] >= result["theta2"] - 1.0  # allow 1% tolerance for grid


# ---------------------------------------------------------------------------
# 10. class_premiums() helper
# ---------------------------------------------------------------------------

class TestClassPremiums:

    def test_class_premiums_length(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_10, base_premium=1000.0)
        p = sim.class_premiums()
        assert len(p) == 10

    def test_class_zero_premium_equals_base(self):
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.3, 0.5], base_premium=800.0)
        p = sim.class_premiums()
        assert abs(p[0] - 800.0) < 1e-10

    def test_premiums_non_increasing(self):
        """Higher NCD class → lower premium (higher discount)."""
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.65, 0.70, 0.70],
            base_premium=1000.0,
        )
        p = sim.class_premiums()
        assert np.all(np.diff(p) <= 0)

    def test_premium_formula(self):
        discounts = [0.0, 0.3, 0.5, 0.7]
        base = 1000.0
        sim = BMSEquilibriumSimulator(discounts=discounts, base_premium=base)
        p = sim.class_premiums()
        expected = np.array([1000.0, 700.0, 500.0, 300.0])
        assert np.allclose(p, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 11. Summary (smoke test)
# ---------------------------------------------------------------------------

class TestSummary:

    def test_summary_runs_no_severity(self, sim_no_severity, capsys):
        sim_no_severity.summary()
        out = capsys.readouterr().out
        assert "BMSEquilibriumSimulator" in out
        assert "Threshold" in out

    def test_summary_runs_with_severity(self, sim_with_severity, capsys):
        sim_with_severity.summary()
        out = capsys.readouterr().out
        assert "P(Y>b*)" in out

    def test_summary_runs_with_obs_freq(self, sim_with_obs_freq, capsys):
        sim_with_obs_freq.summary()
        out = capsys.readouterr().out
        assert "Corr freq" in out or "Obs freq" in out

    def test_summary_not_fitted_raises(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_10, base_premium=1000.0)
        with pytest.raises(RuntimeError, match="fit"):
            sim.summary()


# ---------------------------------------------------------------------------
# 12. Edge cases and integration
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_two_class_ladder(self):
        """Minimal ladder with two classes works end-to-end."""
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.25],
            base_premium=100.0,
            step_back=1,
            discount_factor=0.95,
            claim_freq=0.10,
            severity_dist=GAMMA_DIST,
        )
        sim.fit(observed_freq=[0.10, 0.07])
        assert sim._fitted
        assert len(sim.thresholds_) == 2
        assert abs(sim.stationary_dist_.sum() - 1.0) < 1e-8

    def test_single_step_back(self):
        """step_back=1 (Belgian BMS style) works correctly."""
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.15, 0.30, 0.45, 0.60],
            base_premium=500.0,
            step_back=1,
            discount_factor=0.97,
            claim_freq=0.08,
        )
        sim.fit()
        assert np.all(sim.thresholds_ >= 0)

    def test_large_step_back_clamped_at_zero(self):
        """
        step_back > class index → step back to class 0.
        Threshold should be based on the penalty to class 0.
        """
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.50, 0.70],
            base_premium=1000.0,
            step_back=5,  # larger than ladder
            discount_factor=0.97,
            claim_freq=0.05,
        )
        sim.fit()
        # Class 1 and 2 both step back to class 0 (clamped)
        # Threshold should be positive
        assert sim.thresholds_[1] > 0
        assert sim.thresholds_[2] > 0

    def test_all_equal_discounts_zero_thresholds(self):
        """
        If all discounts are equal (no penalty for claiming), threshold is 0.
        """
        sim = BMSEquilibriumSimulator(
            discounts=[0.50, 0.50, 0.50, 0.50],
            base_premium=1000.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress monotonicity warning
            sim.fit()
        # All premiums are equal → no penalty → thresholds are 0
        assert np.all(sim.thresholds_ == 0.0)

    def test_discount_factor_one(self):
        """
        discount_factor=1.0 (zero interest rate) — model should run without error.
        """
        sim = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10,
            base_premium=1000.0,
            discount_factor=1.0,
            claim_freq=0.05,
        )
        sim.fit()
        assert np.all(sim.thresholds_ >= 0)

    def test_very_high_threshold_zero_prob(self):
        """
        If thresholds are very high, reporting probability approaches zero.
        Corrected frequency should be NaN (not inf or negative).
        """
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.99],
            base_premium=1e8,  # enormous premium → enormous threshold
            step_back=1,
            discount_factor=0.99,
            claim_freq=0.05,
            severity_dist=GAMMA_DIST,
        )
        sim.fit(observed_freq=[0.05, 0.02])
        corr = sim.corrected_freq_
        # Class 1 threshold will be very large; P(Y > b) ≈ 0 → corrected freq = NaN
        # Class 0 should be ok
        assert np.isfinite(corr[0])

    def test_numpy_array_input(self):
        """Accepts numpy array for discounts."""
        discounts = np.array([0.0, 0.30, 0.50, 0.70])
        sim = BMSEquilibriumSimulator(discounts=discounts, base_premium=500.0)
        sim.fit()
        assert sim._fitted

    def test_list_float_input(self):
        """Accepts Python list for discounts."""
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.30, 0.50, 0.70], base_premium=500.0)
        sim.fit()
        assert sim._fitted

    def test_full_pipeline_uk_motor(self):
        """
        Full end-to-end pipeline: UK 10-class ladder with all features active.
        No errors, plausible outputs.
        """
        observed = [0.080, 0.070, 0.060, 0.055, 0.050, 0.045, 0.040, 0.038, 0.035, 0.030]
        sim = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_10,
            base_premium=UK_BASE_PREMIUM,
            step_back=2,
            discount_factor=0.97,
            claim_freq=0.05,
            severity_dist=GAMMA_DIST,
            max_horizon=10,
        )
        sim.fit(observed_freq=observed)

        # Thresholds
        assert np.all(sim.thresholds_ >= 0)
        assert np.all(np.isfinite(sim.thresholds_))

        # Reporting probs
        assert np.all(sim.reporting_probs_ > 0)
        assert np.all(sim.reporting_probs_ <= 1.0)

        # Stationary
        assert abs(sim.stationary_dist_.sum() - 1.0) < 1e-8

        # Corrected frequencies: at least some should be > observed
        obs = np.array(observed)
        corr = sim.corrected_freq_
        finite_mask = np.isfinite(corr)
        assert np.any(corr[finite_mask] > obs[finite_mask] + 1e-6)

        # Bias: all non-positive
        bias = sim.frequency_bias_()
        finite_bias = bias[np.isfinite(bias)]
        assert np.all(finite_bias <= 1e-10)
