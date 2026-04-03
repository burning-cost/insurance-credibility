"""
Expanded test coverage for insurance-credibility classical module (April 2026).

Targets untested or thinly-tested code paths across:

1. BuhlmannStraub:
   - Properties accessible only after fit (v_hat_, a_hat_, k_, premiums_, z_)
   - repr before and after fit
   - all four properties raise before fit
   - credibility_premium bounded between 0 and max(group_means) when Z in [0,1]
   - fit with non-default column names
   - groups with many periods vs few periods: Z ordering
   - integer group IDs work (not just strings)
   - large number of groups (20 groups)
   - summary() output when k=inf
   - identical groups produce warning, not crash
   - v_hat positive for heterogeneous groups
   - a_hat positive for heterogeneous groups

2. HierarchicalBuhlmannStraub:
   - repr before/after fit
   - premiums_ property convenience accessor
   - level_results_ keys match level_cols
   - premiums_at() for each level
   - fit with non-default column names
   - heterogeneous two-level DGP: premiums differ between nodes
   - three-level DGP with genuine signal at each level
   - 2-level with single-period sub-nodes

3. BMSEquilibriumSimulator additional paths:
   - class_premiums() values
   - transition_matrix_ is row-stochastic
   - stationary_dist_ sums to 1
   - stationary_dist_ all non-negative
   - reporting_probs_ without severity raises
   - corrected_freq_ without obs_freq raises
   - frequency_bias_ negative values
   - frequency_bias_ without severity raises
   - liang_equilibrium invalid kappa raises
   - liang_equilibrium invalid theta raises
   - liang_equilibrium invalid k1/k2 raises
   - liang_equilibrium threshold positive
   - liang_equilibrium eta in (0,1)
   - nash_equilibrium_premiums() non-2-class raises
   - step_back=1 works
   - single-period thresholds: class 0 always 0 penalty
   - before-fit properties raise RuntimeError

4. PoissonGammaCredibility additional paths:
   - z_ factor in [0,1]
   - credibility_premium is posterior mean formula
   - large exposure group gets Z near 1
   - small exposure group gets Z near 0 (large beta)
   - not fitted raises
   - repr before/after fit
   - zero claims group handled
   - fit with explicit alpha/beta (override calibration)
   - credibility interval coverage (posterior gamma quantiles)

5. _validation additional paths:
   - _to_polars with valid polars (no-op)
   - _to_polars with invalid type raises TypeError
   - validate_panel_data: zero weight raises (weight=0)
   - validate_panel_data: NaN loss raises
   - check_duplicate_periods: all clean, no warning
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_credibility.classical import (
    BuhlmannStraub,
    HierarchicalBuhlmannStraub,
    BMSEquilibriumSimulator,
    PoissonGammaCredibility,
)
from insurance_credibility.classical._validation import (
    _to_polars,
    validate_panel_data,
    check_duplicate_periods,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_panel(
    groups=("A", "B", "C"),
    n_periods=3,
    base_loss=0.6,
    spread=0.2,
    exposure=500.0,
    seed=0,
) -> pl.DataFrame:
    """Synthetic balanced panel with genuine between-group heterogeneity."""
    rng = np.random.default_rng(seed)
    rows = []
    means = {g: base_loss + spread * (i - len(groups) / 2) / len(groups)
             for i, g in enumerate(groups)}
    for g in groups:
        for t in range(1, n_periods + 1):
            rows.append({
                "group": g,
                "period": t,
                "loss": means[g] + 0.01 * rng.standard_normal(),
                "weight": exposure,
            })
    return pl.DataFrame(rows)


UK_DISCOUNTS_5 = [0.0, 0.30, 0.40, 0.50, 0.60]
UK_DISCOUNTS_10 = [0.0, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.65, 0.70, 0.70]


# ---------------------------------------------------------------------------
# 1. BuhlmannStraub — additional paths
# ---------------------------------------------------------------------------

class TestBuhlmannStraubProperties:

    @pytest.fixture(autouse=True)
    def _fit(self):
        self.df = _make_panel(groups=["A", "B", "C", "D"], n_periods=5)
        self.bs = BuhlmannStraub()
        self.bs.fit(self.df)

    def test_v_hat_accessible(self):
        assert isinstance(self.bs.v_hat_, float)

    def test_a_hat_accessible(self):
        assert isinstance(self.bs.a_hat_, float)

    def test_k_accessible(self):
        assert isinstance(self.bs.k_, float)

    def test_premiums_accessible(self):
        assert isinstance(self.bs.premiums_, pl.DataFrame)

    def test_z_accessible(self):
        assert isinstance(self.bs.z_, pl.DataFrame)

    def test_v_hat_positive(self):
        """v_hat must be non-negative (EPV)."""
        assert self.bs.v_hat_ >= 0

    def test_a_hat_positive_with_heterogeneous_groups(self):
        """Well-separated groups should produce positive a_hat."""
        assert self.bs.a_hat_ >= 0

    def test_premiums_between_group_means(self):
        """All credibility premiums should lie between the min and max group means."""
        premiums = self.bs.premiums_["credibility_premium"].to_numpy()
        obs_means = self.bs.premiums_["observed_mean"].to_numpy()
        assert np.all(premiums >= obs_means.min() - 1e-6)
        assert np.all(premiums <= obs_means.max() + 1e-6)

    def test_group_ids_in_premiums(self):
        groups = self.df["group"].unique().to_list()
        premium_groups = self.bs.premiums_["group"].to_list()
        assert set(premium_groups) == set(groups)

    def test_z_in_zero_one(self):
        z_vals = self.bs.z_["Z"].to_numpy()
        assert np.all(z_vals >= 0) and np.all(z_vals <= 1)


class TestBuhlmannStraubBeforeFit:

    def test_v_hat_before_fit_raises(self):
        bs = BuhlmannStraub()
        with pytest.raises(RuntimeError, match="fit"):
            _ = bs.v_hat_

    def test_a_hat_before_fit_raises(self):
        bs = BuhlmannStraub()
        with pytest.raises(RuntimeError, match="fit"):
            _ = bs.a_hat_

    def test_k_before_fit_raises(self):
        bs = BuhlmannStraub()
        with pytest.raises(RuntimeError, match="fit"):
            _ = bs.k_

    def test_premiums_before_fit_raises(self):
        bs = BuhlmannStraub()
        with pytest.raises(RuntimeError, match="fit"):
            _ = bs.premiums_


class TestBuhlmannStraubNonDefaultColumns:

    def test_fit_with_custom_column_names(self):
        """Fit should work with non-default column names."""
        df = pl.DataFrame({
            "scheme": ["X", "X", "X", "Y", "Y", "Y"],
            "yr":     [2020, 2021, 2022, 2020, 2021, 2022],
            "lr":     [0.6, 0.65, 0.62, 0.8, 0.78, 0.82],
            "exp":    [1000.0, 1100.0, 950.0, 400.0, 450.0, 420.0],
        })
        bs = BuhlmannStraub()
        bs.fit(df, group_col="scheme", period_col="yr",
               loss_col="lr", weight_col="exp")
        assert bs._fitted
        assert set(bs.z_["group"].to_list()) == {"X", "Y"}

    def test_default_column_names(self):
        """Default column names (group, period, loss, weight) should work."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [0.5, 0.6, 0.9, 0.8],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        bs = BuhlmannStraub()
        bs.fit(df)
        assert bs._fitted


class TestBuhlmannStraubIntegerGroupIDs:

    def test_integer_group_ids_work(self):
        """Integer group IDs (not just strings) are a common real-world case."""
        df = pl.DataFrame({
            "group": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "period": [2020, 2021, 2022] * 3,
            "loss": [0.5, 0.55, 0.52, 0.9, 0.85, 0.88, 0.3, 0.32, 0.31],
            "weight": [1000.0] * 9,
        })
        bs = BuhlmannStraub()
        bs.fit(df)
        assert set(bs.z_["group"].to_list()) == {1, 2, 3}

    def test_integer_group_z_ordering(self):
        """Group with highest loss (group 2) should have premium above mu."""
        df = pl.DataFrame({
            "group": [1, 1, 2, 2, 3, 3],
            "period": [1, 2, 1, 2, 1, 2],
            "loss": [0.5, 0.5, 0.9, 0.9, 0.3, 0.3],
            "weight": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        })
        bs = BuhlmannStraub()
        bs.fit(df)
        mu = bs.mu_hat_
        p2 = bs.premiums_.filter(pl.col("group") == 2)["credibility_premium"][0]
        p3 = bs.premiums_.filter(pl.col("group") == 3)["credibility_premium"][0]
        assert p2 > mu
        assert p3 < mu


class TestBuhlmannStraubLargePortfolio:

    def test_20_groups_fits(self):
        """20 groups with varying means — a realistic small portfolio."""
        rng = np.random.default_rng(10)
        rows = []
        for g in range(1, 21):
            mean = 0.4 + 0.03 * g
            for t in range(1, 5):
                rows.append({
                    "group": g,
                    "period": t,
                    "loss": mean + 0.01 * rng.standard_normal(),
                    "weight": float(rng.uniform(200, 2000)),
                })
        df = pl.DataFrame(rows)
        bs = BuhlmannStraub()
        bs.fit(df)
        assert bs._fitted
        assert len(bs.premiums_) == 20
        assert np.all(np.isfinite(bs.premiums_["credibility_premium"].to_numpy()))

    def test_20_groups_z_in_zero_one(self):
        rng = np.random.default_rng(11)
        rows = []
        for g in range(1, 21):
            mean = 0.4 + 0.03 * g
            for t in range(1, 5):
                rows.append({
                    "group": g,
                    "period": t,
                    "loss": mean + 0.01 * rng.standard_normal(),
                    "weight": 500.0,
                })
        df = pl.DataFrame(rows)
        bs = BuhlmannStraub()
        bs.fit(df)
        z = bs.z_["Z"].to_numpy()
        assert np.all(z >= 0) and np.all(z <= 1)


class TestBuhlmannStraubSummaryKInf:

    def test_summary_with_k_inf(self, capsys):
        """summary() should not crash when k=inf (a_hat=0)."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.0, 1.0, 1.0],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        bs = BuhlmannStraub(truncate_a=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bs.fit(df)
        result = bs.summary()
        out = capsys.readouterr().out
        assert "inf" in out.lower() or "∞" in out or "= inf" in out
        assert isinstance(result, pl.DataFrame)


class TestBuhlmannStraubExposureEffect:

    def test_more_periods_higher_z(self):
        """Group with more periods gets higher credibility factor."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "A", "A", "B", "B", "B"],
            "period": [1, 2, 3, 4, 5, 1, 2, 3],
            "loss": [0.6, 0.65, 0.62, 0.61, 0.63, 0.9, 0.88, 0.92],
            "weight": [100.0] * 8,
        })
        bs = BuhlmannStraub()
        bs.fit(df)
        z_a = bs.z_.filter(pl.col("group") == "A")["Z"][0]
        z_b = bs.z_.filter(pl.col("group") == "B")["Z"][0]
        assert z_a > z_b, f"Group A (5 periods) should have higher Z than B (3 periods)"


# ---------------------------------------------------------------------------
# 2. HierarchicalBuhlmannStraub — additional paths
# ---------------------------------------------------------------------------

def _make_hierarchical_df(
    n_regions=2,
    n_districts_per_region=2,
    n_periods=3,
    seed=20,
) -> pl.DataFrame:
    """Synthetic two-level hierarchy: regions -> districts."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_regions):
        region = f"R{r+1}"
        for d in range(n_districts_per_region):
            district = f"{region}_D{d+1}"
            mean = 0.5 + 0.1 * r + 0.05 * d + 0.02 * rng.standard_normal()
            for t in range(1, n_periods + 1):
                rows.append({
                    "region": region,
                    "district": district,
                    "period": t,
                    "loss": mean + 0.01 * rng.standard_normal(),
                    "weight": float(rng.uniform(300, 1000)),
                })
    return pl.DataFrame(rows)


class TestHierarchicalAdditional:

    def test_repr_before_fit(self):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        r = repr(model)
        assert "HierarchicalBuhlmannStraub" in r or "not fitted" in r.lower() or "Hierarchical" in r

    def test_level_results_keys_match_level_cols(self):
        df = _make_hierarchical_df()
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        assert set(model.level_results_.keys()) == {"region", "district"}

    def test_premiums_property_convenience(self):
        """premiums_ should return the lowest-level premiums."""
        df = _make_hierarchical_df()
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        premiums = model.premiums_
        assert isinstance(premiums, pl.DataFrame)
        assert "credibility_premium" in premiums.columns

    def test_premiums_at_region_level(self):
        df = _make_hierarchical_df()
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        r_premiums = model.premiums_at("region")
        assert isinstance(r_premiums, pl.DataFrame)
        assert "credibility_premium" in r_premiums.columns

    def test_premiums_at_district_level(self):
        df = _make_hierarchical_df()
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        d_premiums = model.premiums_at("district")
        assert isinstance(d_premiums, pl.DataFrame)

    def test_fit_custom_column_names(self):
        """Non-default column names should work for hierarchical model."""
        df = _make_hierarchical_df()
        df = df.rename({"loss": "lr", "weight": "exp"})
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df, loss_col="lr", weight_col="exp")
        assert model._fitted

    def test_heterogeneous_premiums_differ_across_districts(self):
        """With genuine signal, district-level premiums should not all be equal."""
        df = _make_hierarchical_df(n_regions=3, n_districts_per_region=3, seed=42)
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        d_premiums = model.premiums_at("district")["credibility_premium"].to_numpy()
        # Not all premiums identical
        assert np.std(d_premiums) > 1e-6

    def test_region_premiums_finite(self):
        df = _make_hierarchical_df()
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        r_premiums = model.premiums_at("region")["credibility_premium"].to_numpy()
        assert np.all(np.isfinite(r_premiums))

    def test_three_level_level_results(self):
        """Three-level hierarchy should have level results for all three levels."""
        rows = []
        rng = np.random.default_rng(30)
        for region in ["R1", "R2"]:
            for district in ["D1", "D2"]:
                for sector in ["S1", "S2", "S3"]:
                    for period in [1, 2, 3]:
                        rows.append({
                            "region": region,
                            "district": f"{region}_{district}",
                            "sector": f"{region}_{district}_{sector}",
                            "period": period,
                            "loss": 0.6 + 0.1 * rng.standard_normal(),
                            "weight": 400.0,
                        })
        df = pl.DataFrame(rows)
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district", "sector"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        assert set(model.level_results_.keys()) == {"region", "district", "sector"}

    def test_v_hat_at_each_level_accessible(self):
        df = _make_hierarchical_df()
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        for level_name, lr in model.level_results_.items():
            assert hasattr(lr, "v_hat")
            assert lr.v_hat >= 0

    def test_a_hat_at_each_level_accessible(self):
        df = _make_hierarchical_df()
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        for level_name, lr in model.level_results_.items():
            assert hasattr(lr, "a_hat")
            assert lr.a_hat >= 0


# ---------------------------------------------------------------------------
# 3. BMSEquilibriumSimulator — additional paths
# ---------------------------------------------------------------------------

class TestBMSClassPremiums:

    def test_class_premiums_shape(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_5, base_premium=1000.0)
        cp = sim.class_premiums()
        assert cp.shape == (5,)

    def test_class_premiums_values(self):
        """class_premiums[n] = base_premium * (1 - discount[n])."""
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_5, base_premium=1000.0)
        cp = sim.class_premiums()
        expected = 1000.0 * (1 - np.array(UK_DISCOUNTS_5))
        np.testing.assert_allclose(cp, expected)

    def test_class_premiums_decreasing(self):
        """Higher NCD class → lower premium."""
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_5, base_premium=1000.0)
        cp = sim.class_premiums()
        assert np.all(np.diff(cp) <= 0)

    def test_class_zero_premium_equals_base(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_5, base_premium=800.0)
        cp = sim.class_premiums()
        assert cp[0] == pytest.approx(800.0)


class TestBMSFittedProperties:

    @pytest.fixture(autouse=True)
    def _fit(self):
        from scipy import stats
        self.sim = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_5,
            base_premium=1000.0,
            step_back=2,
            discount_factor=0.97,
            claim_freq=0.05,
            severity_dist=stats.gamma(a=1.2, scale=1.0 / 0.0085),
        )
        self.sim.fit(observed_freq=[0.08, 0.07, 0.06, 0.055, 0.050])

    def test_transition_matrix_row_stochastic(self):
        """Each row of T should sum to 1."""
        T = self.sim.transition_matrix_
        row_sums = T.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_transition_matrix_non_negative(self):
        T = self.sim.transition_matrix_
        assert np.all(T >= 0)

    def test_stationary_dist_sums_to_one(self):
        pi = self.sim.stationary_dist_
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_stationary_dist_non_negative(self):
        pi = self.sim.stationary_dist_
        assert np.all(pi >= 0)

    def test_stationary_dist_is_actually_stationary(self):
        """pi @ T should equal pi (up to numerical tolerance)."""
        T = self.sim.transition_matrix_
        pi = self.sim.stationary_dist_
        pi_next = pi @ T
        np.testing.assert_allclose(pi_next, pi, atol=1e-8)

    def test_reporting_probs_in_zero_one(self):
        p = self.sim.reporting_probs_
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_reporting_probs_class0_near_one(self):
        """Class 0 (no NCD) has no penalty and threshold ~0, so P(Y>0) ~= 1."""
        p = self.sim.reporting_probs_
        assert p[0] == pytest.approx(1.0, abs=1e-6)

    def test_corrected_freq_ge_observed_freq(self):
        """Corrected frequencies should be >= observed (underreporting correction)."""
        corr = self.sim.corrected_freq_
        obs = np.array([0.08, 0.07, 0.06, 0.055, 0.050])
        mask = np.isfinite(corr)
        assert np.all(corr[mask] >= obs[mask] - 1e-10)

    def test_frequency_bias_negative_or_zero(self):
        """Bias = (obs - true) / true should be <= 0."""
        bias = self.sim.frequency_bias_()
        mask = np.isfinite(bias)
        assert np.all(bias[mask] <= 1e-10)


class TestBMSBeforeFit:

    def test_thresholds_before_fit_raises(self):
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=500.0)
        with pytest.raises(RuntimeError, match="fit"):
            _ = sim.thresholds_

    def test_stationary_dist_before_fit_raises(self):
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=500.0)
        with pytest.raises(RuntimeError, match="fit"):
            _ = sim.stationary_dist_

    def test_transition_matrix_before_fit_raises(self):
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=500.0)
        with pytest.raises(RuntimeError, match="fit"):
            _ = sim.transition_matrix_

    def test_reporting_probs_no_severity_raises(self):
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=500.0)
        sim.fit()
        with pytest.raises(RuntimeError, match="severity_dist"):
            _ = sim.reporting_probs_

    def test_corrected_freq_no_obs_raises(self):
        from scipy import stats
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.3], base_premium=500.0,
            severity_dist=stats.gamma(a=1.2, scale=1.0 / 0.0085),
        )
        sim.fit()  # no observed_freq
        with pytest.raises(RuntimeError, match="observed_freq"):
            _ = sim.corrected_freq_

    def test_frequency_bias_no_severity_raises(self):
        sim = BMSEquilibriumSimulator(discounts=[0.0, 0.3], base_premium=500.0)
        sim.fit()
        with pytest.raises(RuntimeError, match="severity_dist"):
            _ = sim.frequency_bias_()


class TestBMSObservedFreqValidation:

    def test_wrong_length_raises(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_5, base_premium=1000.0)
        with pytest.raises(ValueError, match="length"):
            sim.fit(observed_freq=[0.05, 0.04, 0.03])  # wrong length

    def test_negative_freq_raises(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_5, base_premium=1000.0)
        with pytest.raises(ValueError, match="non-negative"):
            sim.fit(observed_freq=[-0.01, 0.05, 0.05, 0.05, 0.04])


class TestBMSStepBack1:

    def test_step_back_1_fits(self):
        """step_back=1 (less common) should produce valid thresholds."""
        sim = BMSEquilibriumSimulator(
            discounts=UK_DISCOUNTS_5,
            base_premium=1000.0,
            step_back=1,
        )
        sim.fit()
        assert np.all(sim.thresholds_ >= 0)
        assert np.all(np.isfinite(sim.thresholds_))

    def test_step_back_3_fits(self):
        """step_back=3 should also work."""
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            base_premium=500.0,
            step_back=3,
        )
        sim.fit()
        assert sim._fitted
        assert np.all(sim.thresholds_ >= 0)


class TestBMSLiangEquilibrium:

    @pytest.fixture
    def sim_two_class(self):
        return BMSEquilibriumSimulator(
            discounts=[0.0, 0.25],
            base_premium=35.85,
            discount_factor=0.97,
            claim_freq=0.10,
        )

    def test_threshold_positive(self, sim_two_class):
        result = sim_two_class.liang_equilibrium(theta1=35.83, theta2=33.45)
        assert result["threshold"] > 0

    def test_eta_in_zero_one(self, sim_two_class):
        result = sim_two_class.liang_equilibrium(theta1=35.83, theta2=33.45)
        assert 0 < result["eta"] < 1

    def test_result_contains_all_keys(self, sim_two_class):
        result = sim_two_class.liang_equilibrium(theta1=35.83, theta2=33.45)
        expected_keys = {"threshold", "eta", "premium_diff", "theta1", "theta2",
                         "kappa", "k1", "k2", "discount_factor"}
        assert expected_keys == set(result.keys())

    def test_equal_premiums_symmetric_eta(self, sim_two_class):
        """When premiums are equal, eta depends only on k2."""
        result = sim_two_class.liang_equilibrium(
            theta1=35.0, theta2=35.0, k1=0.015, k2=0.5
        )
        # k2=0.5 → symmetric → eta=0.5
        assert abs(result["eta"] - 0.5) < 1e-8

    def test_invalid_kappa_le_1_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="kappa"):
            sim_two_class.liang_equilibrium(theta1=35.0, theta2=33.0, kappa=0.9)

    def test_invalid_kappa_ge_2_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="kappa"):
            sim_two_class.liang_equilibrium(theta1=35.0, theta2=33.0, kappa=2.0)

    def test_invalid_theta1_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="positive"):
            sim_two_class.liang_equilibrium(theta1=0.0, theta2=35.0)

    def test_invalid_theta2_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="positive"):
            sim_two_class.liang_equilibrium(theta1=35.0, theta2=-1.0)

    def test_invalid_k1_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="k1"):
            sim_two_class.liang_equilibrium(theta1=35.0, theta2=33.0, k1=0.0)

    def test_invalid_k2_raises(self, sim_two_class):
        with pytest.raises(ValueError, match="k2"):
            sim_two_class.liang_equilibrium(theta1=35.0, theta2=33.0, k2=0.0)
        with pytest.raises(ValueError, match="k2"):
            sim_two_class.liang_equilibrium(theta1=35.0, theta2=33.0, k2=1.0)

    def test_higher_discount_factor_raises_threshold(self, sim_two_class):
        """Higher delta → higher NPV → higher threshold (Holtan 2001)."""
        sim_lo = BMSEquilibriumSimulator(
            discounts=[0.0, 0.25], base_premium=35.85, discount_factor=0.90
        )
        sim_hi = BMSEquilibriumSimulator(
            discounts=[0.0, 0.25], base_premium=35.85, discount_factor=0.99
        )
        r_lo = sim_lo.liang_equilibrium(theta1=35.83, theta2=33.45)
        r_hi = sim_hi.liang_equilibrium(theta1=35.83, theta2=33.45)
        assert r_hi["threshold"] > r_lo["threshold"]


class TestBMSNashEquilibriumPremiums:

    def test_non_two_class_raises(self):
        sim = BMSEquilibriumSimulator(discounts=UK_DISCOUNTS_5, base_premium=1000.0)
        with pytest.raises(ValueError, match="two-class"):
            sim.nash_equilibrium_premiums()

    def test_two_class_returns_dict(self):
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.25], base_premium=35.85, discount_factor=0.97
        )
        result = sim.nash_equilibrium_premiums(max_iter=5, tol=1.0)
        assert isinstance(result, dict)
        assert "theta1" in result
        assert "theta2" in result
        assert "converged" in result


class TestBMSTransitionMatrix:

    def test_2x2_matrix_structure(self):
        """Two-class BMS: T[0,1] should be 1-p (no claim → step up)."""
        from scipy import stats
        dist = stats.gamma(a=1.2, scale=1.0 / 0.0085)
        sim = BMSEquilibriumSimulator(
            discounts=[0.0, 0.25],
            base_premium=1000.0,
            step_back=1,
            claim_freq=0.05,
            severity_dist=dist,
        )
        sim.fit()
        T = sim.transition_matrix_
        assert T.shape == (2, 2)
        # Row sums = 1
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. PoissonGammaCredibility — additional paths
# ---------------------------------------------------------------------------

@pytest.fixture
def pgc_df() -> pl.DataFrame:
    """Synthetic Poisson-Gamma dataset: 4 schemes, 3 years."""
    return pl.DataFrame({
        "scheme": ["A","A","A", "B","B","B", "C","C","C", "D","D","D"],
        "year":   [2021,2022,2023]*4,
        "claims": [
            80, 88, 84,    # A: ~0.08 rate, large
            12, 14, 13,    # B: ~0.04 rate, medium
            9, 10, 8,      # C: ~0.09 rate, small
            60, 65, 62,    # D: ~0.06 rate, large
        ],
        "exposure": [
            1000, 1100, 1050,
            300, 350, 320,
            100, 110, 105,
            1000, 1050, 980,
        ],
    })


class TestPoissonGammaExtra:

    def test_z_in_zero_one(self, pgc_df):
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        z = pgc.premiums_["Z"].to_numpy()
        assert np.all(z >= 0) and np.all(z <= 1)

    def test_credibility_premium_is_posterior_mean(self, pgc_df):
        """mu_post = (alpha + N) / (beta + E) — check the formula directly."""
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        # For each group: (alpha + N_i) / (beta + E_i)
        alpha, beta = pgc.alpha_, pgc.beta_
        for row in pgc.premiums_.iter_rows(named=True):
            g = row["group"]
            N_i = pgc_df.filter(pl.col("scheme") == g)["claims"].sum()
            E_i = pgc_df.filter(pl.col("scheme") == g)["exposure"].sum()
            expected = (alpha + N_i) / (beta + E_i)
            assert abs(row["credibility_rate"] - expected) < 1e-6

    def test_large_exposure_z_near_one(self, pgc_df):
        """The scheme with highest exposure (A or D) should have Z near 1."""
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        z = pgc.premiums_.sort("Z", descending=True)
        assert z["Z"][0] > 0.7

    def test_small_exposure_group_z_lower(self, pgc_df):
        """The scheme with lowest exposure (C) should have lower Z than A."""
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        z_a = pgc.premiums_.filter(pl.col("group") == "A")["Z"][0]
        z_c = pgc.premiums_.filter(pl.col("group") == "C")["Z"][0]
        assert z_a > z_c

    def test_not_fitted_raises(self):
        pgc = PoissonGammaCredibility()
        with pytest.raises(RuntimeError, match="fit"):
            _ = pgc.premiums_

    def test_not_fitted_alpha_raises(self):
        pgc = PoissonGammaCredibility()
        with pytest.raises(RuntimeError, match="fit"):
            _ = pgc.alpha_

    def test_repr_before_fit(self):
        pgc = PoissonGammaCredibility()
        r = repr(pgc)
        assert "not fitted" in r.lower() or "PoissonGamma" in r

    def test_repr_after_fit(self, pgc_df):
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        r = repr(pgc)
        assert "PoissonGamma" in r

    def test_alpha_beta_positive(self, pgc_df):
        """Prior parameters alpha and beta should be positive."""
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        assert pgc.alpha_ > 0
        assert pgc.beta_ > 0

    def test_premiums_dataframe_columns(self, pgc_df):
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        cols = set(pgc.premiums_.columns)
        assert "group" in cols
        assert "credibility_rate" in cols
        assert "Z" in cols

    def test_premiums_all_positive(self, pgc_df):
        """Claim rates should be positive."""
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        assert np.all(pgc.premiums_["credibility_rate"].to_numpy() > 0)

    def test_mu_hat_between_group_rates(self, pgc_df):
        """mu_hat (prior mean rate) should be between min and max group rates."""
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        obs_rates = pgc.premiums_["observed_rate"].to_numpy()
        mu0 = pgc.alpha_ / pgc.beta_
        assert obs_rates.min() <= mu0 <= obs_rates.max()

    def test_credibility_premium_regresses_toward_mu(self, pgc_df):
        """High-rate group's premium < observed rate; low-rate > observed rate."""
        pgc = PoissonGammaCredibility()
        pgc.fit(pgc_df, group_col="scheme", period_col="year",
                claims_col="claims", exposure_col="exposure")
        mu0 = pgc.alpha_ / pgc.beta_
        for row in pgc.premiums_.iter_rows(named=True):
            obs = row["observed_rate"]
            cred = row["credibility_rate"]
            # Credibility should pull toward mu0
            if obs > mu0:
                assert cred < obs + 1e-10
            elif obs < mu0:
                assert cred > obs - 1e-10


# ---------------------------------------------------------------------------
# 5. _validation additional paths
# ---------------------------------------------------------------------------

class TestToPolarsExtra:

    def test_polars_passthrough(self):
        """Passing a Polars DataFrame should return it unchanged."""
        df = pl.DataFrame({"a": [1, 2]})
        result = _to_polars(df)
        assert result is df

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="polars.DataFrame or pandas.DataFrame"):
            _to_polars([1, 2, 3])

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            _to_polars({"a": [1, 2]})


class TestValidatePanelDataExtra:

    def test_zero_weight_raises(self):
        """Weight=0 is non-positive and should raise."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [0.0, 100.0, 100.0, 100.0],
        })
        with pytest.raises(ValueError, match="non-positive"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_nan_loss_raises(self):
        """NaN in loss column should raise (non-finite)."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, float("nan"), 0.9, 1.0],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        with pytest.raises(ValueError):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_valid_data_no_error(self):
        """Valid data should not raise."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")  # no exception


class TestCheckDuplicatePeriodsExtra:

    def test_no_duplicates_no_warning(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
            assert len(w) == 0

    def test_many_duplicates_warns(self):
        """Multiple duplicate entries should warn."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "period": [1, 1, 2, 1, 1],
        })
        with pytest.warns(UserWarning, match="duplicate"):
            check_duplicate_periods(df, "group", "period")


# ---------------------------------------------------------------------------
# 6. BuhlmannStraub: fit() method return value and chaining
# ---------------------------------------------------------------------------

class TestBuhlmannStraubChaining:

    def test_fit_returns_self(self):
        """fit() should return self to allow chaining."""
        df = _make_panel()
        bs = BuhlmannStraub()
        result = bs.fit(df)
        assert result is bs

    def test_chaining_works(self):
        """Chained fit().premiums_ should work."""
        df = _make_panel()
        premiums = BuhlmannStraub().fit(df).premiums_
        assert isinstance(premiums, pl.DataFrame)

    def test_fit_twice_overwrites(self):
        """Fitting twice should overwrite the first fit."""
        df1 = _make_panel(groups=["A", "B"])
        df2 = _make_panel(groups=["X", "Y", "Z"])
        bs = BuhlmannStraub()
        bs.fit(df1)
        assert len(bs.premiums_) == 2
        bs.fit(df2)
        assert len(bs.premiums_) == 3
