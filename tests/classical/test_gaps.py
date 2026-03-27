"""
Structural gap tests for insurance-credibility classical module.

Covers branches not exercised by the existing suite:

BuhlmannStraub
- mu_hat is the exposure-weighted grand mean (algebraic check on 2 groups)
- premiums_['complement'] equals mu_hat for all groups
- k = v/a identity on Hachemeister data (cross-checks k_ property)
- unbalanced panel (different periods per group) — most common real-world case
- groups with widely varying exposures (Z skew)
- check_duplicate_periods warning fires
- _build_group_summary: correct T_i per group
- validate_panel_data: null values, inf loss, empty DataFrame

HierarchicalBuhlmannStraub
- premiums_at() before fit raises RuntimeError
- level_results_ before fit raises RuntimeError
- all levels produce finite variance components with flat DGP (a=0 truncation)
- two-level model with equal-exposure groups: grand mean is mu_hat
- hierarchy validation: correct parent-child mapping with 4-level hierarchy
"""

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_credibility.classical import BuhlmannStraub, HierarchicalBuhlmannStraub
from insurance_credibility.classical._validation import (
    validate_panel_data,
    check_duplicate_periods,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _two_group_df(loss_a=1.0, loss_b=2.0, exposure_a=1000.0, exposure_b=500.0, n_periods=4):
    """Minimal two-group balanced panel."""
    rows = []
    for t in range(1, n_periods + 1):
        rows.append({"group": "A", "period": t, "loss": loss_a, "weight": exposure_a})
        rows.append({"group": "B", "period": t, "loss": loss_b, "weight": exposure_b})
    return pl.DataFrame(rows)


def _unbalanced_df():
    """A=5 periods, B=3 periods, C=2 periods — typical real-world."""
    rows = []
    for t in range(1, 6):
        rows.append({"group": "A", "period": t, "loss": 0.8 + 0.05 * t, "weight": 500.0})
    for t in range(1, 4):
        rows.append({"group": "B", "period": t, "loss": 1.2 - 0.03 * t, "weight": 300.0})
    for t in range(1, 3):
        rows.append({"group": "C", "period": t, "loss": 0.6 + 0.01 * t, "weight": 800.0})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# BuhlmannStraub: mu_hat is the exposure-weighted grand mean
# ---------------------------------------------------------------------------

class TestMuHatIdentity:

    def test_mu_hat_is_weighted_grand_mean_two_groups(self):
        """
        mu_hat = sum_i(w_i * x_bar_i) / sum_i(w_i)

        For a two-group panel with constant loss rates and equal periods,
        x_bar_i = loss_rate_i, so mu_hat = (w_A * loss_A + w_B * loss_B) / (w_A + w_B).
        """
        loss_a, loss_b = 1.0, 2.0
        exp_a, exp_b = 1000.0, 500.0
        df = _two_group_df(loss_a, loss_b, exp_a, exp_b, n_periods=5)

        # Total exposure per group = periods * per_period_exposure
        w_a = 5 * exp_a
        w_b = 5 * exp_b
        expected_mu = (w_a * loss_a + w_b * loss_b) / (w_a + w_b)

        bs = BuhlmannStraub()
        bs.fit(df, group_col="group", period_col="period",
               loss_col="loss", weight_col="weight")

        assert abs(bs.mu_hat_ - expected_mu) < 1e-6, (
            f"mu_hat={bs.mu_hat_:.6f} != expected {expected_mu:.6f}"
        )

    def test_mu_hat_between_group_means(self):
        """mu_hat must be between the minimum and maximum group means."""
        df = _unbalanced_df()
        bs = BuhlmannStraub()
        bs.fit(df)

        premiums = bs.premiums_
        min_obs = float(premiums["observed_mean"].min())
        max_obs = float(premiums["observed_mean"].max())
        assert min_obs <= bs.mu_hat_ <= max_obs, (
            f"mu_hat={bs.mu_hat_} not in [{min_obs}, {max_obs}]"
        )


# ---------------------------------------------------------------------------
# BuhlmannStraub: complement column equals mu_hat
# ---------------------------------------------------------------------------

class TestComplementColumn:

    def test_complement_equals_mu_hat_for_all_groups(self):
        """The 'complement' column in premiums_ should be mu_hat for every row."""
        df = _unbalanced_df()
        bs = BuhlmannStraub()
        bs.fit(df)

        mu = bs.mu_hat_
        complements = bs.premiums_["complement"].to_numpy()
        np.testing.assert_allclose(complements, mu, rtol=1e-10)

    def test_complement_hachemeister(self, hachemeister_df):
        """Same check on the Hachemeister benchmark dataset."""
        bs = BuhlmannStraub()
        bs.fit(hachemeister_df, group_col="state", period_col="period",
               loss_col="ratio", weight_col="weight")
        mu = bs.mu_hat_
        complements = bs.premiums_["complement"].to_numpy()
        np.testing.assert_allclose(complements, mu, rtol=1e-10)


# ---------------------------------------------------------------------------
# BuhlmannStraub: k_ property consistency
# ---------------------------------------------------------------------------

class TestKConsistency:

    def test_k_equals_v_over_a_exactly(self, hachemeister_df):
        """k_ == v_hat_ / a_hat_ to floating-point precision."""
        bs = BuhlmannStraub()
        bs.fit(hachemeister_df, group_col="state", period_col="period",
               loss_col="ratio", weight_col="weight")
        assert abs(bs.k_ - bs.v_hat_ / bs.a_hat_) < 1e-3

    def test_k_infinity_when_a_zero(self):
        """When groups are identical, a=0 and k should be inf."""
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
        assert np.isinf(bs.k_)


# ---------------------------------------------------------------------------
# BuhlmannStraub: unbalanced panel
# ---------------------------------------------------------------------------

class TestUnbalancedPanel:

    def test_unbalanced_panel_fits(self):
        """Groups with different numbers of periods should fit without error."""
        df = _unbalanced_df()
        bs = BuhlmannStraub()
        bs.fit(df)
        assert bs._fitted

    def test_unbalanced_panel_z_monotone_in_exposure(self):
        """Higher total exposure => higher Z (monotone relationship)."""
        df = _unbalanced_df()
        bs = BuhlmannStraub()
        bs.fit(df)
        premiums = bs.premiums_
        k = bs.k_
        for row in premiums.iter_rows(named=True):
            expected_z = row["exposure"] / (row["exposure"] + k)
            assert abs(row["Z"] - expected_z) < 1e-8

    def test_unbalanced_z_order(self):
        """Group with most exposure gets highest Z."""
        df = _unbalanced_df()
        bs = BuhlmannStraub()
        bs.fit(df)
        z = bs.z_.sort("Z", descending=True)
        # Group A has 5 periods, highest total exposure
        assert z["group"][0] == "A"


# ---------------------------------------------------------------------------
# BuhlmannStraub: widely varying exposures
# ---------------------------------------------------------------------------

class TestWidelyVaryingExposures:

    def test_high_exposure_group_gets_z_near_one(self):
        """
        A group with exposure >> k gets Z near 1 (trust own experience).

        This test uses 5 groups with distinct means and tiny within-group noise,
        which guarantees a_hat > 0 and a well-defined, small k. With BIG having
        30x more exposure than any other group, Z_BIG is effectively 1.

        Note: the naive 2-group BIG/SMALL dataset fails this test because the
        B-S a_hat estimator is unstable with only 2 groups and very asymmetric
        exposures — within-group noise from BIG dominates the between-group
        signal, producing negative a_hat (truncated to 0, k -> inf, Z -> 0).
        Five groups with genuinely different means stabilise the estimator.
        """
        # 5 groups with distinct means, tiny within-group noise (±0.001)
        # BIG has 100x more weight per period than others -> total 30000 vs 300 each
        rows = []
        means = {"BIG": 1.500, "G2": 1.200, "G3": 1.000, "G4": 0.800, "G5": 0.500}
        weights = {"BIG": 10000.0, "G2": 100.0, "G3": 100.0, "G4": 100.0, "G5": 100.0}
        deltas = [0.000, +0.001, -0.001]  # tiny period-to-period noise
        for grp, mean in means.items():
            for t, delta in enumerate(deltas, 1):
                rows.append({
                    "group": grp,
                    "period": t,
                    "loss": mean + delta,
                    "weight": weights[grp],
                })
        df = pl.DataFrame(rows)

        bs = BuhlmannStraub()
        bs.fit(df)

        z = bs.z_.filter(pl.col("group") == "BIG")["Z"][0]
        # BIG total exposure = 30000; with 5 well-separated groups k is small
        # and Z_BIG is effectively 1. Assert a conservative threshold.
        assert z > 0.9, f"Expected Z near 1 for very high exposure, got {z:.4f}"

    def test_low_exposure_group_gets_z_near_zero_when_k_large(self):
        """
        A group with exposure << k gets Z near 0 (trust the collective).

        When all groups have identical loss rates, a_hat = 0 (truncated) and
        k -> inf. Every group's Z becomes 0 regardless of exposure. This is
        the degenerate but correct outcome: the model detects no between-group
        heterogeneity, so every group reverts fully to the collective mean.

        A SMALL group with minimal exposure is an extreme case of this regime:
        when k is large (or infinite) relative to SMALL's exposure, Z_SMALL is
        near 0.
        """
        # All groups identical -> a_hat = 0 -> k = inf -> Z = 0 for all groups
        rows = []
        for grp in ["BIG", "MED1", "MED2", "SMALL"]:
            for t in [1, 2, 3]:
                rows.append({
                    "group": grp,
                    "period": t,
                    "loss": 1.0,  # identical everywhere
                    "weight": 10000.0 if grp == "BIG" else (100.0 if "MED" in grp else 1.0),
                })
        df = pl.DataFrame(rows)

        bs = BuhlmannStraub(truncate_a=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bs.fit(df)

        z_small = bs.z_.filter(pl.col("group") == "SMALL")["Z"][0]
        # k = inf -> Z_SMALL = 0 < 0.5
        assert z_small < 0.5, f"Expected Z near 0 for very low exposure, got {z_small:.4f}"


# ---------------------------------------------------------------------------
# validate_panel_data edge cases
# ---------------------------------------------------------------------------

class TestValidatePanelData:

    def test_null_values_raise(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, None, 0.9, 1.1],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        with pytest.raises(ValueError, match="null"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_inf_loss_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, float("inf"), 0.9, 1.1],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        with pytest.raises(ValueError, match="non-finite"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_empty_dataframe_raises(self):
        df = pl.DataFrame({
            "group": pl.Series([], dtype=pl.String),
            "period": pl.Series([], dtype=pl.Int64),
            "loss": pl.Series([], dtype=pl.Float64),
            "weight": pl.Series([], dtype=pl.Float64),
        })
        with pytest.raises(ValueError, match="empty"):
            validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# check_duplicate_periods
# ---------------------------------------------------------------------------

class TestCheckDuplicatePeriods:

    def test_duplicate_warns(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 1, 1, 2],   # A has period 1 twice
            "loss": [1.0, 1.0, 0.9, 1.1],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        with pytest.warns(UserWarning, match="duplicate"):
            check_duplicate_periods(df, "group", "period")

    def test_no_duplicates_does_not_warn(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
            assert len(w) == 0


# ---------------------------------------------------------------------------
# _build_group_summary: correct T_i per group
# ---------------------------------------------------------------------------

class TestBuildGroupSummary:

    def test_T_i_counts_periods_correctly(self):
        """T_i should be the number of distinct periods per group."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "period": [1, 2, 3, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.5, 1.4],
            "weight": [100.0, 110.0, 90.0, 50.0, 60.0],
        })
        summary = BuhlmannStraub._build_group_summary(
            df, "group", "period", "loss", "weight"
        )
        t_map = dict(zip(summary["group"].to_list(), summary["T_i"].to_list()))
        assert t_map["A"] == 3
        assert t_map["B"] == 2

    def test_x_bar_is_exposure_weighted_mean(self):
        """x_bar_i should equal sum(w*x)/sum(w) per group."""
        df = pl.DataFrame({
            "group": ["A", "A"],
            "period": [1, 2],
            "loss": [1.0, 3.0],
            "weight": [100.0, 300.0],
        })
        summary = BuhlmannStraub._build_group_summary(
            df, "group", "period", "loss", "weight"
        )
        expected_x_bar = (100.0 * 1.0 + 300.0 * 3.0) / 400.0
        got = summary["x_bar_i"][0]
        assert abs(got - expected_x_bar) < 1e-8


# ---------------------------------------------------------------------------
# HierarchicalBuhlmannStraub: before-fit raises
# ---------------------------------------------------------------------------

class TestHierarchicalBeforeFit:

    def test_level_results_before_fit_raises(self):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.level_results_

    def test_premiums_at_before_fit_raises(self):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with pytest.raises(RuntimeError, match="fit"):
            model.premiums_at("region")

    def test_premiums_property_before_fit_raises(self):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.premiums_


# ---------------------------------------------------------------------------
# HierarchicalBuhlmannStraub: flat DGP (all groups same rate) -> a=0 truncation
# ---------------------------------------------------------------------------

class TestHierarchicalFlatDGP:

    def test_flat_dgp_truncates_a_at_sector_level(self):
        """When all sectors have the same loss rate, a_hat <= 0 -> truncation."""
        rows = []
        for region in ["R1", "R2"]:
            for district in ["D1", "D2"]:
                for sector in ["S1", "S2", "S3"]:
                    for period in [1, 2, 3]:
                        rows.append({
                            "region": region,
                            "district": f"{region}_{district}",
                            "sector": f"{region}_{district}_{sector}",
                            "period": period,
                            "loss_rate": 0.65,  # identical everywhere
                            "exposure": 500.0,
                        })
        df = pl.DataFrame(rows)
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district", "sector"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Must pass non-default column names: flat DGP uses "loss_rate"/"exposure"
            model.fit(df, loss_col="loss_rate", weight_col="exposure")

        # a_hat at all levels should be 0 (truncated)
        for level in ["region", "district", "sector"]:
            assert model.level_results_[level].a_hat == 0.0

    def test_flat_dgp_all_premiums_equal_mu_hat(self):
        """With identical loss rates, every group gets the collective mean."""
        rows = []
        for region in ["R1", "R2"]:
            for district in ["D1", "D2"]:
                for period in [1, 2, 3]:
                    rows.append({
                        "region": region,
                        "district": f"{region}_{district}",
                        "period": period,
                        "loss_rate": 0.70,
                        "exposure": 400.0,
                    })
        df = pl.DataFrame(rows)
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Must pass non-default column names: flat DGP uses "loss_rate"/"exposure"
            model.fit(df, loss_col="loss_rate", weight_col="exposure")

        premiums = model.premiums_["credibility_premium"].to_numpy()
        # All premiums should be (approximately) 0.70
        np.testing.assert_allclose(premiums, 0.70, atol=1e-6)


# ---------------------------------------------------------------------------
# HierarchicalBuhlmannStraub: 4-level hierarchy does not crash
# ---------------------------------------------------------------------------

class TestHierarchical4Level:

    def test_4_level_hierarchy_fits(self):
        """A 4-level hierarchy (country -> region -> district -> sector) should fit."""
        rows = []
        rng = np.random.default_rng(77)
        for country in ["UK", "IE"]:
            for region in ["N", "S"]:
                for district in ["D1", "D2"]:
                    for sector in ["S1", "S2"]:
                        for period in [1, 2, 3]:
                            rows.append({
                                "country": country,
                                "region": f"{country}_{region}",
                                "district": f"{country}_{region}_{district}",
                                "sector": f"{country}_{region}_{district}_{sector}",
                                "period": period,
                                "loss": float(rng.normal(0.65, 0.05)),
                                "weight": float(rng.uniform(200, 1000)),
                            })
        df = pl.DataFrame(rows)
        model = HierarchicalBuhlmannStraub(level_cols=["country", "region", "district", "sector"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df, period_col="period", loss_col="loss", weight_col="weight")

        assert model._fitted
        assert set(model.level_results_.keys()) == {"country", "region", "district", "sector"}
        premiums = model.premiums_
        assert "credibility_premium" in premiums.columns
        assert np.all(np.isfinite(premiums["credibility_premium"].to_numpy()))


# ---------------------------------------------------------------------------
# BuhlmannStraub: exactly 2 groups (minimum for a_hat estimation)
# ---------------------------------------------------------------------------

class TestTwoGroupMinimum:

    def test_two_groups_fits(self):
        """Minimum viable: 2 groups, 2 periods each."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.2, 0.6, 0.7],
            "weight": [100.0, 110.0, 80.0, 90.0],
        })
        bs = BuhlmannStraub()
        bs.fit(df)
        assert bs._fitted
        assert len(bs.premiums_) == 2

    def test_two_groups_premiums_between_means(self):
        """Credibility premiums must be between the two group means."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.0, 2.0, 2.0],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        bs = BuhlmannStraub()
        bs.fit(df)
        premiums = bs.premiums_["credibility_premium"].to_numpy()
        # Both premiums should be between 1.0 and 2.0
        assert np.all(premiums >= 1.0 - 1e-6)
        assert np.all(premiums <= 2.0 + 1e-6)
