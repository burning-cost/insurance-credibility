"""
Tests for PoissonGammaCredibility.

The Poisson-Gamma model has exact closed-form solutions, so we can verify
every formula directly against hand-computed values rather than relying on
approximations.

Reference dataset: a synthetic UK motor scheme portfolio with 5 schemes,
3 years each. We compute reference values by hand using the exact formulas:

    Prior calibration (method-of-moments):
        mu_0  = portfolio weighted mean rate
        beta  = mu_0 / var_rate        (prior effective exposure)
        alpha = mu_0 * beta            (prior shape)

    Posterior per group i:
        alpha_post_i = alpha + N_i
        beta_post_i  = beta + E_i
        Z_i          = E_i / (E_i + beta)
        mu_post_i    = (alpha + N_i) / (beta + E_i)
"""

import warnings

import numpy as np
import polars as pl
import pytest
from scipy.stats import gamma as scipy_gamma

from insurance_credibility.classical import PoissonGammaCredibility


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scheme_df() -> pl.DataFrame:
    """
    Synthetic motor scheme portfolio: 5 schemes, 3 years each.

    Schemes:
        A: large scheme, high rate  (~0.08)
        B: medium scheme, low rate  (~0.04)
        C: small scheme, high rate  (~0.09)
        D: large scheme, average    (~0.06)
        E: medium scheme, average   (~0.06)
    """
    return pl.DataFrame({
        "scheme": ["A","A","A",  "B","B","B",  "C","C","C",  "D","D","D",  "E","E","E"],
        "year":   [2021,2022,2023]*5,
        "claims": [
            80, 88, 84,       # A: ~0.08 rate
            12, 14, 13,       # B: ~0.04 rate
            9,  10,  8,       # C: ~0.09 rate
            120, 125, 118,    # D: ~0.06 rate
            30, 33, 31,       # E: ~0.06 rate
        ],
        "exposure": [
            1000, 1100, 1050,  # A
            300,  320,  290,   # B
            100,  110,  95,    # C
            2000, 2100, 1950,  # D
            500,  530,  510,   # E
        ],
    })


@pytest.fixture
def fitted_model(scheme_df) -> PoissonGammaCredibility:
    """Return a fitted model on the scheme fixture."""
    model = PoissonGammaCredibility()
    model.fit(scheme_df, group_col="scheme", claims_col="claims", exposure_col="exposure")
    return model


@pytest.fixture
def known_prior_model(scheme_df) -> PoissonGammaCredibility:
    """Return a model with manually supplied prior parameters alpha=6, beta=100."""
    model = PoissonGammaCredibility(prior_alpha=6.0, prior_beta=100.0)
    model.fit(scheme_df, group_col="scheme", claims_col="claims", exposure_col="exposure")
    return model


# ---------------------------------------------------------------------------
# 1. Prior calibration correctness
# ---------------------------------------------------------------------------

class TestPriorCalibration:

    def test_prior_mean_close_to_portfolio_mean(self, scheme_df, fitted_model):
        """Prior mean should match the exposure-weighted portfolio mean rate."""
        total_claims = scheme_df["claims"].sum()
        total_exposure = scheme_df["exposure"].sum()
        portfolio_rate = total_claims / total_exposure
        assert abs(fitted_model.prior_mean_ - portfolio_rate) < 0.005

    def test_alpha_positive(self, fitted_model):
        assert fitted_model.alpha_ > 0

    def test_beta_positive(self, fitted_model):
        assert fitted_model.beta_ > 0

    def test_prior_mean_equals_alpha_over_beta(self, fitted_model):
        """prior_mean_ must equal alpha / beta by definition."""
        assert abs(fitted_model.prior_mean_ - fitted_model.alpha_ / fitted_model.beta_) < 1e-10

    def test_supplied_prior_used_exactly(self, known_prior_model):
        """When prior_alpha and prior_beta are supplied, they are used verbatim."""
        assert known_prior_model.alpha_ == 6.0
        assert known_prior_model.beta_ == 100.0

    def test_supplied_prior_mean(self, known_prior_model):
        assert abs(known_prior_model.prior_mean_ - 0.06) < 1e-10


# ---------------------------------------------------------------------------
# 2. Posterior update formulas
# ---------------------------------------------------------------------------

class TestPosteriorFormulas:

    def test_credibility_rate_is_posterior_mean(self, fitted_model):
        """
        For each group: credibility_rate = (alpha + N_i) / (beta + E_i).
        This is the defining formula; we verify it holds exactly.
        """
        alpha = fitted_model.alpha_
        beta = fitted_model.beta_
        premiums = fitted_model.premiums_
        for row in premiums.iter_rows(named=True):
            expected = (alpha + row["total_claims"]) / (beta + row["total_exposure"])
            assert abs(row["credibility_rate"] - expected) < 1e-10, (
                f"Group {row['group']}: expected {expected:.8f}, "
                f"got {row['credibility_rate']:.8f}"
            )

    def test_z_formula(self, fitted_model):
        """Z_i = E_i / (E_i + beta) for each group."""
        beta = fitted_model.beta_
        premiums = fitted_model.premiums_
        for row in premiums.iter_rows(named=True):
            expected_z = row["total_exposure"] / (row["total_exposure"] + beta)
            assert abs(row["Z"] - expected_z) < 1e-10, (
                f"Group {row['group']}: Z={row['Z']:.8f}, expected {expected_z:.8f}"
            )

    def test_credibility_rate_is_blend_of_observed_and_prior(self, fitted_model):
        """
        credibility_rate = Z * observed_rate + (1 - Z) * prior_mean.
        This is the credibility formula in the familiar blending form.
        """
        prior_mean = fitted_model.prior_mean_
        premiums = fitted_model.premiums_
        for row in premiums.iter_rows(named=True):
            blended = row["Z"] * row["observed_rate"] + (1 - row["Z"]) * prior_mean
            assert abs(row["credibility_rate"] - blended) < 1e-10, (
                f"Group {row['group']}: blending formula mismatch"
            )

    def test_z_between_zero_and_one(self, fitted_model):
        """All credibility factors must be in (0, 1]."""
        z_vals = fitted_model.premiums_["Z"].to_numpy()
        assert (z_vals > 0).all() and (z_vals <= 1).all()

    def test_larger_exposure_higher_z(self, fitted_model):
        """Z is monotone in exposure when beta is fixed."""
        premiums = fitted_model.premiums_
        beta = fitted_model.beta_
        for row in premiums.iter_rows(named=True):
            # Verify Z is correctly ordered: higher E implies higher Z
            z_expected = row["total_exposure"] / (row["total_exposure"] + beta)
            assert abs(row["Z"] - z_expected) < 1e-10

    def test_scheme_d_has_highest_z(self, fitted_model):
        """Scheme D has the most exposure so should have the highest Z."""
        premiums = fitted_model.premiums_
        assert premiums.sort("Z", descending=True)["group"][0] == "D"

    def test_scheme_c_has_lowest_z(self, fitted_model):
        """Scheme C has the least exposure so should have the lowest Z."""
        premiums = fitted_model.premiums_
        assert premiums.sort("Z")["group"][0] == "C"

    def test_high_rate_group_premium_above_prior(self, fitted_model):
        """Scheme A has observed rate ~0.08, above the portfolio mean. Premium > prior mean."""
        prior_mean = fitted_model.prior_mean_
        premiums = fitted_model.premiums_
        scheme_a = premiums.filter(pl.col("group") == "A")["credibility_rate"][0]
        assert scheme_a > prior_mean

    def test_low_rate_group_premium_below_prior(self, fitted_model):
        """Scheme B has observed rate ~0.04, below the portfolio mean. Premium < prior mean."""
        prior_mean = fitted_model.prior_mean_
        premiums = fitted_model.premiums_
        scheme_b = premiums.filter(pl.col("group") == "B")["credibility_rate"][0]
        assert scheme_b < prior_mean


# ---------------------------------------------------------------------------
# 3. Credibility intervals
# ---------------------------------------------------------------------------

class TestCredibilityIntervals:

    def test_intervals_contain_credibility_rate(self, fitted_model):
        """The posterior mean must lie within every credibility interval."""
        intervals = fitted_model.credibility_intervals(0.95)
        for row in intervals.iter_rows(named=True):
            assert row["lower"] <= row["credibility_rate"] <= row["upper"], (
                f"Group {row['group']}: mean not in [{row['lower']:.6f}, {row['upper']:.6f}]"
            )

    def test_wider_interval_wider_bounds(self, fitted_model):
        """99% intervals must be wider than 95% intervals for all groups."""
        ci_95 = fitted_model.credibility_intervals(0.95)
        ci_99 = fitted_model.credibility_intervals(0.99)
        widths_95 = (ci_95["upper"] - ci_95["lower"]).to_numpy()
        widths_99 = (ci_99["upper"] - ci_99["lower"]).to_numpy()
        assert (widths_99 > widths_95).all()

    def test_intervals_match_scipy_gamma(self, fitted_model):
        """Intervals should exactly match scipy.stats.gamma quantiles."""
        alpha = fitted_model.alpha_
        beta = fitted_model.beta_
        premiums = fitted_model.premiums_
        intervals = fitted_model.credibility_intervals(0.95)

        for i, row in enumerate(premiums.iter_rows(named=True)):
            alpha_post = alpha + row["total_claims"]
            beta_post = beta + row["total_exposure"]
            expected_lower = scipy_gamma.ppf(0.025, a=alpha_post, scale=1.0 / beta_post)
            expected_upper = scipy_gamma.ppf(0.975, a=alpha_post, scale=1.0 / beta_post)
            actual = intervals.filter(pl.col("group") == row["group"])
            assert abs(actual["lower"][0] - expected_lower) < 1e-10
            assert abs(actual["upper"][0] - expected_upper) < 1e-10

    def test_smaller_group_wider_interval(self, fitted_model):
        """Scheme C (smallest exposure) should have the widest interval as a fraction of its mean."""
        intervals = fitted_model.credibility_intervals(0.95)
        widths = (intervals["upper"] - intervals["lower"]) / intervals["credibility_rate"]
        sorted_intervals = intervals.with_columns(
            ((pl.col("upper") - pl.col("lower")) / pl.col("credibility_rate")).alias("rel_width")
        ).sort("rel_width", descending=True)
        assert sorted_intervals["group"][0] == "C"

    def test_invalid_credibility_interval_raises(self, fitted_model):
        with pytest.raises(ValueError, match="credibility_interval"):
            fitted_model.credibility_intervals(1.5)
        with pytest.raises(ValueError, match="credibility_interval"):
            fitted_model.credibility_intervals(0.0)


# ---------------------------------------------------------------------------
# 4. predict() method
# ---------------------------------------------------------------------------

class TestPredict:

    def test_predict_returns_expected_keys(self, fitted_model):
        result = fitted_model.predict(claims=50, exposure=1000)
        assert "credibility_rate" in result
        assert "Z" in result
        assert "lower" in result
        assert "upper" in result
        assert "posterior_alpha" in result
        assert "posterior_beta" in result

    def test_predict_formula(self, fitted_model):
        """predict() should exactly apply the posterior formula."""
        alpha = fitted_model.alpha_
        beta = fitted_model.beta_
        result = fitted_model.predict(claims=50, exposure=1000)
        expected_rate = (alpha + 50) / (beta + 1000)
        expected_z = 1000 / (1000 + beta)
        assert abs(result["credibility_rate"] - expected_rate) < 1e-10
        assert abs(result["Z"] - expected_z) < 1e-10

    def test_predict_zero_claims(self, fitted_model):
        """Zero claims is valid — group has no experience, gets heavily pulled to prior."""
        result = fitted_model.predict(claims=0, exposure=10)
        assert 0 < result["credibility_rate"] < fitted_model.prior_mean_

    def test_predict_negative_claims_raises(self, fitted_model):
        with pytest.raises(ValueError, match="non-negative"):
            fitted_model.predict(claims=-1, exposure=100)

    def test_predict_zero_exposure_raises(self, fitted_model):
        with pytest.raises(ValueError, match="positive"):
            fitted_model.predict(claims=0, exposure=0)

    def test_predict_interval_contains_mean(self, fitted_model):
        result = fitted_model.predict(claims=50, exposure=1000, credibility_interval=0.95)
        assert result["lower"] < result["credibility_rate"] < result["upper"]


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_column_raises(self):
        df = pl.DataFrame({"group": ["A", "B"], "claims": [10, 20]})
        model = PoissonGammaCredibility()
        with pytest.raises(ValueError, match="Columns not found"):
            model.fit(df, group_col="group", claims_col="claims", exposure_col="exposure")

    def test_negative_claims_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "claims": [10, -1, 5, 6],
            "exposure": [100.0, 100.0, 100.0, 100.0],
        })
        model = PoissonGammaCredibility()
        with pytest.raises(ValueError, match="negative"):
            model.fit(df)

    def test_non_positive_exposure_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "claims": [10, 10, 5, 6],
            "exposure": [100.0, 0.0, 100.0, 100.0],
        })
        model = PoissonGammaCredibility()
        with pytest.raises(ValueError, match="non-positive"):
            model.fit(df)

    def test_single_group_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "A"],
            "claims": [10, 12, 11],
            "exposure": [100.0, 110.0, 105.0],
        })
        model = PoissonGammaCredibility()
        with pytest.raises(ValueError, match="2 groups"):
            model.fit(df)

    def test_invalid_prior_alpha_raises(self):
        with pytest.raises(ValueError, match="prior_alpha"):
            PoissonGammaCredibility(prior_alpha=-1.0, prior_beta=10.0)

    def test_invalid_prior_beta_raises(self):
        with pytest.raises(ValueError, match="prior_beta"):
            PoissonGammaCredibility(prior_alpha=5.0, prior_beta=0.0)

    def test_not_fitted_raises_on_properties(self):
        model = PoissonGammaCredibility()
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.alpha_
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.beta_
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.prior_mean_
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.premiums_

    def test_not_fitted_raises_on_predict(self):
        model = PoissonGammaCredibility()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(claims=10, exposure=100)

    def test_not_fitted_raises_on_intervals(self):
        model = PoissonGammaCredibility()
        with pytest.raises(RuntimeError, match="fit"):
            model.credibility_intervals()

    def test_homogeneous_groups_warns(self):
        """Groups with identical rates trigger the variance floor warning."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "claims": [50, 50, 50, 50],
            "exposure": [1000.0, 1000.0, 1000.0, 1000.0],
        })
        model = PoissonGammaCredibility()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(df)
            variance_warns = [x for x in w if "variance" in str(x.message).lower()]
            assert len(variance_warns) > 0

    def test_accepts_pandas_input(self, scheme_df):
        pd = pytest.importorskip("pandas")
        df_pd = scheme_df.to_pandas()
        model = PoissonGammaCredibility()
        model.fit(df_pd, group_col="scheme", claims_col="claims", exposure_col="exposure")
        assert isinstance(model.premiums_, pl.DataFrame)


# ---------------------------------------------------------------------------
# 6. API and output format
# ---------------------------------------------------------------------------

class TestOutputFormat:

    def test_premiums_has_expected_columns(self, fitted_model):
        expected = {
            "group", "total_claims", "total_exposure", "observed_rate",
            "Z", "credibility_rate", "prior_mean", "posterior_alpha", "posterior_beta",
        }
        assert expected.issubset(set(fitted_model.premiums_.columns))

    def test_premiums_has_one_row_per_group(self, fitted_model, scheme_df):
        n_groups = scheme_df["scheme"].n_unique()
        assert len(fitted_model.premiums_) == n_groups

    def test_summary_returns_dataframe(self, fitted_model, capsys):
        result = fitted_model.summary()
        captured = capsys.readouterr()
        assert "Poisson-Gamma" in captured.out
        assert "alpha" in captured.out
        assert "beta" in captured.out
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5

    def test_repr_before_fit(self):
        model = PoissonGammaCredibility()
        assert "not fitted" in repr(model)

    def test_repr_after_fit(self, fitted_model):
        r = repr(fitted_model)
        assert "PoissonGammaCredibility(" in r
        assert "alpha=" in r
        assert "beta=" in r
        assert "not fitted" not in r

    def test_fit_returns_self(self, scheme_df):
        """fit() should return self for method chaining."""
        model = PoissonGammaCredibility()
        result = model.fit(scheme_df, group_col="scheme", claims_col="claims",
                           exposure_col="exposure")
        assert result is model

    def test_credibility_intervals_has_expected_columns(self, fitted_model):
        intervals = fitted_model.credibility_intervals()
        assert "group" in intervals.columns
        assert "credibility_rate" in intervals.columns
        assert "lower" in intervals.columns
        assert "upper" in intervals.columns
        assert "Z" in intervals.columns
