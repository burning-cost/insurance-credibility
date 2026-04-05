"""
Direct tests for insurance_credibility.classical._validation.

These functions are called by every classical estimator but had no direct
tests. Coverage here means validation failures are caught at the source
rather than surfacing as opaque errors inside BuhlmannStraub.fit().
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_credibility.classical._validation import (
    _to_polars,
    validate_panel_data,
    check_duplicate_periods,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_valid_df() -> pl.DataFrame:
    """Two groups, three periods each — the smallest valid panel."""
    return pl.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B"],
        "period": [1, 2, 3, 1, 2, 3],
        "loss": [1.0, 1.1, 0.9, 1.5, 1.4, 1.6],
        "weight": [100.0, 110.0, 90.0, 50.0, 60.0, 55.0],
    })


# ---------------------------------------------------------------------------
# _to_polars
# ---------------------------------------------------------------------------

class TestToPolars:

    def test_polars_passthrough(self):
        """A Polars DataFrame should be returned unchanged (zero-copy path)."""
        df = _minimal_valid_df()
        result = _to_polars(df)
        assert result is df

    def test_pandas_conversion(self):
        """A pandas DataFrame should be converted to Polars correctly."""
        pd = pytest.importorskip("pandas")
        df_pd = pd.DataFrame({
            "group": ["A", "B"],
            "period": [1, 1],
            "loss": [1.0, 2.0],
            "weight": [100.0, 200.0],
        })
        result = _to_polars(df_pd)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 4)

    def test_invalid_type_raises_type_error(self):
        """Passing a plain dict should raise TypeError with a helpful message."""
        with pytest.raises(TypeError, match="polars.DataFrame or pandas.DataFrame"):
            _to_polars({"group": [1, 2]})

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="polars.DataFrame or pandas.DataFrame"):
            _to_polars([[1, 2], [3, 4]])


# ---------------------------------------------------------------------------
# validate_panel_data — valid inputs
# ---------------------------------------------------------------------------

class TestValidatePanelDataValid:

    def test_minimal_valid_df_passes(self):
        """Minimum viable panel should not raise."""
        df = _minimal_valid_df()
        validate_panel_data(df, "group", "period", "loss", "weight")  # should not raise

    def test_single_row_group_with_multi_row_groups_warns(self):
        """One single-period group among multi-period groups should warn but not raise."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B", "C"],
            "period": [1, 2, 3, 1, 2, 3, 1],
            "loss": [1.0, 1.1, 0.9, 1.5, 1.4, 1.6, 1.2],
            "weight": [100.0, 110.0, 90.0, 50.0, 60.0, 55.0, 80.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_panel_data(df, "group", "period", "loss", "weight")
        assert any("one period" in str(warning.message) for warning in w)

    def test_many_single_period_groups_warns_with_truncated_list(self):
        """When >5 groups have one period, the warning should show '...' truncation."""
        rows = {"group": [], "period": [], "loss": [], "weight": []}
        # 8 single-period groups + 2 multi-period groups
        for g in list("ABCDEFGH"):
            rows["group"].append(g)
            rows["period"].append(1)
            rows["loss"].append(1.0)
            rows["weight"].append(100.0)
        for p in [1, 2, 3]:
            rows["group"].append("X")
            rows["period"].append(p)
            rows["loss"].append(1.0)
            rows["weight"].append(100.0)
        for p in [1, 2, 3]:
            rows["group"].append("Y")
            rows["period"].append(p)
            rows["loss"].append(1.0)
            rows["weight"].append(100.0)
        df = pl.DataFrame(rows)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_panel_data(df, "group", "period", "loss", "weight")
        assert any("..." in str(warning.message) for warning in w)

    def test_integer_group_ids_pass(self):
        """Integer group IDs are valid."""
        df = pl.DataFrame({
            "group": [1, 1, 2, 2],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_string_period_column_passes(self):
        """Period column can be strings (e.g. '2020Q1')."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": ["2023Q1", "2023Q2", "2023Q1", "2023Q2"],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_large_weights_pass(self):
        """Very large exposure weights (e.g. policy-years in hundreds of thousands) are fine."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [1e6, 2e6, 5e5, 1.5e6],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data — missing columns
# ---------------------------------------------------------------------------

class TestValidatePanelDataMissingColumns:

    def test_missing_weight_column_raises(self):
        df = pl.DataFrame({"group": ["A"], "period": [1], "loss": [1.0]})
        with pytest.raises(ValueError, match="Columns not found"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_missing_loss_column_raises(self):
        df = pl.DataFrame({"group": ["A"], "period": [1], "weight": [100.0]})
        with pytest.raises(ValueError, match="Columns not found"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_missing_group_column_raises(self):
        df = pl.DataFrame({"period": [1], "loss": [1.0], "weight": [100.0]})
        with pytest.raises(ValueError, match="Columns not found"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_missing_period_column_raises(self):
        df = pl.DataFrame({"group": ["A"], "loss": [1.0], "weight": [100.0]})
        with pytest.raises(ValueError, match="Columns not found"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_all_columns_missing_raises(self):
        df = pl.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="Columns not found"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_error_message_includes_missing_column_name(self):
        """The error should tell the user which column is missing."""
        df = pl.DataFrame({"group": ["A"], "period": [1], "loss": [1.0]})
        with pytest.raises(ValueError, match="weight"):
            validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data — empty DataFrame
# ---------------------------------------------------------------------------

class TestValidatePanelDataEmpty:

    def test_empty_dataframe_raises(self):
        df = pl.DataFrame({
            "group": pl.Series([], dtype=pl.Utf8),
            "period": pl.Series([], dtype=pl.Int64),
            "loss": pl.Series([], dtype=pl.Float64),
            "weight": pl.Series([], dtype=pl.Float64),
        })
        with pytest.raises(ValueError, match="empty"):
            validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data — null values
# ---------------------------------------------------------------------------

class TestValidatePanelDataNulls:

    def test_null_in_loss_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, None, 0.9, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="null"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_null_in_weight_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, None, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="null"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_null_in_group_raises(self):
        df = pl.DataFrame({
            "group": ["A", None, "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="null"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_error_message_includes_column_name_and_count(self):
        """Error message should say which column has nulls and how many."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, None, None, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="loss") as exc_info:
            validate_panel_data(df, "group", "period", "loss", "weight")
        assert "2" in str(exc_info.value)  # two nulls reported


# ---------------------------------------------------------------------------
# validate_panel_data — non-positive weights
# ---------------------------------------------------------------------------

class TestValidatePanelDataWeights:

    def test_zero_weight_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, 0.0, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="non-positive"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_negative_weight_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, -10.0, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="non-positive"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_very_small_positive_weight_passes(self):
        """Epsilon-positive weights are valid (unusual but not illegal)."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [1e-10, 100.0, 90.0, 100.0],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data — non-finite loss values
# ---------------------------------------------------------------------------

class TestValidatePanelDataNonFiniteLoss:

    def test_inf_loss_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, float("inf"), 0.9, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="non-finite"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_neg_inf_loss_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, float("-inf"), 0.9, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="non-finite"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_nan_loss_raises(self):
        """NaN in loss is caught by the non-finite check (np.isfinite is False for NaN)."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, float("nan"), 0.9, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        with pytest.raises(ValueError, match="non-finite"):
            validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data — group count constraints
# ---------------------------------------------------------------------------

class TestValidatePanelDataGroupConstraints:

    def test_single_group_raises(self):
        """Cannot estimate between-group variance with only one group."""
        df = pl.DataFrame({
            "group": ["A", "A", "A"],
            "period": [1, 2, 3],
            "loss": [1.0, 1.1, 0.9],
            "weight": [100.0, 110.0, 90.0],
        })
        with pytest.raises(ValueError, match="2 group"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_all_single_period_groups_raises(self):
        """Cannot estimate within-group variance if every group has one period."""
        df = pl.DataFrame({
            "group": ["A", "B", "C"],
            "period": [1, 1, 1],
            "loss": [1.0, 1.2, 0.8],
            "weight": [100.0, 200.0, 150.0],
        })
        with pytest.raises(ValueError, match="one period"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_exactly_two_groups_two_periods_passes(self):
        """The absolute minimum: 2 groups, 2 periods each."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, 110.0, 90.0, 100.0],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_two_groups_one_has_single_period_warns_not_raises(self):
        """
        One group with a single period warns but does not prevent fitting.
        We need at least one group with >=2 periods for v_hat.
        """
        df = pl.DataFrame({
            "group": ["A", "A", "B"],
            "period": [1, 2, 1],
            "loss": [1.0, 1.1, 0.9],
            "weight": [100.0, 110.0, 90.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_panel_data(df, "group", "period", "loss", "weight")
        assert any("one period" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# check_duplicate_periods
# ---------------------------------------------------------------------------

class TestCheckDuplicatePeriods:

    def test_no_duplicates_no_warning(self):
        """Clean panel data should produce no warnings."""
        df = _minimal_valid_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
        dup_warns = [x for x in w if "duplicate" in str(x.message).lower()]
        assert len(dup_warns) == 0

    def test_single_duplicate_warns(self):
        """One duplicate (group, period) pair should trigger a warning."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "period": [1, 1, 2, 1, 2],  # A/period=1 appears twice
            "loss": [1.0, 1.0, 1.1, 0.9, 1.0],
            "weight": [50.0, 50.0, 110.0, 90.0, 100.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
        assert any("duplicate" in str(warning.message).lower() for warning in w)

    def test_multiple_duplicates_warns_with_count(self):
        """Warning message should report the number of duplicate rows."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "A", "B", "B"],
            "period": [1, 1, 2, 2, 1, 2],  # 2 duplicates
            "loss": [1.0, 1.0, 1.1, 1.1, 0.9, 1.0],
            "weight": [50.0, 50.0, 55.0, 55.0, 90.0, 100.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
        warning_messages = [str(x.message) for x in w if "duplicate" in str(x.message).lower()]
        assert len(warning_messages) > 0
        assert "2" in warning_messages[0]

    def test_single_row_df_no_warning(self):
        """A single-row DataFrame has no duplicates by definition."""
        df = pl.DataFrame({
            "group": ["A"],
            "period": [1],
            "loss": [1.0],
            "weight": [100.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
        assert not any("duplicate" in str(x.message).lower() for x in w)

    def test_all_same_group_and_period_warns(self):
        """Three rows all with the same (group, period) — two are duplicates."""
        df = pl.DataFrame({
            "group": ["A", "A", "A"],
            "period": [1, 1, 1],
            "loss": [1.0, 1.0, 1.0],
            "weight": [100.0, 100.0, 100.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
        assert any("duplicate" in str(warning.message).lower() for warning in w)

    def test_different_groups_same_period_is_not_duplicate(self):
        """Same period number in different groups is fine — not a duplicate."""
        df = pl.DataFrame({
            "group": ["A", "B", "C"],
            "period": [1, 1, 1],
            "loss": [1.0, 1.2, 0.8],
            "weight": [100.0, 200.0, 150.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
        assert not any("duplicate" in str(x.message).lower() for x in w)
