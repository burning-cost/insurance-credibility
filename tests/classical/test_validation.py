"""Tests for insurance_credibility.classical._validation.

These tests cover the two public functions exposed by the module:
  - validate_panel_data: raises ValueError for bad input, warns for edge cases
  - check_duplicate_periods: warns when (group, period) pairs repeat

The validation module is the first line of defence before Bühlmann-Straub
fitting. Its contract matters: it should reject unrecoverable input and
produce actionable error messages, while only warning (not raising) for
situations the model can handle automatically.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_credibility.classical._validation import (
    check_duplicate_periods,
    validate_panel_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_valid_panel(n_groups: int = 4, n_periods: int = 3) -> pl.DataFrame:
    """Return a minimal valid panel DataFrame for Bühlmann-Straub fitting."""
    rng = np.random.default_rng(42)
    rows = []
    for g in range(n_groups):
        for p in range(n_periods):
            rows.append({
                "group": f"G{g}",
                "period": p,
                "loss": float(rng.uniform(0.4, 1.2)),
                "weight": float(rng.uniform(50, 500)),
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_panel_data: column presence
# ---------------------------------------------------------------------------

class TestValidatePanelDataColumnChecks:
    def test_valid_panel_passes_without_error(self):
        df = _make_valid_panel()
        # Should not raise or warn
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_missing_group_col_raises(self):
        df = _make_valid_panel().drop("group")
        with pytest.raises(ValueError, match="group"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_missing_period_col_raises(self):
        df = _make_valid_panel().drop("period")
        with pytest.raises(ValueError, match="period"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_missing_loss_col_raises(self):
        df = _make_valid_panel().drop("loss")
        with pytest.raises(ValueError, match="loss"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_missing_weight_col_raises(self):
        df = _make_valid_panel().drop("weight")
        with pytest.raises(ValueError, match="weight"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_all_columns_missing_raises(self):
        df = pl.DataFrame({"unrelated": [1, 2, 3]})
        with pytest.raises(ValueError, match="Columns not found"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_error_message_lists_missing_columns(self):
        df = _make_valid_panel().drop("loss").drop("weight")
        with pytest.raises(ValueError, match="Columns not found"):
            validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data: empty data
# ---------------------------------------------------------------------------

class TestValidatePanelDataEmptyData:
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
# validate_panel_data: null values
# ---------------------------------------------------------------------------

class TestValidatePanelDataNulls:
    def test_null_in_group_raises(self):
        df = _make_valid_panel().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("group"))
            .alias("group")
        )
        with pytest.raises(ValueError, match="null"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_null_in_loss_raises(self):
        df = _make_valid_panel().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("loss"))
            .alias("loss")
        )
        with pytest.raises(ValueError, match="null"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_null_in_weight_raises(self):
        df = _make_valid_panel().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        with pytest.raises(ValueError, match="null"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_null_in_period_raises(self):
        df = _make_valid_panel().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("period"))
            .alias("period")
        )
        with pytest.raises(ValueError, match="null"):
            validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data: weight constraints
# ---------------------------------------------------------------------------

class TestValidatePanelDataWeights:
    def test_zero_weight_raises(self):
        df = _make_valid_panel()
        # Set one weight to zero
        weights = df["weight"].to_list()
        weights[0] = 0.0
        df = df.with_columns(pl.Series("weight", weights))
        with pytest.raises(ValueError, match="non-positive"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_negative_weight_raises(self):
        df = _make_valid_panel()
        weights = df["weight"].to_list()
        weights[2] = -10.0
        df = df.with_columns(pl.Series("weight", weights))
        with pytest.raises(ValueError, match="non-positive"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_very_small_positive_weight_passes(self):
        """Tiny but positive weight is valid — not for us to reject."""
        df = _make_valid_panel()
        weights = df["weight"].to_list()
        weights[0] = 1e-10
        df = df.with_columns(pl.Series("weight", weights))
        # Should not raise
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_large_weights_pass(self):
        """High-exposure segments (e.g. fleet of 50k vehicles) should pass."""
        df = _make_valid_panel()
        df = df.with_columns((pl.col("weight") * 1_000_000).alias("weight"))
        validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data: non-finite loss values
# ---------------------------------------------------------------------------

class TestValidatePanelDataFiniteness:
    def test_inf_loss_raises(self):
        df = _make_valid_panel()
        losses = df["loss"].to_list()
        losses[1] = float("inf")
        df = df.with_columns(pl.Series("loss", losses))
        with pytest.raises(ValueError, match="non-finite"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_negative_inf_loss_raises(self):
        df = _make_valid_panel()
        losses = df["loss"].to_list()
        losses[0] = float("-inf")
        df = df.with_columns(pl.Series("loss", losses))
        with pytest.raises(ValueError, match="non-finite"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_nan_loss_raises(self):
        df = _make_valid_panel()
        losses = df["loss"].to_list()
        losses[3] = float("nan")
        df = df.with_columns(pl.Series("loss", losses))
        with pytest.raises(ValueError, match="non-finite"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_zero_loss_passes(self):
        """Zero loss (a clean year with no claims) is valid."""
        df = _make_valid_panel()
        losses = df["loss"].to_list()
        losses[0] = 0.0
        df = df.with_columns(pl.Series("loss", losses))
        # Should not raise
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_negative_loss_passes(self):
        """Negative loss values (salvage recoveries exceeding paid claims) are
        uncommon but financially valid. Validation should not reject them."""
        df = _make_valid_panel()
        losses = df["loss"].to_list()
        losses[0] = -0.1
        df = df.with_columns(pl.Series("loss", losses))
        # Should not raise — negative loss is a data reality, not an error
        validate_panel_data(df, "group", "period", "loss", "weight")


# ---------------------------------------------------------------------------
# validate_panel_data: single-period edge cases
# ---------------------------------------------------------------------------

class TestValidatePanelDataPeriodRequirements:
    def test_all_single_period_raises(self):
        """All groups have exactly one period: v_hat is unidentifiable."""
        df = pl.DataFrame({
            "group": ["A", "B", "C", "D"],
            "period": [1, 1, 1, 1],
            "loss": [0.6, 0.7, 0.8, 0.5],
            "weight": [100.0, 150.0, 200.0, 120.0],
        })
        with pytest.raises(ValueError, match="one period"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_single_group_raises(self):
        """Only one group: a_hat (between-group variance) is unidentifiable."""
        df = pl.DataFrame({
            "group": ["A", "A", "A"],
            "period": [1, 2, 3],
            "loss": [0.6, 0.7, 0.65],
            "weight": [100.0, 120.0, 110.0],
        })
        with pytest.raises(ValueError, match="2 groups"):
            validate_panel_data(df, "group", "period", "loss", "weight")

    def test_two_groups_each_two_periods_passes(self):
        """Minimum viable dataset: 2 groups, 2 periods each."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [0.6, 0.7, 0.5, 0.55],
            "weight": [100.0, 110.0, 80.0, 90.0],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_some_single_period_groups_warns_not_raises(self):
        """Some single-period groups are acceptable — they get credibility
        premiums but don't contribute to v_hat estimation."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B", "C"],
            "period": [1, 2, 3, 1, 2, 1],
            "loss": [0.6, 0.65, 0.7, 0.55, 0.58, 0.80],
            "weight": [100.0, 110.0, 120.0, 80.0, 85.0, 50.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_panel_data(df, "group", "period", "loss", "weight")
        # Should warn about the single-period group C
        assert any("one period" in str(x.message).lower() for x in w)

    def test_single_period_warning_names_the_group(self):
        """The warning message should identify which groups are affected."""
        df = pl.DataFrame({
            "group": ["A", "A", "B"],
            "period": [1, 2, 1],
            "loss": [0.6, 0.65, 0.55],
            "weight": [100.0, 110.0, 80.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_panel_data(df, "group", "period", "loss", "weight")
        warning_msgs = [str(x.message) for x in w]
        assert any("B" in msg for msg in warning_msgs)

    def test_many_single_period_groups_truncates_warning(self):
        """When many groups are affected, the warning should not print all of
        them verbatim (to avoid flooding logs in large books)."""
        groups_multi = [f"G{i}" for i in range(10)]
        groups_single = [f"S{i}" for i in range(20)]
        rows = (
            [{"group": g, "period": p, "loss": 0.6, "weight": 100.0}
             for g in groups_multi for p in [1, 2]]
            + [{"group": g, "period": 1, "loss": 0.6, "weight": 100.0}
               for g in groups_single]
        )
        df = pl.DataFrame(rows)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_panel_data(df, "group", "period", "loss", "weight")
        assert len(w) >= 1
        # Warning should exist but should not exceed a sane length
        msg = str(w[0].message)
        assert len(msg) < 2000  # not printing all 20 group names verbatim


# ---------------------------------------------------------------------------
# validate_panel_data: realistic insurance scenarios
# ---------------------------------------------------------------------------

class TestValidatePanelDataRealisticInsurance:
    def test_motor_book_with_100_segments_passes(self):
        """A typical UK motor book panel should pass without issue."""
        rng = np.random.default_rng(99)
        n_segments, n_periods = 100, 8
        rows = []
        for s in range(n_segments):
            base_lr = rng.uniform(0.55, 0.85)
            for p in range(n_periods):
                rows.append({
                    "segment_id": f"SEG_{s:03d}",
                    "period": p + 1,
                    "loss_ratio": float(np.clip(rng.normal(base_lr, 0.05), 0.1, 2.0)),
                    "earned_premium": float(rng.uniform(10_000, 200_000)),
                })
        df = pl.DataFrame(rows)
        validate_panel_data(df, "segment_id", "period", "loss_ratio", "earned_premium")

    def test_high_loss_ratio_segment_passes(self):
        """A severely loss-making segment (LR > 200%) should not be rejected
        — validation is about data integrity, not business outcomes."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [2.5, 3.1, 0.6, 0.7],  # A is extremely loss-making
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_unequal_periods_per_group_passes(self):
        """Unbalanced panels (some groups observed longer) are valid inputs."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "period": [1, 2, 3, 2, 3],
            "loss": [0.6, 0.65, 0.7, 0.55, 0.58],
            "weight": [100.0, 110.0, 120.0, 80.0, 85.0],
        })
        validate_panel_data(df, "group", "period", "loss", "weight")

    def test_custom_column_names(self):
        """Validation works regardless of column naming convention."""
        df = pl.DataFrame({
            "segment": ["A", "A", "B", "B"],
            "year": [2021, 2022, 2021, 2022],
            "lr": [0.65, 0.70, 0.55, 0.60],
            "premium": [100_000.0, 110_000.0, 80_000.0, 85_000.0],
        })
        validate_panel_data(df, "segment", "year", "lr", "premium")


# ---------------------------------------------------------------------------
# check_duplicate_periods
# ---------------------------------------------------------------------------

class TestCheckDuplicatePeriods:
    def test_no_duplicates_no_warning(self):
        df = _make_valid_panel()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
        assert len(w) == 0

    def test_one_duplicate_warns(self):
        """A single duplicated (group, period) pair triggers a warning."""
        df = _make_valid_panel()
        duplicate_row = df.head(1)
        df_with_dup = pl.concat([df, duplicate_row])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df_with_dup, "group", "period")
        assert len(w) == 1
        assert "duplicate" in str(w[0].message).lower()

    def test_multiple_duplicates_counted_correctly(self):
        """Warning message should report the correct duplicate count."""
        df = _make_valid_panel(n_groups=2, n_periods=3)
        # Duplicate all rows for group G0
        extra = df.filter(pl.col("group") == "G0")
        df_with_dups = pl.concat([df, extra])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df_with_dups, "group", "period")
        assert len(w) == 1
        # Should mention the count
        msg = str(w[0].message)
        assert "3" in msg  # 3 extra rows = 3 duplicates

    def test_warning_contains_aggregation_advice(self):
        """The warning should suggest aggregating before fitting."""
        df = _make_valid_panel(n_groups=2, n_periods=2)
        dup = df.head(1)
        df_dup = pl.concat([df, dup])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df_dup, "group", "period")
        assert len(w) == 1
        msg = str(w[0].message).lower()
        assert "aggregat" in msg

    def test_all_rows_duplicated_warns(self):
        """Doubling the entire dataset means N duplicates."""
        df = _make_valid_panel(n_groups=2, n_periods=2)
        df_doubled = pl.concat([df, df])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df_doubled, "group", "period")
        assert len(w) == 1

    def test_works_with_custom_column_names(self):
        """Column names don't have to be 'group' and 'period'."""
        df = pl.DataFrame({
            "segment": ["A", "A", "B", "B"],
            "year": [2021, 2022, 2021, 2022],
            "lr": [0.65, 0.70, 0.55, 0.60],
            "premium": [100_000.0, 110_000.0, 80_000.0, 85_000.0],
        })
        # No duplicates — should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "segment", "year")
        assert len(w) == 0

    def test_duplicate_detection_is_per_group_period_pair(self):
        """Same period appearing twice in *different* groups is not a duplicate."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],  # same periods but different groups
            "loss": [0.6, 0.7, 0.5, 0.55],
            "weight": [100.0, 110.0, 80.0, 90.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_duplicate_periods(df, "group", "period")
        assert len(w) == 0
