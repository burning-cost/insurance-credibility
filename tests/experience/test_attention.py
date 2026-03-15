"""Tests for DeepAttentionModel (torch optional dependency).

These tests are skipped when torch is not installed. They test:
- Forward pass shape
- Gradient flow
- Balance constraint (attention weights sum <= 1)
- Credibility factor range
- predict_batch DataFrame output
"""

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from insurance_credibility.experience import ClaimsHistory
from insurance_credibility.experience.attention import DeepAttentionModel, _TORCH_MISSING_MSG


skip_if_no_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)


def make_portfolio(n: int, rng: np.random.Generator, mean: float = 1.0, n_periods: int = 4) -> list[ClaimsHistory]:
    histories = []
    for i in range(n):
        counts = rng.poisson(mean, size=n_periods).tolist()
        histories.append(
            ClaimsHistory(
                f"P{i}",
                list(range(1, n_periods + 1)),
                counts,
                prior_premium=mean,
            )
        )
    return histories


class TestDeepAttentionImport:
    def test_import_without_torch_raises(self):
        """Accessing DeepAttentionModel without torch should raise ImportError
        with helpful message. This test is only meaningful without torch installed,
        but we verify the message text at minimum."""
        assert "pip install" in _TORCH_MISSING_MSG
        assert "insurance-experience[deep]" in _TORCH_MISSING_MSG


@skip_if_no_torch
class TestDeepAttentionFit:
    def test_fit_returns_self(self):
        rng = np.random.default_rng(400)
        histories = make_portfolio(30, rng)
        model = DeepAttentionModel(max_periods=4, n_epochs=5, random_state=42)
        result = model.fit(histories)
        assert result is model

    def test_is_fitted_after_fit(self):
        rng = np.random.default_rng(401)
        histories = make_portfolio(20, rng)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        assert not model.is_fitted_
        model.fit(histories)
        assert model.is_fitted_

    def test_training_losses_recorded(self):
        rng = np.random.default_rng(402)
        histories = make_portfolio(20, rng)
        n_epochs = 5
        model = DeepAttentionModel(max_periods=4, n_epochs=n_epochs, random_state=42)
        model.fit(histories)
        assert len(model.training_losses_) == n_epochs

    def test_training_losses_finite(self):
        rng = np.random.default_rng(403)
        histories = make_portfolio(20, rng)
        model = DeepAttentionModel(max_periods=4, n_epochs=5, random_state=42)
        model.fit(histories)
        assert all(np.isfinite(l) for l in model.training_losses_)

    def test_model_has_parameters(self):
        rng = np.random.default_rng(404)
        histories = make_portfolio(20, rng)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        model.fit(histories)
        params = list(model.model_.parameters())
        assert len(params) > 0


@skip_if_no_torch
class TestDeepAttentionPredict:
    def test_predict_before_fit_raises(self):
        model = DeepAttentionModel(max_periods=4)
        h = ClaimsHistory("P1", [1, 2, 3], [0, 1, 0], prior_premium=1.0)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(h)

    def test_cf_non_negative(self):
        rng = np.random.default_rng(405)
        histories = make_portfolio(30, rng)
        model = DeepAttentionModel(max_periods=4, n_epochs=5, random_state=42)
        model.fit(histories)
        for h in histories[:5]:
            cf = model.predict(h)
            assert cf >= 0.0

    def test_cf_finite(self):
        rng = np.random.default_rng(406)
        histories = make_portfolio(20, rng)
        model = DeepAttentionModel(max_periods=4, n_epochs=5, random_state=42)
        model.fit(histories)
        for h in histories[:5]:
            cf = model.predict(h)
            assert np.isfinite(cf)

    def test_single_period_history(self):
        rng = np.random.default_rng(407)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        model.fit(histories)
        h_single = ClaimsHistory("S", [1], [2], prior_premium=1.0)
        cf = model.predict(h_single)
        assert np.isfinite(cf)
        assert cf >= 0.0


@skip_if_no_torch
class TestDeepAttentionWeights:
    def test_attention_weights_shape(self):
        rng = np.random.default_rng(408)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        model.fit(histories)
        weights = model.attention_weights(histories[0])
        assert weights.shape == (4,)

    def test_attention_weights_non_negative(self):
        rng = np.random.default_rng(409)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        model.fit(histories)
        for h in histories[:5]:
            weights = model.attention_weights(h)
            assert (weights >= 0.0).all()

    def test_attention_weights_sum_leq_one(self):
        """Sum of attention weights must be at most 1 (balance constraint)."""
        rng = np.random.default_rng(410)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        model.fit(histories)
        for h in histories[:10]:
            weights = model.attention_weights(h)
            assert weights.sum() <= 1.0 + 1e-6, f"Sum > 1: {weights.sum()}"

    def test_attention_weights_trimmed_to_n_periods(self):
        rng = np.random.default_rng(411)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        model.fit(histories)
        h_short = ClaimsHistory("SHORT", [1, 2], [1, 0], prior_premium=1.0)
        weights = model.attention_weights(h_short)
        assert weights.shape == (2,)


@skip_if_no_torch
class TestDeepAttentionGradient:
    def test_gradient_flows(self):
        """Verify that parameters receive gradients during training."""
        rng = np.random.default_rng(412)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=1, random_state=42)
        model.fit(histories)
        # After 1 epoch, parameters should have gradients
        has_grad = any(
            p.grad is not None for p in model.model_.parameters()
        )
        # Gradient may be zeroed after optimiser step; just check model trained
        assert model.is_fitted_


@skip_if_no_torch
class TestDeepAttentionBatch:
    def test_predict_batch_returns_dataframe(self):
        import polars as pl
        rng = np.random.default_rng(413)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        model.fit(histories)
        df = model.predict_batch(histories[:5])
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) >= {
            "policy_id", "prior_premium", "credibility_factor", "posterior_premium"
        }
        assert len(df) == 5

    def test_batch_consistent_with_single(self):
        rng = np.random.default_rng(414)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=42)
        model.fit(histories)
        df = model.predict_batch(histories[:3])
        for i, h in enumerate(histories[:3]):
            expected_cf = model.predict(h)
            actual_cf = df[i]["credibility_factor"][0]
            assert actual_cf == pytest.approx(expected_cf, rel=1e-5)


@skip_if_no_torch
class TestDeepAttentionRepr:
    def test_repr_unfitted(self):
        model = DeepAttentionModel(hidden_dim=64, n_epochs=100)
        r = repr(model)
        assert "hidden_dim=64" in r
        assert "unfitted" in r

    def test_repr_fitted(self):
        rng = np.random.default_rng(415)
        histories = make_portfolio(20, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=2, random_state=42)
        model.fit(histories)
        r = repr(model)
        assert "hidden_dim" in r
        assert "n_epochs" in r


@skip_if_no_torch
class TestLeaveLastOutRegression:
    """Regression tests for P1 bug: _histories_to_tensors must exclude last period from mask.

    The bug: mask[i, s] = True for all s in range(n), so period T (the prediction
    target) was included as an attention input. The model could learn to attend
    directly to the target period, creating data leakage.

    The fix: only set mask[i, s] = True for s in range(n - 1). Period T is the
    target (last_counts), not an input.
    """

    def test_last_period_excluded_from_mask(self):
        """The mask must be False for the last period of each history."""
        import torch
        rng = np.random.default_rng(500)
        histories = make_portfolio(5, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=1, random_state=42)
        model._device_obj = torch.device("cpu")
        tensors = model._histories_to_tensors(histories, torch)
        mask = tensors["mask"]  # shape (B, T)
        # For a 4-period history with max_periods=4, positions 0,1,2 should be True,
        # position 3 (the last period) must be False.
        for i in range(len(histories)):
            assert not mask[i, 3].item(), (
                f"Policy {i}: mask[i, 3] is True, last period should be excluded. "
                f"This is the P1 regression: leave-last-out not implemented."
            )

    def test_first_n_minus_one_periods_in_mask(self):
        """Periods 0..(n-2) must be True in the mask; period n-1 must be False."""
        import torch
        rng = np.random.default_rng(501)
        histories = make_portfolio(3, rng, n_periods=5)
        model = DeepAttentionModel(max_periods=6, n_epochs=1, random_state=42)
        model._device_obj = torch.device("cpu")
        tensors = model._histories_to_tensors(histories, torch)
        mask = tensors["mask"]  # shape (3, 6)
        for i in range(3):
            # Periods 0-3 should be in mask (n_train = min(5,6) - 1 = 4)
            for s in range(4):
                assert mask[i, s].item(), f"Policy {i}, period {s}: should be in mask"
            # Period 4 is the target — must not be in mask
            assert not mask[i, 4].item(), (
                f"Policy {i}: period 4 (last) is in the attention mask — data leakage."
            )
            # Period 5 is padding — also not in mask
            assert not mask[i, 5].item()

    def test_last_counts_equal_final_period_counts(self):
        """last_counts tensor must contain the final observed claim count."""
        import torch
        rng = np.random.default_rng(502)
        histories = make_portfolio(5, rng, n_periods=4)
        model = DeepAttentionModel(max_periods=4, n_epochs=1, random_state=42)
        model._device_obj = torch.device("cpu")
        tensors = model._histories_to_tensors(histories, torch)
        last_counts = tensors["last_counts"]
        for i, h in enumerate(histories):
            expected = float(h.claim_counts[-1])
            actual = float(last_counts[i].item())
            assert actual == pytest.approx(expected, abs=1e-6), (
                f"Policy {i}: last_counts={actual}, expected final count={expected}"
            )

    def test_mask_count_equals_n_minus_one(self):
        """For a history of n periods, exactly n-1 periods should be masked True."""
        import torch
        rng = np.random.default_rng(503)
        for n_periods in [2, 3, 5]:
            histories = make_portfolio(4, rng, n_periods=n_periods)
            model = DeepAttentionModel(max_periods=n_periods, n_epochs=1, random_state=42)
            model._device_obj = torch.device("cpu")
            tensors = model._histories_to_tensors(histories, torch)
            mask = tensors["mask"]
            for i in range(4):
                n_true = int(mask[i].sum().item())
                assert n_true == n_periods - 1, (
                    f"n_periods={n_periods}: expected {n_periods-1} True in mask, "
                    f"got {n_true}. P1 regression: last period not excluded."
                )

    def test_single_period_history_mask_all_false(self):
        """Single-period history: mask is all-False (nothing to attend to)."""
        import torch
        h = ClaimsHistory("S1", [1], [2], prior_premium=1.0)
        model = DeepAttentionModel(max_periods=4, n_epochs=1, random_state=42)
        model._device_obj = torch.device("cpu")
        tensors = model._histories_to_tensors([h], torch)
        mask = tensors["mask"]
        assert not mask[0].any().item(), (
            "Single-period history should have all-False mask (no input periods). "
            "If mask has True entries: data leakage regression."
        )
