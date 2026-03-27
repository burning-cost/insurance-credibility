"""
Targeted coverage tests for insurance_credibility/experience/attention.py.

The existing test_attention.py covers the happy path and the leave-last-out
regression. This file covers the remaining uncovered branches:

- _require_torch when torch is absent (mocked)
- verbose=True fit path (prints epoch losses)
- device resolution: explicit device string, cuda unavailable fallback to cpu
- _poisson_deviance static method (y=0 guard, non-zero y)
- _histories_to_tensors edge cases: single period, max_periods truncation,
  period index encoding, exposure rate clamp
- predict / predict_batch / attention_weights before fit: RuntimeError
- predict_batch posterior_premium = prior * cf
- attention_weights with single-period history (all-False mask)
- __repr__ with custom hidden_dim and n_epochs
- fit with random_state=None (no seeding branch)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from insurance_credibility.experience import ClaimsHistory
from insurance_credibility.experience.attention import (
    DeepAttentionModel,
    _TORCH_MISSING_MSG,
    _require_torch,
    _build_attention_model,
)

skip_if_no_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _portfolio(n: int, n_periods: int = 4, seed: int = 0) -> list[ClaimsHistory]:
    rng = np.random.default_rng(seed)
    return [
        ClaimsHistory(
            policy_id=f"P{i}",
            periods=list(range(1, n_periods + 1)),
            claim_counts=rng.poisson(0.5, size=n_periods).tolist(),
            prior_premium=1.0,
        )
        for i in range(n)
    ]


def _fitted_model(n_periods: int = 4, n_policies: int = 20, seed: int = 99) -> DeepAttentionModel:
    """Return a quickly-fitted model for reuse in tests."""
    m = DeepAttentionModel(max_periods=n_periods, n_epochs=3, random_state=seed)
    m.fit(_portfolio(n_policies, n_periods=n_periods, seed=seed))
    return m


# ---------------------------------------------------------------------------
# 1. _require_torch without torch installed (mocked)
# ---------------------------------------------------------------------------

class TestRequireTorchMocked:
    def test_require_torch_raises_when_torch_missing(self):
        """_require_torch should raise ImportError with pip install hint."""
        with patch.dict(sys.modules, {"torch": None}):
            with pytest.raises(ImportError) as exc_info:
                _require_torch()
        assert "pip install" in str(exc_info.value)

    def test_missing_msg_content(self):
        assert "insurance-experience[deep]" in _TORCH_MISSING_MSG
        assert "torch" in _TORCH_MISSING_MSG

    def test_init_raises_when_torch_missing(self):
        """DeepAttentionModel.__init__ calls _require_torch — if torch absent, ImportError raised."""
        from unittest.mock import patch as _patch
        from insurance_credibility.experience import attention as _attn
        with _patch.object(_attn, '_require_torch', side_effect=ImportError(_TORCH_MISSING_MSG)):
            with pytest.raises(ImportError):
                DeepAttentionModel(max_periods=4)

    def test_require_torch_message_has_pip_install(self):
        # Verify the message wording directly; doesn't need torch absent
        assert "pip install insurance-experience[deep]" in _TORCH_MISSING_MSG


# ---------------------------------------------------------------------------
# 2. fit: verbose=True and random_state=None branches
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestFitVerboseAndNoSeed:
    def test_fit_verbose_prints_output(self, capsys):
        """verbose=True should print epoch losses at multiples of 10."""
        hists = _portfolio(20, seed=10)
        m = DeepAttentionModel(max_periods=4, n_epochs=20, random_state=1)
        m.fit(hists, verbose=True)
        out = capsys.readouterr().out
        assert "Epoch" in out
        assert "deviance" in out

    def test_fit_verbose_false_no_output(self, capsys):
        hists = _portfolio(20, seed=11)
        m = DeepAttentionModel(max_periods=4, n_epochs=10, random_state=2)
        m.fit(hists, verbose=False)
        out = capsys.readouterr().out
        assert out == ""

    def test_fit_no_random_state(self):
        """random_state=None should not raise; model fits normally."""
        hists = _portfolio(20, seed=12)
        m = DeepAttentionModel(max_periods=4, n_epochs=3, random_state=None)
        m.fit(hists)
        assert m.is_fitted_

    def test_fit_losses_length_equals_epochs(self):
        hists = _portfolio(20, seed=13)
        n_epochs = 7
        m = DeepAttentionModel(max_periods=4, n_epochs=n_epochs, random_state=3)
        m.fit(hists)
        assert len(m.training_losses_) == n_epochs


# ---------------------------------------------------------------------------
# 3. Device resolution
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestDeviceResolution:
    def test_explicit_cpu_device(self):
        """Passing device='cpu' should store cpu device object."""
        hists = _portfolio(20, seed=20)
        m = DeepAttentionModel(max_periods=4, n_epochs=2, device="cpu", random_state=4)
        m.fit(hists)
        assert str(m._device_obj) == "cpu"

    def test_no_device_defaults_to_cpu_when_no_cuda(self):
        """When device=None and CUDA not available, should resolve to cpu."""
        hists = _portfolio(20, seed=21)
        m = DeepAttentionModel(max_periods=4, n_epochs=2, device=None, random_state=5)
        m.fit(hists)
        # On a Raspberry Pi (no CUDA), should always be cpu
        assert "cpu" in str(m._device_obj)

    def test_resolve_device_with_explicit_string(self):
        """_resolve_device with explicit string bypasses cuda check."""
        import torch
        m = DeepAttentionModel(max_periods=4, device="cpu")
        dev = m._resolve_device(torch)
        assert dev.type == "cpu"

    def test_resolve_device_none_returns_cuda_or_cpu(self):
        """_resolve_device(None) returns cuda if available, else cpu."""
        import torch
        m = DeepAttentionModel(max_periods=4, device=None)
        dev = m._resolve_device(torch)
        assert dev.type in ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# 4. _poisson_deviance static method
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestPoissonDeviance:
    def test_zero_y_does_not_raise(self):
        """y=0 guard: clamp(y, min=1e-8) prevents log(0)."""
        import torch
        y = torch.tensor([0.0, 0.0, 0.0])
        mu = torch.tensor([0.5, 1.0, 2.0])
        loss = DeepAttentionModel._poisson_deviance(y, mu)
        assert torch.isfinite(loss)

    def test_positive_y_finite(self):
        import torch
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = torch.tensor([1.0, 2.0, 3.0])
        loss = DeepAttentionModel._poisson_deviance(y, mu)
        # When y == mu, deviance should be zero
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_deviance_positive_when_miscalibrated(self):
        """Deviance is non-negative for any y, mu > 0."""
        import torch
        rng = np.random.default_rng(30)
        y = torch.tensor(rng.poisson(1.0, 50).astype(np.float32))
        mu = torch.tensor(np.full(50, 0.5, dtype=np.float32))
        loss = DeepAttentionModel._poisson_deviance(y, mu)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 5. _histories_to_tensors edge cases
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestHistoriesToTensors:
    def _model_with_device(self, max_periods: int = 6) -> DeepAttentionModel:
        import torch
        m = DeepAttentionModel(max_periods=max_periods)
        m._device_obj = torch.device("cpu")
        return m

    def test_exposure_division_is_claim_rate(self):
        """claim_rates tensor should equal claim_counts / exposure for each period."""
        import torch
        h = ClaimsHistory("P1", [1, 2, 3], [2, 4, 0], exposures=[1.0, 2.0, 0.5], prior_premium=1.0)
        m = self._model_with_device(max_periods=4)
        tensors = m._histories_to_tensors([h], torch)
        # leave-last-out: periods 0 and 1 are inputs (n_train = 3-1 = 2)
        assert tensors["claim_rates"][0, 0].item() == pytest.approx(2.0 / 1.0, rel=1e-5)
        assert tensors["claim_rates"][0, 1].item() == pytest.approx(4.0 / 2.0, rel=1e-5)
        # Period 2 is excluded from mask (last period)
        assert tensors["mask"][0, 2].item() == False

    def test_period_indices_stored(self):
        """period_indices[i, s] should equal s (0-indexed)."""
        import torch
        h = ClaimsHistory("P1", [1, 2, 3, 4], [0, 1, 0, 2], prior_premium=1.0)
        m = self._model_with_device(max_periods=4)
        tensors = m._histories_to_tensors([h], torch)
        # n_train = 3 (periods 0-2 are inputs)
        for s in range(3):
            assert tensors["period_indices"][0, s].item() == s

    def test_max_periods_truncation(self):
        """Histories longer than max_periods should be truncated to max_periods."""
        import torch
        h = ClaimsHistory("P1", list(range(1, 9)), [0] * 8, prior_premium=1.0)
        m = self._model_with_device(max_periods=4)
        tensors = m._histories_to_tensors([h], torch)
        # n = min(8, 4) = 4; n_train = 3
        assert int(tensors["mask"][0].sum().item()) == 3

    def test_prior_premiums_stored(self):
        import torch
        h = ClaimsHistory("P1", [1, 2], [0, 1], prior_premium=5.0)
        m = self._model_with_device(max_periods=4)
        tensors = m._histories_to_tensors([h], torch)
        assert tensors["prior_premiums"][0].item() == pytest.approx(5.0)

    def test_last_counts_zero_for_empty_claim_counts(self):
        """Edge case: history with single zero-count period."""
        import torch
        h = ClaimsHistory("P1", [1], [0], prior_premium=1.0)
        m = self._model_with_device(max_periods=4)
        tensors = m._histories_to_tensors([h], torch)
        assert tensors["last_counts"][0].item() == pytest.approx(0.0)

    def test_batch_size_matches_histories(self):
        import torch
        hists = _portfolio(5, n_periods=3, seed=40)
        m = self._model_with_device(max_periods=4)
        tensors = m._histories_to_tensors(hists, torch)
        assert tensors["claim_rates"].shape[0] == 5
        assert tensors["mask"].shape == (5, 4)

    def test_near_zero_exposure_clamped(self):
        """Exposure very close to zero should not produce inf claim rate (clamped to 1e-8)."""
        import torch
        # exposure of 0.0 is not allowed by ClaimsHistory validation, but 1e-10 is fine
        # We test via a very small exposure value (valid: > 0)
        h = ClaimsHistory("P1", [1, 2], [1, 0], exposures=[1e-9, 1.0], prior_premium=1.0)
        m = self._model_with_device(max_periods=4)
        tensors = m._histories_to_tensors([h], torch)
        rate = tensors["claim_rates"][0, 0].item()
        # Clamped: 1 / max(1e-9, 1e-8) = 1 / 1e-8 = 1e8 (large but finite)
        assert np.isfinite(rate)


# ---------------------------------------------------------------------------
# 6. Not-fitted error paths
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestNotFittedErrors:
    def test_predict_before_fit_raises_runtime_error(self):
        m = DeepAttentionModel(max_periods=4)
        h = ClaimsHistory("P1", [1, 2], [0, 1], prior_premium=1.0)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict(h)

    def test_predict_batch_before_fit_raises(self):
        m = DeepAttentionModel(max_periods=4)
        hists = _portfolio(5, seed=50)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict_batch(hists)

    def test_attention_weights_before_fit_raises(self):
        m = DeepAttentionModel(max_periods=4)
        h = ClaimsHistory("P1", [1, 2], [0, 1], prior_premium=1.0)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.attention_weights(h)


# ---------------------------------------------------------------------------
# 7. predict_batch: posterior = prior * cf
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestPredictBatchContent:
    def test_posterior_equals_prior_times_cf(self):
        hists = _portfolio(20, seed=60)
        m = _fitted_model(n_periods=4, n_policies=20, seed=60)
        df = m.predict_batch(hists[:5])
        for row in df.iter_rows(named=True):
            expected = row["prior_premium"] * row["credibility_factor"]
            assert row["posterior_premium"] == pytest.approx(expected, rel=1e-5)

    def test_predict_batch_policy_ids(self):
        hists = _portfolio(5, seed=61)
        m = _fitted_model(seed=61)
        df = m.predict_batch(hists)
        ids = df["policy_id"].to_list()
        expected_ids = [h.policy_id for h in hists]
        assert ids == expected_ids

    def test_predict_batch_cf_non_negative(self):
        hists = _portfolio(20, seed=62)
        m = _fitted_model(seed=62)
        df = m.predict_batch(hists)
        assert (df["credibility_factor"].to_numpy() >= 0.0).all()

    def test_predict_batch_empty_list_returns_empty_df(self):
        """Empty input should return an empty DataFrame with correct columns."""
        m = _fitted_model(seed=63)
        df = m.predict_batch([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# 8. attention_weights: single-period history
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestAttentionWeightsSinglePeriod:
    def test_single_period_weights_shape(self):
        """Single-period history should return weights of shape (1,)."""
        m = _fitted_model()
        h = ClaimsHistory("S", [1], [1], prior_premium=1.0)
        w = m.attention_weights(h)
        assert w.shape == (1,)

    def test_single_period_weights_all_zero(self):
        """With all-False mask, sigmoid weights are zeroed. Array may still be non-zero
        due to mask multiplication, but the mask zeros out the weights."""
        m = _fitted_model()
        h = ClaimsHistory("S", [1], [0], prior_premium=1.0)
        w = m.attention_weights(h)
        # All periods are padding-masked, so weights * mask.float() = 0
        assert w[0] == pytest.approx(0.0, abs=1e-6)

    def test_multi_period_weights_trimmed_to_n_periods(self):
        """Returned weights should match history length, not max_periods."""
        m = _fitted_model(n_periods=4)  # max_periods=4
        h = ClaimsHistory("T", [1, 2, 3], [0, 1, 0], prior_premium=1.0)
        w = m.attention_weights(h)
        assert w.shape == (3,)


# ---------------------------------------------------------------------------
# 9. __repr__
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestRepr:
    def test_repr_unfitted_shows_unfitted(self):
        m = DeepAttentionModel(hidden_dim=16, n_epochs=50)
        r = repr(m)
        assert "hidden_dim=16" in r
        assert "unfitted" in r

    def test_repr_fitted_shows_n_epochs(self):
        hists = _portfolio(20, seed=70)
        m = DeepAttentionModel(max_periods=4, n_epochs=3, hidden_dim=8, random_state=7)
        m.fit(hists)
        r = repr(m)
        assert "n_epochs=3" in r
        assert "hidden_dim=8" in r
        assert "unfitted" not in r

    def test_repr_contains_class_name(self):
        m = DeepAttentionModel(max_periods=4)
        assert "DeepAttentionModel" in repr(m)


# ---------------------------------------------------------------------------
# 10. _build_attention_model internals
# ---------------------------------------------------------------------------

@skip_if_no_torch
class TestBuildAttentionModel:
    def test_forward_shape(self):
        """Forward pass should return (B,) tensor."""
        import torch
        model = _build_attention_model(max_periods=5, hidden_dim=16)
        B, T = 4, 5
        claim_rates = torch.rand(B, T)
        exposures = torch.rand(B, T).abs() + 0.1
        period_indices = torch.arange(T).unsqueeze(0).expand(B, -1)
        prior_premium = torch.rand(B) + 0.5
        mask = torch.ones(B, T, dtype=torch.bool)

        out = model(claim_rates, exposures, period_indices, prior_premium, mask)
        assert out.shape == (B,)

    def test_forward_all_masked_uses_prior(self):
        """When mask is all-False, output should equal prior_premium."""
        import torch
        model = _build_attention_model(max_periods=4, hidden_dim=8)
        B, T = 2, 4
        claim_rates = torch.zeros(B, T)
        exposures = torch.zeros(B, T)
        period_indices = torch.zeros(B, T, dtype=torch.long)
        prior = torch.tensor([2.0, 3.0])
        mask = torch.zeros(B, T, dtype=torch.bool)

        out = model(claim_rates, exposures, period_indices, prior, mask)
        # With all-False mask: attn_weights * mask.float() = 0
        # residual_prior = (1 - 0) * prior = prior
        assert out[0].item() == pytest.approx(2.0, rel=1e-4)
        assert out[1].item() == pytest.approx(3.0, rel=1e-4)

    def test_attention_weights_sum_leq_one(self):
        """sigmoid/T normalisation keeps sum(omega) <= 1."""
        import torch
        model = _build_attention_model(max_periods=6, hidden_dim=16)
        B, T = 8, 6
        claim_rates = torch.rand(B, T)
        exposures = torch.rand(B, T) + 0.1
        period_indices = torch.arange(T).unsqueeze(0).expand(B, -1)
        prior = torch.rand(B) + 0.5
        mask = torch.ones(B, T, dtype=torch.bool)

        # Reconstruct attention weights manually to verify constraint
        feats = torch.stack([claim_rates, exposures, period_indices.float() / T], dim=-1)
        attn_logits = model.attn_mlp(feats).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = torch.sigmoid(attn_logits) / T * mask.float()
        sums = attn_weights.sum(dim=1)
        assert (sums <= 1.0 + 1e-5).all()
