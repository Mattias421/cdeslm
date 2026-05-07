"""Tests for model.py (NeuralCDE with mimi_mel control)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from model import NeuralCDE


class TestNeuralCDE:
    @pytest.fixture
    def model(self):
        return NeuralCDE(
            mel_dim=80,
            hidden_size=64,
            width_size=128,
            depth=1,
            solver="euler",
            dt0=1.0,
            key=jr.PRNGKey(0),
        )

    def test_model_creates(self, model):
        assert model is not None

    def test_forward_shape(self, model):
        seq_len = 256
        mimi_mel = jnp.ones((seq_len, 80), dtype=jnp.float32)
        ts = jnp.linspace(0, 1, seq_len, dtype=jnp.float32)
        pred, hidden = model(mimi_mel, ts)
        assert pred.shape == (seq_len, 80)
        assert hidden.shape == (seq_len, 64)

    def test_forward_output_finite(self, model):
        seq_len = 256
        mimi_mel = jnp.ones((seq_len, 80), dtype=jnp.float32)
        ts = jnp.linspace(0, 1, seq_len, dtype=jnp.float32)
        pred, hidden = model(mimi_mel, ts)
        assert jnp.isfinite(pred).all()
        assert jnp.isfinite(hidden).all()

    def test_vmap(self, model):
        B, T, D = 4, 256, 80
        mimi_mel = jnp.ones((B, T, D), dtype=jnp.float32)
        ts = jnp.tile(jnp.linspace(0, 1, T)[None, :], (B, 1))
        pred, hidden = jax.vmap(model, in_axes=(0, 0))(mimi_mel, ts)
        assert pred.shape == (B, T, D)
        assert hidden.shape == (B, T, 64)

    def test_different_inputs_different_outputs(self, model):
        seq_len = 256
        ts = jnp.linspace(0, 1, seq_len)
        mimi_zeros = jnp.zeros((seq_len, 80))
        mimi_ones = jnp.ones((seq_len, 80))
        pred_zeros, _ = model(mimi_zeros, ts)
        pred_ones, _ = model(mimi_ones, ts)
        assert not jnp.allclose(pred_zeros, pred_ones)

    def test_linear_readout(self):
        model = NeuralCDE(
            mel_dim=80,
            hidden_size=64,
            width_size=128,
            depth=1,
            readout_type="linear",
            key=jr.PRNGKey(0),
        )
        seq_len = 256
        mimi_mel = jnp.ones((seq_len, 80), dtype=jnp.float32)
        ts = jnp.linspace(0, 1, seq_len)
        pred, _ = model(mimi_mel, ts)
        assert pred.shape == (seq_len, 80)

    def test_tsit5_solver(self):
        model = NeuralCDE(
            mel_dim=80,
            hidden_size=64,
            width_size=128,
            depth=1,
            solver="tsit5",
            key=jr.PRNGKey(0),
        )
        seq_len = 256
        mimi_mel = jnp.ones((seq_len, 80), dtype=jnp.float32)
        ts = jnp.linspace(0, 1, seq_len)
        pred, _ = model(mimi_mel, ts)
        assert pred.shape == (seq_len, 80)

    def test_invalid_solver(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            NeuralCDE(
                mel_dim=80,
                hidden_size=64,
                width_size=128,
                depth=1,
                solver="dopri5",
                key=jr.PRNGKey(0),
            )
