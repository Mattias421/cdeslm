"""Tests for train.py."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from model import NeuralCDE
from train import mel_loss, grad_loss, make_step


class TestMelLoss:
    @pytest.fixture
    def model(self):
        return NeuralCDE(
            mel_dim=80,
            hidden_size=64,
            width_size=128,
            depth=1,
            solver="euler",
            key=jr.PRNGKey(0),
        )

    def test_mse_loss_shape(self, model):
        B, T, D = 4, 256, 80
        mimi_mel = jnp.ones((B, T, D))
        ts = jnp.tile(jnp.linspace(0, 1, T)[None, :], (B, 1))
        target = jnp.ones((B, T, D))
        loss = mel_loss(model, mimi_mel, ts, target)
        assert loss.ndim == 0
        assert loss > 0

    def test_mse_loss_perfect_prediction(self, model):
        B, T, D = 2, 64, 80
        mimi_mel = jnp.zeros((B, T, D))
        ts = jnp.tile(jnp.linspace(0, 1, T)[None, :], (B, 1))
        target = jnp.zeros((B, T, D))
        loss = mel_loss(model, mimi_mel, ts, target)
        assert jnp.isfinite(loss)

    def test_grad_loss(self, model):
        B, T, D = 2, 64, 80
        mimi_mel = jnp.ones((B, T, D))
        ts = jnp.tile(jnp.linspace(0, 1, T)[None, :], (B, 1))
        target = jnp.ones((B, T, D))
        loss, grads = grad_loss(model, mimi_mel, ts, target)
        assert jnp.isfinite(loss)
        for g in jax.tree_util.tree_leaves(grads):
            assert jnp.isfinite(g).all()

    def test_make_step(self, model):
        B, T, D = 2, 64, 80
        mimi_mel = jnp.ones((B, T, D))
        ts = jnp.tile(jnp.linspace(0, 1, T)[None, :], (B, 1))
        target = jnp.ones((B, T, D))
        optim = optax.adam(1e-3)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        loss, new_model, new_opt_state = make_step(
            model, mimi_mel, ts, target, optim, opt_state
        )
        assert jnp.isfinite(loss)
        assert loss > 0
        assert new_model is not None
        assert new_opt_state is not None

    def test_loss_with_masking(self, model):
        B, T, D = 4, 256, 80
        mimi_mel = jnp.ones((B, T, D))
        ts = jnp.tile(jnp.linspace(0, 1, T)[None, :], (B, 1))
        target = jnp.ones((B, T, D))
        mel_lens = jnp.array([100, 200, 256, 50], dtype=jnp.int32)
        loss_masked = mel_loss(model, mimi_mel, ts, target, mel_lens)
        loss_full = mel_loss(model, mimi_mel, ts, target)
        assert jnp.isfinite(loss_masked)
        assert loss_masked != loss_full
