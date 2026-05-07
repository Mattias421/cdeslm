"""Neural CDE: reconstruct target mels from mimi mels via controlled dynamics."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import nn as jnn
import jax.random as jr
import diffrax


class Func(eqx.Module):
    """Vector field for Neural CDE."""

    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(
        self, data_size: int, hidden_size: int, width_size: int, depth: int, *, key
    ):
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    """Neural CDE that evolves hidden state driven by mimi_mel path."""

    initial: eqx.nn.MLP
    func: Func
    readout: eqx.nn.Conv1d | eqx.nn.Linear
    readout_type: str
    solver: diffrax.AbstractSolver
    dt0: float
    max_steps: int
    stepsize_controller: diffrax.AbstractStepSizeController | None

    def __init__(
        self,
        mel_dim: int = 80,
        hidden_size: int = 256,
        width_size: int = 512,
        depth: int = 1,
        dt0: float = 1.0,
        max_steps: int = 10000,
        solver: str = "euler",
        rtol: float = 1e-2,
        atol: float = 1e-4,
        readout_type: str = "conv",
        *,
        key,
    ):
        ikey, fkey, rkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(mel_dim, hidden_size, width_size, depth, key=ikey)
        self.func = Func(mel_dim, hidden_size, width_size, depth, key=fkey)

        self.readout_type = readout_type
        if readout_type == "linear":
            self.readout = eqx.nn.Linear(
                in_features=hidden_size,
                out_features=mel_dim,
                use_bias=False,
                key=rkey,
            )
        else:
            self.readout = eqx.nn.Conv1d(
                in_channels=hidden_size,
                out_channels=mel_dim,
                kernel_size=5,
                padding=2,
                key=rkey,
            )

        if solver == "euler":
            self.solver = diffrax.Euler()
            self.stepsize_controller = None
        elif solver == "tsit5":
            self.solver = diffrax.Tsit5()
            self.stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'euler' or 'tsit5'.")

        self.dt0 = dt0
        self.max_steps = max_steps

    def __call__(self, mimi_mel: jnp.ndarray, ts: jnp.ndarray):
        """Forward pass.

        Args:
            mimi_mel: (seq_len, 80) — corrupted mel spectrogram from Mimi decode
            ts: (seq_len,) — timestamps normalized to [0, 1)

        Returns:
            pred_mel: (seq_len, 80)
            hidden: (seq_len, hidden_size)
        """
        control = diffrax.LinearInterpolation(ts, mimi_mel)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        y0 = self.initial(control.evaluate(ts[0]))

        dt0 = self.dt0 / mimi_mel.shape[0]
        solve_kwargs = {
            "terms": term,
            "solver": self.solver,
            "t0": ts[0],
            "t1": ts[-1],
            "dt0": dt0,
            "y0": y0,
            "saveat": diffrax.SaveAt(ts=ts),
            "max_steps": self.max_steps,
        }
        if self.stepsize_controller is not None:
            solve_kwargs["stepsize_controller"] = diffrax.ClipStepSizeController(
                self.stepsize_controller, jump_ts=ts
            )

        solution = diffrax.diffeqsolve(**solve_kwargs)
        hidden = solution.ys

        if self.readout_type == "linear":
            mel_out = jax.vmap(self.readout)(hidden)
        else:
            hidden_t = hidden.T
            mel_out = self.readout(hidden_t).T

        return mel_out, hidden
