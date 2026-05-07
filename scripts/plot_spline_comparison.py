#!/usr/bin/env python3
"""Plot spline comparison: mu_x, ground truth mel, cubic spline, and rectilinear."""

import argparse
from pathlib import Path

import diffrax as dx
from diffrax import rectilinear_interpolation, LinearInterpolation
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

MAX_LENGTH = 1024


def plot_spline_comparison(
    data_path: str, sample_idx: int, save_path: str | None = None
):
    """Plot comparison of mu_x, ground truth mel, cubic spline, and rectilinear.

    Args:
        data_path: Path to the .npz data file
        sample_idx: Index of sample to visualize
        save_path: If provided, save figure to this path instead of showing
    """
    data = np.load(data_path)

    time_array = data["time_array"]
    mel = data["mel"]
    mel_lengths = data["mel_lengths"]
    coeffs_a = data["coeffs_a"]

    mel_len = int(mel_lengths[sample_idx])
    ts_all = time_array[sample_idx]  # Full 1024 knot timestamps
    mel_vals = mel[sample_idx]

    # mu_x at knot positions (first 80 columns, 1023 intervals = 1024 knots)
    mu_x_knots = coeffs_a[sample_idx][:, :80]  # (1023, 80)

    # Get number of knots (one more than intervals in coeffs)
    n_knots = min(len(ts_all), mu_x_knots.shape[0] + 1)
    ts = ts_all[:n_knots]

    # mu_x at knots - append the last one since coeffs has n-1 intervals
    mu_x_vals = np.concatenate([mu_x_knots, mu_x_knots[-1:]], axis=0)[:n_knots]

    # Compute mel frame indices for each knot using time_array
    knot_mel_indices = np.clip((ts * MAX_LENGTH).round().astype(int), 0, mel_len - 1)

    # Rectilinear manual alignment (for comparison)
    mu_x_aligned = np.zeros((mel_len, 80), dtype=np.float32)
    for k in range(n_knots - 1):
        start_idx = knot_mel_indices[k]
        end_idx = knot_mel_indices[k + 1]
        if end_idx > start_idx:
            mu_x_aligned[start_idx:end_idx] = mu_x_vals[k]
        elif end_idx == start_idx:
            mu_x_aligned[start_idx] = mu_x_vals[k]
    mu_x_aligned[knot_mel_indices[-1] :] = mu_x_vals[-1]
    mu_x_aligned = np.where(mu_x_aligned == 0, mu_x_aligned[-1], mu_x_aligned)

    # Uniform times for evaluation
    uniform_ts = jnp.linspace(0, 1, 1024)

    # Rectilinear interpolation of mu_x using diffrax
    ts_rect, ys_rect = rectilinear_interpolation(
        jnp.array(ts), jnp.array(mu_x_vals[:, :80])
    )
    n_rect = len(ts_rect)
    rect_control = dx.CubicInterpolation(
        ts_rect,
        (
            jnp.zeros((n_rect - 1, 80)),  # d
            jnp.zeros((n_rect - 1, 80)),  # c
            jnp.zeros((n_rect - 1, 80)),  # b
            jnp.array(ys_rect[:-1]),  # a
        ),
    )
    rectilinear_mel = jnp.stack([rect_control.evaluate(t) for t in uniform_ts])

    # Linear interpolation of mu_x using diffrax
    linear_interp = LinearInterpolation(jnp.array(ts), jnp.array(mu_x_vals[:, :80]))
    linear_mel = jnp.stack([linear_interp.evaluate(t) for t in uniform_ts])

    # Cubic spline interpolation of mel using PRECOMPUTED coefficients
    coeffs_d = data["coeffs_d"][sample_idx][:, :80]
    coeffs_c = data["coeffs_c"][sample_idx][:, :80]
    coeffs_b = data["coeffs_b"][sample_idx][:, :80]
    coeffs_a_mel = data["coeffs_a"][sample_idx][:, :80]
    control = dx.CubicInterpolation(
        jnp.array(ts_all),
        (
            jnp.array(coeffs_d),
            jnp.array(coeffs_c),
            jnp.array(coeffs_b),
            jnp.array(coeffs_a_mel),
        ),
    )
    spline_mel = jnp.stack([control.evaluate(t) for t in uniform_ts])

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))

    vmin, vmax = -3, 3

    ax = axes[0]
    mu_x_plot = mu_x_aligned[:mel_len].T
    im = ax.imshow(
        mu_x_plot, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax
    )
    ax.set_ylabel("Features (80)")
    ax.set_xlabel("Phoneme idx")
    ax.set_title(f"mu_x (rectilinear aligned, seq_len={mel_len})")
    plt.colorbar(im, ax=ax, label="value")

    ax = axes[1]
    mel_plot = mel_vals[:mel_len].T
    im = ax.imshow(
        mel_plot, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax
    )
    ax.set_ylabel("Mel (80)")
    ax.set_xlabel("Time idx")
    ax.set_title(f"Ground truth mel (mel_len={mel_len})")
    plt.colorbar(im, ax=ax, label="value")

    ax = axes[2]
    spline_plot = np.array(spline_mel[:mel_len]).T
    im = ax.imshow(
        spline_plot, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax
    )
    ax.set_ylabel("Mel (80)")
    ax.set_xlabel("Time idx")
    mse = np.sum((spline_mel[:mel_len] - mel_vals[:mel_len]) ** 2) / (mel_len * 80)
    ax.set_title(f"Cubic spline interpolation (MSE={mse:.4f})")
    plt.colorbar(im, ax=ax, label="value")

    ax = axes[3]
    rect_plot = np.array(rectilinear_mel[:mel_len]).T
    im = ax.imshow(
        rect_plot, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax
    )
    ax.set_ylabel("Mel (80)")
    ax.set_xlabel("Time idx")
    mse_rect = np.sum((rectilinear_mel[:mel_len] - mel_vals[:mel_len]) ** 2) / (
        mel_len * 80
    )
    ax.set_title(f"Rectilinear mu_x interpolation (MSE={mse_rect:.4f})")
    plt.colorbar(im, ax=ax, label="value")

    ax = axes[4]
    linear_plot = np.array(linear_mel[:mel_len]).T
    im = ax.imshow(
        linear_plot, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax
    )
    ax.set_ylabel("Mel (80)")
    ax.set_xlabel("Time idx")
    mse_linear = np.sum((linear_mel[:mel_len] - mel_vals[:mel_len]) ** 2) / (
        mel_len * 80
    )
    ax.set_title(f"Linear mu_x interpolation (MSE={mse_linear:.4f})")
    plt.colorbar(im, ax=ax, label="value")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot spline comparison")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to .npz data file (default: auto from exp_root)",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default=None,
        help="Experiment root path (for auto-detecting data_path)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Which split to use",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample to visualize",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save figure to this path instead of showing",
    )
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    elif args.exp_root:
        exp_root = args.exp_root
        data_path = f"{exp_root}/ljspeech_feats/{args.split}_data.npz"
    else:
        data_path = str(Path.home() / "exp/ljspeech_feats/valid_data.npz")

    plot_spline_comparison(data_path, args.sample_idx, args.save)


if __name__ == "__main__":
    main()
