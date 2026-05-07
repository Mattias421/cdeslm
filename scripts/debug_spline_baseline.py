#!/usr/bin/env python3
"""Debug script to compute cubic spline baseline MSE vs ground truth mel."""

import argparse

import diffrax as dx
import jax.numpy as jnp
import numpy as np


def compute_spline_baseline_mse(data_path: str, num_samples: int = None):
    """Compute MSE of cubic spline interpolation vs ground truth mel.

    For baseline: compute coefficients from ground truth mel at knot positions,
    then evaluate at uniform times to get spline prediction.

    Args:
        data_path: Path to the .npz data file
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Average MSE across samples
    """
    data = np.load(data_path)

    time_array = data["time_array"]  # (N, 1024)
    mel = data["mel"]  # (N, 1024, 80)
    mel_lengths = data["mel_lengths"]  # (N,)

    n_total = len(mel_lengths)
    if num_samples is not None:
        n_samples = min(num_samples, n_total)
    else:
        n_samples = n_total

    print("=== Cubic Spline Baseline ===")
    print(f"Data file: {data_path}")
    print(f"Total samples: {n_total}")
    print(f"Evaluating: {n_samples} samples")

    total_mse_sum = 0.0
    total_active_dims = 0

    for i in range(n_samples):
        mel_len = int(mel_lengths[i])
        ts = time_array[i]  # (1024,) - full time array padded to MAX_LENGTH
        mel_vals = mel[i]  # (1024, 80) - full mel padded

        ts_jax = jnp.array(ts)
        mel_jax = jnp.array(mel_vals)

        d, c, b, a = dx.backward_hermite_coefficients(ts_jax, mel_jax)
        coeffs = (d, c, b, a)

        control = dx.CubicInterpolation(ts_jax, coeffs)

        uniform_ts = jnp.linspace(0, 1, 1024)
        spline_mel = jnp.stack([control.evaluate(t) for t in uniform_ts])

        target_mel = jnp.array(mel[i][:mel_len])

        mse = (spline_mel[:mel_len] - target_mel) ** 2
        total_mse_sum += float(jnp.sum(mse))
        total_active_dims += mel_len * 80

        if (i + 1) % 10 == 0:
            sample_mse = float(jnp.sum(mse)) / (mel_len * 80)
            print(f"  Processed {i + 1}/{n_samples} samples, MSE={sample_mse:.6f}")

    avg_mse = total_mse_sum / total_active_dims
    print("\n=== Results ===")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"RMSE: {jnp.sqrt(jnp.array(avg_mse)):.6f}")

    return avg_mse


def main():
    parser = argparse.ArgumentParser(
        description="Compute cubic spline baseline MSE vs ground truth mel"
    )
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
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    elif args.exp_root:
        exp_root = args.exp_root
        data_path = f"{exp_root}/ljspeech_feats/{args.split}_data.npz"
    else:
        raise ValueError("Must provide either --data_path or --exp_root")

    compute_spline_baseline_mse(data_path, args.num_samples)


if __name__ == "__main__":
    main()
