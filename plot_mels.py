#!/usr/bin/env python3
"""Plot and save mel spectrograms from .npy files using jet colormap."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_mel(mel: np.ndarray, figsize: tuple = (10, 4)) -> plt.Figure:
    """Plot mel spectrogram with jet colormap.

    Args:
        mel: Mel spectrogram of shape (time, 80) or (80, time)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure
    """
    if mel.shape[0] == 80:
        mel = mel.T
    if mel.shape[0] != 80:
        mel = mel.T

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(mel, aspect="auto", origin="lower", cmap="jet")
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time frame")
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Magnitude")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot mel spectrograms from .npy files"
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .npy files")
    parser.add_argument("output_dir", type=Path, help="Output directory for plots")
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(10, 4), help="Figure size"
    )
    parser.add_argument("--dpi", type=int, default=100, help="Image DPI")
    parser.add_argument(
        "--prefix", type=str, default="mel", help="Filename prefix for output images"
    )
    parser.add_argument("--format", type=str, default="png", help="Output image format")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(args.input_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {args.input_dir}")
        return

    print(f"Found {len(npy_files)} .npy files")

    for i, npy_path in enumerate(npy_files):
        mel = np.load(npy_path)
        fig = plot_mel(mel, figsize=tuple(args.figsize))
        output_path = args.output_dir / f"{args.prefix}_{i:04d}.{args.format}"
        fig.savefig(output_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {output_path}")

    print(f"Done. Saved {len(npy_files)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
