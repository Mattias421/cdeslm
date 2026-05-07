#!/usr/bin/env python3
"""Plot side-by-side comparison of target vs mimi mel spectrograms."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(
    data_root: Path, split: str = "train", idx: int = 0, save: bool = False
):
    split_dir = data_root / split

    ids = np.load(split_dir / "ids.npy", allow_pickle=True)
    target = np.load(split_dir / "target_mels.npz", allow_pickle=True)
    mimi = np.load(split_dir / "mimi_mels.npz", allow_pickle=True)

    target_mel = target.f.arr_0[idx]
    mimi_mel = mimi.f.arr_0[idx]
    sid = ids[idx]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    vmin = min(target_mel.min(), mimi_mel.min())
    vmax = max(target_mel.max(), mimi_mel.max())

    im1 = ax1.imshow(
        target_mel.T, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax
    )
    ax1.set_title(f"Target mel — {sid}")
    ax1.set_ylabel("Mel band")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(
        mimi_mel.T, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax
    )
    ax2.set_title(f"Mimi mel — {sid}")
    ax2.set_ylabel("Mel band")
    fig.colorbar(im2, ax=ax2)

    diff = target_mel - mimi_mel
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    im3 = ax3.imshow(
        diff.T,
        aspect="auto",
        origin="lower",
        cmap="jet",
        vmin=-vmax_diff,
        vmax=vmax_diff,
    )
    ax3.set_title("Target − Mimi")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Mel band")
    fig.colorbar(im3, ax=ax3)

    fig.suptitle(
        f"Split: {split}, Sample {idx}: {sid}  |  target={target_mel.shape}  mimi={mimi_mel.shape}",
        fontsize=12,
    )
    plt.tight_layout()

    if save:
        out_path = Path(f"{split}_{idx}_{sid}_mel_comparison.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=Path.home() / "exp/small_run/data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    plot_comparison(Path(args.data_root), args.split, args.idx, save=args.save)
