#!/usr/bin/env python3
"""Visualize extracted features from Matcha-TTS."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_PATH = Path.home() / "exp/ljspeech_feats"
NUM_MELS = 80


def load_split(split):
    features = np.load(OUTPUT_PATH / f"{split}_features.npz")
    labels = np.load(OUTPUT_PATH / f"{split}_labels.npz")
    return features, labels


def get_sample(features, labels, idx):
    x_lens = features["x_lens"]
    mel_lens = labels["mel_lens"]

    cumsum = np.concatenate([[0], np.cumsum(x_lens)])
    start, end = cumsum[idx], cumsum[idx + 1]

    mel_cumsum = np.concatenate([[0], np.cumsum(mel_lens)])
    start_mel, end_mel = mel_cumsum[idx], mel_cumsum[idx + 1]

    return {
        "mu_x": features["mu_x"][start:end],
        "dur_pred": features["dur_pred"][start:end],
        "mas_durations": features["mas_durations"][start:end],
        "phonemes": features["phonemes"][start:end],
        "x_len": x_lens[idx],
        "mel": labels["mel"][start_mel:end_mel],
        "mel_len": mel_lens[idx],
    }


def plot_figure1(sample, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    img = sample["mu_x"].T
    ax.imshow(img, aspect="auto", origin="lower", cmap="jet")
    ax.set_ylabel("Features (80)")
    ax.set_xlabel("Phoneme idx")
    ax.set_title(f"mu_x (seq_len={sample['x_len']})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_figure2(sample, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    img = sample["mel"].T
    ax.imshow(img, aspect="auto", origin="lower", cmap="jet")
    ax.set_ylabel("Mel (80)")
    ax.set_xlabel("Time idx")
    ax.set_title(f"mel (mel_len={sample['mel_len']})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_figure3(sample, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.imshow(sample["mu_x"].T, aspect="auto", origin="lower", cmap="jet")
    x = np.arange(sample["x_len"])
    ax.bar(x, sample["dur_pred"], alpha=0.6, color="red", label="dur_pred", width=1.0)
    ax.bar(
        x,
        sample["mas_durations"],
        alpha=0.6,
        color="lime",
        label="mas_durations",
        width=0.5,
    )
    ax.set_ylabel("Features (80)")
    ax.set_xlabel("Phoneme idx")
    ax.set_title("mu_x with duration overlays")
    ax.legend()

    ax = axes[1]
    ax.hist(sample["dur_pred"], bins=30, alpha=0.6, color="red", label="dur_pred")
    ax.hist(
        sample["mas_durations"],
        bins=30,
        alpha=0.6,
        color="green",
        label="mas_durations",
    )
    ax.set_xlabel("Duration")
    ax.set_ylabel("Count")
    ax.set_title("Duration distribution")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save", action="store_true", help="Save figures instead of showing"
    )
    args = parser.parse_args()

    features, labels = load_split("train")
    sample = get_sample(features, labels, idx=0)

    print(f"Sample 0:")
    print(f"  x_len: {sample['x_len']}")
    print(f"  mel_len: {sample['mel_len']}")
    print(f"  dur_pred sum: {sample['dur_pred'].sum():.1f}")
    print(f"  mas_durations sum: {sample['mas_durations'].sum():.1f}")

    if args.save:
        plot_figure1(sample, "fig1_mu_x.png")
        plot_figure2(sample, "fig2_mel.png")
        plot_figure3(sample, "fig3_alignment.png")
        print("Saved figures to fig1_mu_x.png, fig2_mel.png, fig3_alignment.png")
    else:
        plot_figure1(sample)
        plot_figure2(sample)
        plot_figure3(sample)


if __name__ == "__main__":
    main()
