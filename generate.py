#!/usr/bin/env python3
"""Generate mel spectrograms from trained Neural CDE model."""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

import diffrax

from data import load_data
from utils.checkpoint import load_checkpoint, get_best_step

CONFIG_PATH = Path(__file__).parent / "configs" / "ljspeech.yaml"


def load_config(config_path: Path | None = None):
    if config_path is None:
        config_path = CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


_config = load_config()
MEL_MEAN = _config["normalize"]["mel_mean"]
MEL_STD = _config["normalize"]["mel_std"]


@eqx.filter_jit
def forward_batch(model, coeffs, ts):
    """Forward pass for a batch."""
    mel_out, hidden = jax.vmap(model, in_axes=(0, 0))(coeffs, ts)
    return mel_out


def denormalize(mel, mu=MEL_MEAN, std=MEL_STD):
    """Denormalize mel spectrogram for vocoder input."""
    return mel * std + mu


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


def plot_and_save_mels(
    gen_dir: Path, plots_dir: Path, figsize: tuple = (10, 4), dpi: int = 100
):
    """Plot all .npy files in gen_dir and save to plots_dir."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(gen_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {gen_dir}")
        return

    print(f"Plotting {len(npy_files)} mel spectrograms...")
    for i, npy_path in enumerate(npy_files):
        mel = np.load(npy_path)
        fig = plot_mel(mel[:100], figsize=figsize)
        output_path = plots_dir / f"{npy_path.stem}.png"
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    print(f"Saved {len(npy_files)} plots to {plots_dir}")


def main(
    checkpoint_dir,
    step=None,
    split="valid",
    batch_size=16,
    num_samples=None,
    output_dir=None,
    save_mel=True,
    exp_root=None,
    config_path: Path | None = None,
    plot_mels=True,
    figsize: tuple = (10, 4),
    dpi: int = 100,
    dt0=None,
    solver=None,
    max_steps=None,
):
    """Generate mel spectrograms from checkpoint."""

    global MEL_MEAN, MEL_STD

    if config_path is not None:
        _config = load_config(Path(config_path))
        MEL_MEAN = _config["normalize"]["mel_mean"]
        MEL_STD = _config["normalize"]["mel_std"]

    step = get_best_step(checkpoint_dir) if step is None else step

    print(f"Loading checkpoint from {checkpoint_dir} at step {step}...")
    model, _, _, _ = load_checkpoint(checkpoint_dir, step=step)
    print(f"Loaded checkpoint from step {step}")

    if dt0 is not None:
        print(f"  Overriding dt0: {model.dt0} -> {dt0}")
        model = eqx.tree_at(lambda m: m.dt0, model, dt0)
    if solver is not None:
        if solver == "euler":
            new_solver = diffrax.Euler()
        elif solver == "tsit5":
            new_solver = diffrax.Tsit5()
        else:
            raise ValueError(f"Unknown solver: {solver}")
        print(f"  Overriding solver: {model.solver} -> {new_solver}")
        model = eqx.tree_at(lambda m: m.solver, model, new_solver)
    if max_steps is not None:
        print(f"  Overriding max_steps: {model.max_steps} -> {max_steps}")
        model = eqx.tree_at(lambda m: m.max_steps, model, max_steps)

    if output_dir is None:
        output_dir = checkpoint_dir
    gen_dir = Path(output_dir) / f"gen_{split}"
    plots_dir = Path(output_dir) / f"gen_{split}_plots"
    gen_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {split} split...")
    loader = load_data(
        split,
        batch_size=batch_size,
        shuffle=False,
        exp_root=exp_root,
        training=(split == "train"),
    )

    from data import DEFAULT_OUTPUT_PATH

    if exp_root is not None:
        dataset_path = Path(exp_root) / "ljspeech_feats" / f"{split}_data.npz"
    else:
        dataset_path = DEFAULT_OUTPUT_PATH / f"{split}_data.npz"

    if dataset_path.exists():
        dataset = np.load(dataset_path, allow_pickle=True)
        total_samples = len(dataset["mel_lengths"])
    else:
        total_samples = float("inf")

    if num_samples is None and split == "train":
        max_samples = 100
    else:
        max_samples = num_samples if num_samples else total_samples

    print(f"Generating up to {max_samples} samples...")

    generated = 0

    for batch in loader:
        if generated >= max_samples:
            break

        coeffs = batch["coeffs"]
        ts = jnp.asarray(batch["ts"])
        mel_lens = jnp.asarray(batch["mel_lengths"]) if "mel_lengths" in batch else None
        ids = batch["ids"]

        pred = forward_batch(model, coeffs, ts)

        for j, sample_id in enumerate(ids):
            if generated >= max_samples:
                break

            if isinstance(sample_id, tuple):
                base_id, patch_start = sample_id
                mel_len = 256
                out_name = f"{base_id}_patch_{patch_start}"
            else:
                mel_len = int(mel_lens[j]) if mel_lens is not None else 256
                out_name = sample_id

            mel_pred = np.array(pred[j, :mel_len, :])
            mel_denorm = denormalize(mel_pred)

            mel_path = gen_dir / f"{out_name}.npy"
            np.save(mel_path, mel_denorm)

            if isinstance(sample_id, tuple):
                mel_orig = np.array(batch["mel"][j])
                mel_orig_denorm = denormalize(mel_orig)
                orig_path = gen_dir / f"{out_name}_orig.npy"
                np.save(orig_path, mel_orig_denorm)

            generated += 1

        if generated % 100 == 0:
            print(f"Generated {generated} samples...")

    if save_mel:
        print(f"Saved {generated} mel spectrograms to {gen_dir}")

    if plot_mels:
        plot_and_save_mels(gen_dir, plots_dir, figsize, dpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mel spectrograms")
    parser.add_argument("checkpoint_dir", type=str, help="Checkpoint directory")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: best checkpoint by valid loss)",
    )
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--exp_root", type=str, default=None, help="Experiment root path"
    )
    parser.add_argument(
        "--config", default="configs/ljspeech.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--no_plot", action="store_true", help="Skip plotting mel spectrograms"
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(10, 4), help="Figure size"
    )
    parser.add_argument("--dpi", type=int, default=100, help="Image DPI")
    parser.add_argument(
        "--dt0",
        type=float,
        default=None,
        help="Override initial timestep for CDE solver",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        choices=["euler", "tsit5"],
        help="Override CDE solver",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Override max solver steps"
    )
    args = parser.parse_args()

    main(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        split=args.split,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        save_mel=True,
        exp_root=args.exp_root,
        config_path=Path(args.config) if args.config else None,
        plot_mels=not args.no_plot,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        dt0=args.dt0,
        solver=args.solver,
        max_steps=args.max_steps,
    )
