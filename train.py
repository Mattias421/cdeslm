"""Training script: neural CDE on Mimi mel cleanup."""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from data import load_data, PATCH_LEN
from model import NeuralCDE
from utils.log_utils import setup_logger
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_step,
    clean_old_checkpoints,
)


@eqx.filter_jit
def mel_loss(model, mimi_mel, ts, target_mel, mel_lens=None):
    """MSE loss between predicted and target mels."""
    pred, _ = jax.vmap(model, in_axes=(0, 0))(mimi_mel, ts)
    if mel_lens is not None:
        seq_len = ts.shape[1]
        mel_lens_capped = jnp.minimum(mel_lens, seq_len)
        indices = jnp.arange(seq_len)
        mask = indices[None, :] < mel_lens_capped[:, None]
        loss = (pred - target_mel[:, :seq_len, :]) ** 2 * mask[:, :, None]
        return jnp.sum(loss) / jnp.sum(mask)
    return jnp.mean((pred - target_mel) ** 2)


grad_loss = eqx.filter_value_and_grad(mel_loss)


@eqx.filter_jit
def make_step(model, mimi_mel, ts, target_mel, optim, opt_state):
    loss, grads = grad_loss(model, mimi_mel, ts, target_mel)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def valid_step(model, data_root, batch_size_valid):
    """Compute average validation loss over all batches."""
    loader_valid = load_data(
        "valid",
        batch_size=batch_size_valid,
        shuffle=False,
        data_root=data_root,
        num_epochs=1,
        training=False,
    )
    total_loss = 0.0
    count = 0
    for batch in loader_valid:
        mimi_mel = batch["mimi_mel"]
        ts = jnp.linspace(0, 1, mimi_mel.shape[1])
        ts = jnp.tile(ts[None, :], (mimi_mel.shape[0], 1))
        target_mel = batch["target_mel"]
        mel_lens = batch.get("mel_lengths")
        loss = mel_loss(model, mimi_mel, ts, target_mel, mel_lens)
        total_loss += float(loss)
        count += 1
    return total_loss / count if count > 0 else 0.0


def generate_visualizations(model, output_dir, step, data_root, batch_size_valid):
    """Plot pred vs target for a batch."""
    viz_dir = Path(output_dir) / "gen_plots" / f"step_{step:06d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    valid_batch = next(
        load_data(
            "valid",
            batch_size=batch_size_valid,
            shuffle=False,
            data_root=data_root,
            num_epochs=1,
            training=False,
        )
    )
    mimi_mel = valid_batch["mimi_mel"]
    ts = jnp.linspace(0, 1, mimi_mel.shape[1])
    ts = jnp.tile(ts[None, :], (mimi_mel.shape[0], 1))
    target_mel = valid_batch["target_mel"]
    valid_ids = valid_batch["ids"]

    pred_mel, _ = jax.vmap(model, in_axes=(0, 0))(mimi_mel, ts)

    for i in range(min(4, len(valid_ids))):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        pred = np.array(pred_mel[i])
        ref = np.array(target_mel[i])
        error = np.abs(pred - ref)

        im1 = ax1.imshow(pred.T, aspect="auto", origin="lower", cmap="jet")
        ax1.set_title(f"{valid_ids[i]} — Predicted")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(ref.T, aspect="auto", origin="lower", cmap="jet")
        ax2.set_title(f"{valid_ids[i]} — Target")
        plt.colorbar(im2, ax=ax2)

        im3 = ax3.imshow(error.T, aspect="auto", origin="lower", cmap="hot")
        ax3.set_title(f"{valid_ids[i]} — Abs Error")
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        fig.savefig(viz_dir / f"valid_{valid_ids[i]}.png", dpi=100)
        plt.close(fig)


def main(
    batch_size=32,
    batch_size_valid=4,
    lr=1e-3,
    steps=1000,
    hidden_size=256,
    width_size=512,
    depth=1,
    dt0=1,
    max_steps=10000,
    solver="euler",
    rtol=1e-2,
    atol=1e-4,
    readout_type="conv",
    seed=5678,
    data_root=None,
    output_dir=None,
    save_every=500,
):
    logger = setup_logger(output_dir)
    key = jr.PRNGKey(seed)

    model_config = {
        "mel_dim": 80,
        "hidden_size": hidden_size,
        "width_size": width_size,
        "depth": depth,
        "dt0": dt0,
        "max_steps": max_steps,
        "solver": solver,
        "rtol": rtol,
        "atol": atol,
        "readout_type": readout_type,
    }

    start_step = 1
    latest_step = get_latest_step(output_dir) if output_dir else None
    opt_state = None

    if latest_step is not None and output_dir:
        logger.info(f"Found checkpoint at step {latest_step}, loading...")
        model, opt_state, start_step, last_loss = load_checkpoint(output_dir)
        logger.info(f"Resuming from step {start_step}, loss: {last_loss:.6f}")
        key = jr.split(key, start_step + 1)[start_step]
    else:
        logger.info("No checkpoint found, starting fresh")
        model = NeuralCDE(**model_config, key=key)

    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    if opt_state is None:
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    loader = load_data(
        "train",
        batch_size=batch_size,
        shuffle=True,
        data_root=data_root,
        step=start_step,
        training=True,
    )

    logger.config(
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        depth=depth,
        hidden_size=hidden_size,
        width_size=width_size,
        solver=solver,
        dt0=dt0,
        seed=seed,
        data_root=str(data_root) if data_root else "default",
        output_dir=output_dir,
        save_every=save_every,
        loss="mse",
    )

    logger.model_summary(model)
    logger.info(f"Patch length: {PATCH_LEN}")
    logger.info("Dataset: Mimi cleanup (mimi_mel → target_mel)")

    step = start_step

    for batch in loader:
        mimi_mel = batch["mimi_mel"]
        seq_len = mimi_mel.shape[1]
        ts = jnp.linspace(0, 1, seq_len)
        ts = jnp.tile(ts[None, :], (mimi_mel.shape[0], 1))
        target_mel = batch["target_mel"]

        loss, model, opt_state = make_step(
            model, mimi_mel, ts, target_mel, optim, opt_state
        )

        logger.train(step, loss)

        if output_dir and step > start_step and step % save_every == 0:
            valid_loss = valid_step(model, data_root, batch_size_valid)
            logger.valid(step, valid_loss)
            generate_visualizations(
                model, output_dir, step, data_root, batch_size_valid
            )
            save_checkpoint(
                model, opt_state, step, valid_loss, output_dir, model_config
            )
            logger.info(f"Checkpoint saved at step {step}")
            clean_old_checkpoints(output_dir, keep_recent=1000)

        step += 1
        if step >= steps:
            break

    if output_dir:
        valid_loss = valid_step(model, data_root, batch_size_valid)
        logger.valid(step, valid_loss)
        save_checkpoint(model, opt_state, step, valid_loss, output_dir, model_config)
        logger.info(f"Final checkpoint saved at step {step}")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural CDE on Mimi cleanup")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_valid", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--width_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--dt0", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--solver", default="euler", choices=["euler", "tsit5"])
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--linear_readout", action="store_true")
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.batch_size = 2
        args.batch_size_valid = 2
        args.steps = 8
        args.hidden_size = 32
        args.width_size = 64
        args.save_every = 4
        args.output_dir = "/home/me/exp/cde_debug"
        print(
            "DEBUG MODE: batch_size=2, steps=8, hidden_size=32, width_size=64, save_every=4"
        )

    main(
        batch_size=args.batch_size,
        batch_size_valid=args.batch_size_valid,
        lr=args.lr,
        steps=args.steps,
        hidden_size=args.hidden_size,
        width_size=args.width_size,
        depth=args.depth,
        dt0=args.dt0,
        max_steps=args.max_steps,
        solver=args.solver,
        rtol=args.rtol,
        atol=args.atol,
        readout_type="linear" if args.linear_readout else "conv",
        seed=args.seed,
        data_root=Path(args.data_root) if args.data_root else None,
        output_dir=args.output_dir,
        save_every=args.save_every,
    )
