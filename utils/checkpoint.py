"""Checkpoint utilities using eqx.tree_serialise_leaves."""

import json
import equinox as eqx
import jax.random as jr
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


def get_checkpoint_dir(output_dir: str) -> Path:
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_checkpoint_path(output_dir: str, step: int) -> Path:
    ckpt_dir = get_checkpoint_dir(output_dir)
    return ckpt_dir / f"checkpoint_step_{step:08d}.eqx"


def get_latest_step(output_dir: str) -> Optional[int]:
    ckpt_dir = get_checkpoint_dir(output_dir)
    if not ckpt_dir.exists():
        return None

    steps = []
    for f in ckpt_dir.glob("checkpoint_step_*.eqx"):
        try:
            step = int(f.stem.replace("checkpoint_step_", ""))
            steps.append(step)
        except ValueError:
            continue

    if not steps:
        return None
    return max(steps)


def save_checkpoint(
    model,
    opt_state,
    step: int,
    loss: float,
    output_dir: str,
    hyperparams: Dict[str, Any],
):
    """Save checkpoint using eqx.tree_serialise_leaves."""
    ckpt_path = get_checkpoint_path(output_dir, step)

    data = {
        "step": step,
        "loss": float(loss),
        **hyperparams,
    }

    with open(ckpt_path, "wb") as f:
        f.write((json.dumps(data) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_checkpoint(
    output_dir: str, step: Optional[int] = None
) -> Tuple[Any, Any, int, float]:
    """Load checkpoint: recreate model and load weights.

    Returns:
        Tuple of (model, opt_state, step, loss)
    """
    # Import here to avoid circular import
    from model import NeuralCDE
    import optax

    if step is None:
        step = get_latest_step(output_dir)
        if step is None:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")

    ckpt_path = get_checkpoint_path(output_dir, step)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(ckpt_path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        step = hyperparams["step"]
        loss = hyperparams["loss"]

        model = NeuralCDE(
            mel_dim=hyperparams.get("mel_dim", 80),
            hidden_size=hyperparams.get("hidden_size", 256),
            width_size=hyperparams.get("width_size", 512),
            depth=hyperparams.get("depth", 1),
            dt0=hyperparams.get("dt0", 1),
            max_steps=hyperparams.get("max_steps", 10000),
            solver=hyperparams.get("solver", "euler"),
            rtol=hyperparams.get("rtol", 1e-2),
            atol=hyperparams.get("atol", 1e-4),
            readout_type=hyperparams.get("readout_type", "conv"),
            key=jr.PRNGKey(0),
        )

        # Load model weights from serialized data
        model = eqx.tree_deserialise_leaves(f, model)

        # Recreate optimizer state with same structure as train.py
        optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(1.0),  # LR doesn't matter for structure
        )
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        return model, opt_state, step, loss


def clean_old_checkpoints(output_dir: str, keep_recent: int = 5):
    """Clean up old checkpoints."""
    ckpt_dir = get_checkpoint_dir(output_dir)
    if not ckpt_dir.exists():
        return

    steps = []
    for f in ckpt_dir.glob("checkpoint_step_*.eqx"):
        try:
            step = int(f.stem.replace("checkpoint_step_", ""))
            steps.append((step, f))
        except ValueError:
            continue

    if len(steps) <= keep_recent:
        return

    steps.sort(key=lambda x: x[0])
    for step, path in steps[:-keep_recent]:
        path.unlink()


def get_best_step(output_dir: str) -> int:
    """Get step with minimum validation loss from valid_loss.tsv."""
    valid_loss_file = Path(output_dir) / "valid_loss.tsv"
    if not valid_loss_file.exists():
        return get_latest_step(output_dir)

    losses = []
    with open(valid_loss_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                step, loss = int(parts[0]), float(parts[1])
                losses.append((step, loss))

    if not losses:
        return get_latest_step(output_dir)

    return min(losses, key=lambda x: x[1])[0]
