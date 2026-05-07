"""Mimi dataset: patch-based loading for CDE training."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

PATCH_LEN = 256
DEFAULT_DATA_ROOT = Path.home() / "exp/small_run/data"


class MimiDataset:
    """Variable-length mimi/target mel pairs with patch-based access."""

    def __init__(
        self, split: str, data_root: Path | None = None, min_length: int = PATCH_LEN
    ):
        self.split = split
        if data_root is None:
            data_root = DEFAULT_DATA_ROOT
        split_dir = Path(data_root) / split

        self.ids = np.load(split_dir / "ids.npy", allow_pickle=True)
        target = np.load(split_dir / "target_mels.npz", allow_pickle=True)
        mimi = np.load(split_dir / "mimi_mels.npz", allow_pickle=True)
        self.target_mels = target.f.arr_0
        self.mimi_mels = mimi.f.arr_0
        self.mel_lengths = np.load(split_dir / "mel_lengths.npy")

        if min_length > 0:
            mask = self.mel_lengths >= min_length
            self.ids = self.ids[mask]
            self.target_mels = self.target_mels[mask]
            self.mimi_mels = self.mimi_mels[mask]
            self.mel_lengths = self.mel_lengths[mask]

        self.num_samples = len(self.mel_lengths)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "target_mel": self.target_mels[idx],
            "mimi_mel": self.mimi_mels[idx],
            "mel_length": int(self.mel_lengths[idx]),
            "id": self.ids[idx],
        }


def collate_patch(samples: list[dict], patch_len: int = PATCH_LEN) -> dict:
    """Collate into a batch with random patches, aligned across mimi and target."""
    mimi_list = []
    target_list = []
    ids_list = []

    for s in samples:
        mel_len = s["mel_length"]
        start = np.random.randint(0, mel_len - patch_len + 1)
        mimi_list.append(s["mimi_mel"][start : start + patch_len])
        target_list.append(s["target_mel"][start : start + patch_len])
        ids_list.append((s["id"], start))

    return {
        "mimi_mel": jnp.asarray(np.stack(mimi_list), dtype=jnp.float32),
        "target_mel": jnp.asarray(np.stack(target_list), dtype=jnp.float32),
        "ids": ids_list,
    }


def collate_valid(samples: list[dict], max_len: int = 1024) -> dict:
    """Collate for validation: pad to max_len, track original lengths."""
    mimi_list = []
    target_list = []
    mel_lens = []
    ids_list = []

    for s in samples:
        mel_len = min(s["mel_length"], max_len)
        mel_lens.append(mel_len)

        mimi_pad = np.zeros((max_len, 80), dtype=np.float32)
        mimi_pad[:mel_len] = s["mimi_mel"][:mel_len]
        mimi_list.append(mimi_pad)

        target_pad = np.zeros((max_len, 80), dtype=np.float32)
        target_pad[:mel_len] = s["target_mel"][:mel_len]
        target_list.append(target_pad)

        ids_list.append(s["id"])

    return {
        "mimi_mel": jnp.asarray(np.stack(mimi_list), dtype=jnp.float32),
        "target_mel": jnp.asarray(np.stack(target_list), dtype=jnp.float32),
        "mel_lengths": jnp.array(mel_lens, dtype=jnp.int32),
        "ids": np.array(ids_list, dtype=object),
    }


def load_data(
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    data_root: Path | None = None,
    step: int = 0,
    num_epochs: int | None = None,
    patch_len: int = PATCH_LEN,
    training: bool = True,
):
    """Generator that yields batches.

    Args:
        split: train/valid/test
        batch_size: samples per batch
        shuffle: whether to shuffle
        data_root: root of the dataset (contains train/, valid/, test/)
        step: current training step (for deterministic resumption)
        num_epochs: number of epochs. None = infinite, 1 = single pass
        patch_len: patch length for training crops
        training: True for random patch collate, False for padded collate
    """
    min_length = patch_len if training else 0
    dataset = MimiDataset(split, data_root, min_length=min_length)
    n_samples = len(dataset)

    epoch = step // n_samples if n_samples > 0 else 0

    while True:
        indices = np.arange(n_samples)
        if shuffle:
            np.random.seed(epoch)
            np.random.shuffle(indices)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            samples = [dataset[int(idx)] for idx in batch_indices]

            if training:
                yield collate_patch(samples, patch_len)
            else:
                yield collate_valid(samples)

        epoch += 1
        if num_epochs and epoch >= num_epochs:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--patch_len", type=int, default=PATCH_LEN)
    args = parser.parse_args()

    dataset = MimiDataset(args.split, Path(args.data_root))
    print(f"Dataset: {args.split}, {len(dataset)} samples")
    print(
        f"  Mel lengths: min={dataset.mel_lengths.min()}, max={dataset.mel_lengths.max()}"
    )

    loader = load_data(
        args.split,
        args.batch_size,
        shuffle=False,
        data_root=Path(args.data_root),
        training=True,
    )
    batch = next(loader)
    print(
        f"Train batch: mimi={batch['mimi_mel'].shape}, target={batch['target_mel'].shape}"
    )

    loader = load_data(
        args.split,
        args.batch_size,
        shuffle=False,
        data_root=Path(args.data_root),
        training=False,
    )
    batch = next(loader)
    print(
        f"Valid batch: mimi={batch['mimi_mel'].shape}, target={batch['target_mel'].shape}, lens={batch['mel_lengths']}"
    )
