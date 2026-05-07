"""Tests for data.py (MimiDataset)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from data import MimiDataset, collate_patch, collate_valid, load_data

DATA_ROOT = Path.home() / "exp/small_run/data"


class TestMimiDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        if not (DATA_ROOT / "train").exists():
            pytest.skip("Dataset not prepared")
        return MimiDataset("train", DATA_ROOT)

    def test_len(self, dataset):
        assert len(dataset) > 0

    def test_sample_keys(self, dataset):
        sample = dataset[0]
        assert "target_mel" in sample
        assert "mimi_mel" in sample
        assert "mel_length" in sample
        assert "id" in sample

    def test_sample_shapes(self, dataset):
        sample = dataset[0]
        assert sample["target_mel"].ndim == 2
        assert sample["target_mel"].shape[1] == 80
        assert sample["mimi_mel"].shape == sample["target_mel"].shape
        assert sample["mel_length"] == sample["target_mel"].shape[0]

    def test_dtypes(self, dataset):
        sample = dataset[0]
        assert sample["target_mel"].dtype == np.float32
        assert sample["mimi_mel"].dtype == np.float32
        assert isinstance(sample["mel_length"], int)
        assert isinstance(sample["id"], str)

    def test_min_length_filter(self):
        if not (DATA_ROOT / "train").exists():
            pytest.skip("Dataset not prepared")
        ds_long = MimiDataset("train", DATA_ROOT, min_length=500)
        ds_all = MimiDataset("train", DATA_ROOT, min_length=0)
        assert len(ds_long) < len(ds_all) or len(ds_long) == len(ds_all)


class TestCollate:
    @pytest.fixture(scope="class")
    def samples(self):
        if not (DATA_ROOT / "train").exists():
            pytest.skip("Dataset not prepared")
        ds = MimiDataset("train", DATA_ROOT)
        return [ds[i] for i in range(4)]

    def test_collate_patch_shapes(self, samples):
        batch = collate_patch(samples, 256)
        assert batch["mimi_mel"].shape == (4, 256, 80)
        assert batch["target_mel"].shape == (4, 256, 80)
        assert len(batch["ids"]) == 4

    def test_collate_patch_dtype(self, samples):
        batch = collate_patch(samples, 256)
        assert batch["mimi_mel"].dtype == jnp.float32
        assert batch["target_mel"].dtype == jnp.float32

    def test_collate_valid_shapes(self, samples):
        batch = collate_valid(samples, max_len=512)
        assert batch["mimi_mel"].shape == (4, 512, 80)
        assert batch["target_mel"].shape == (4, 512, 80)
        assert batch["mel_lengths"].shape == (4,)
        assert batch["mel_lengths"].dtype == jnp.int32

    def test_collate_valid_respects_length(self, samples):
        batch = collate_valid(samples, max_len=512)
        for i in range(4):
            mel_len = int(batch["mel_lengths"][i])
            assert mel_len <= 512
            assert mel_len == samples[i]["mel_length"] or mel_len == 512


class TestLoadData:
    def test_train_loader(self):
        if not (DATA_ROOT / "train").exists():
            pytest.skip("Dataset not prepared")
        loader = load_data(
            "train", batch_size=4, shuffle=False, data_root=DATA_ROOT, num_epochs=1
        )
        batch = next(loader)
        assert "mimi_mel" in batch
        assert "target_mel" in batch
        assert batch["mimi_mel"].shape[0] == 4
        assert batch["mimi_mel"].shape[2] == 80

    def test_valid_loader(self):
        if not (DATA_ROOT / "valid").exists():
            pytest.skip("Dataset not prepared")
        loader = load_data(
            "valid",
            batch_size=4,
            shuffle=False,
            data_root=DATA_ROOT,
            num_epochs=1,
            training=False,
        )
        batch = next(loader)
        assert "mel_lengths" in batch
        assert batch["mimi_mel"].shape[0] == 4
