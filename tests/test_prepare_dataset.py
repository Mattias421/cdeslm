"""Tests for prepare_dataset.py."""

from pathlib import Path

import numpy as np
import pytest
import torch

from prepare_dataset import mel_spectrogram, load_audio, parse_filelist

N_FFT = 1024
NUM_MELS = 80
HOP_LENGTH = 279
SAMPLE_RATE = 24000


class TestMelSpectrogram:
    def test_mel_shape(self):
        audio = torch.randn(24000)  # 1 second at 24kHz
        mel = mel_spectrogram(audio)
        pad = int((1024 - 279) / 2)
        expected_frames = (24000 + 2 * pad - 1024) // 279 + 1
        assert mel.shape == (expected_frames, NUM_MELS), (
            f"Expected ({expected_frames}, {NUM_MELS}), got {mel.shape}"
        )

    def test_mel_is_float32(self):
        audio = torch.randn(24000)
        mel = mel_spectrogram(audio)
        assert mel.dtype == torch.float32

    def test_mel_values_are_finite(self):
        audio = torch.randn(24000)
        mel = mel_spectrogram(audio)
        assert torch.isfinite(mel).all()

    def test_mel_silence(self):
        audio = torch.zeros(24000)
        mel = mel_spectrogram(audio)
        assert torch.isfinite(mel).all()
        assert (mel < 0).all()  # log of near-zero gives large negative

    def test_mel_sine(self):
        t = torch.linspace(0, 1, 24000)
        audio = 0.5 * torch.sin(2 * torch.pi * 440 * t)
        mel = mel_spectrogram(audio)
        assert torch.isfinite(mel).all()
        assert mel.shape[-1] == NUM_MELS

    def test_mel_normalizes_loud_input(self):
        audio = torch.randn(24000) * 10
        mel = mel_spectrogram(audio)
        assert torch.isfinite(mel).all()
        assert mel.dtype == torch.float32


class TestLoadAudio:
    def test_load_ljspeech_wav(self):
        path = Path.home() / "data/LJSpeech-1.1/wavs/LJ001-0001.wav"
        if not path.exists():
            pytest.skip("LJSpeech not available")
        audio = load_audio(path, 24000)
        assert audio.dim() == 1
        assert audio.dtype == torch.float32
        assert torch.isfinite(audio).all()

    def test_resampling(self):
        path = Path.home() / "data/LJSpeech-1.1/wavs/LJ001-0001.wav"
        if not path.exists():
            pytest.skip("LJSpeech not available")
        audio_22k = load_audio(path, 22050)
        audio_24k = load_audio(path, 24000)
        assert audio_22k.shape[0] != audio_24k.shape[0]
        ratio = audio_24k.shape[0] / audio_22k.shape[0]
        assert abs(ratio - 24000 / 22050) < 0.01


class TestParseFilelist:
    def test_parse_train_filelist(self):
        fl_path = Path.home() / "cdetts/filelists/ljs_audio_text_train_filelist.txt"
        if not fl_path.exists():
            pytest.skip("Filelist not available")
        data_root = Path.home() / "data/LJSpeech-1.1"
        entries = parse_filelist(fl_path, data_root)
        assert len(entries) > 0
        for wav_path, text in entries:
            assert wav_path.exists(), f"Missing: {wav_path}"
            assert isinstance(text, str) and len(text) > 0
            assert wav_path.suffix == ".wav"

    def test_parse_valid_filelist(self):
        fl_path = Path.home() / "cdetts/filelists/ljs_audio_text_valid_filelist.txt"
        if not fl_path.exists():
            pytest.skip("Valid filelist not available")
        data_root = Path.home() / "data/LJSpeech-1.1"
        entries = parse_filelist(fl_path, data_root)
        assert len(entries) > 0


class TestMimiCodec:
    @pytest.fixture(scope="class")
    def models(self):
        from transformers import MimiModel, AutoFeatureExtractor

        model = MimiModel.from_pretrained("kyutai/mimi")
        feat = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        return model, feat

    def test_mimi_encode_shape(self, models):
        model, feat = models
        audio = torch.randn(1, 24000)
        inputs = feat(
            raw_audio=audio.squeeze(0).numpy(), sampling_rate=24000, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model.encode(inputs["input_values"])
        codes = outputs.audio_codes
        assert codes.dim() == 3
        assert codes.shape[0] == 1  # batch
        assert codes.shape[1] == 32  # all codebooks
        assert codes.shape[2] > 0
        assert codes.dtype == torch.int64

    def test_mimi_decode_shape(self, models):
        model, feat = models
        audio = torch.randn(1, 24000)
        inputs = feat(
            raw_audio=audio.squeeze(0).numpy(), sampling_rate=24000, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model.encode(inputs["input_values"])
            codes = outputs.audio_codes[:, :8, :]
            recon = model.decode(codes, inputs["input_values"])
        audio_values = recon["audio_values"]
        assert audio_values.dim() == 3
        assert audio_values.shape[0] == 1
        assert audio_values.shape[1] == 1
        assert audio_values.shape[2] > 0
        assert torch.isfinite(audio_values).all()

    def test_mimi_8_codebooks(self, models):
        model, feat = models
        audio = torch.randn(1, 24000)
        inputs = feat(
            raw_audio=audio.squeeze(0).numpy(), sampling_rate=24000, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model.encode(inputs["input_values"])
        codes = outputs.audio_codes[:, :8, :]
        assert codes.shape[1] == 8
