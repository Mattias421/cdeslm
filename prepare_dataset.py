#!/usr/bin/env python3
"""Prepare LJSpeech dataset with Mimi codes and HiFiGAN-style mel spectrograms at 24kHz."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import tqdm
import yaml

from transformers import MimiModel, AutoFeatureExtractor

CONFIG_PATH = Path(__file__).parent / "configs" / "ljspeech.yaml"

N_FFT = 1024
NUM_MELS = 80
HOP_LENGTH = 279
WIN_LENGTH = 1024
SAMPLE_RATE = 24000
F_MIN = 0
F_MAX = 12000

NUM_CODEBOOKS = 8


def load_config(config_path: Path | None = None):
    if config_path is None:
        config_path = CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


mel_basis_cache = {}
hann_window_cache = {}


def mel_spectrogram(y: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """HiFiGAN-style mel spectrogram. Input: (T,) float32 in [-1, 1]."""
    if y.dim() == 1:
        y = y.unsqueeze(0)
    n_fft = N_FFT
    hop_size = HOP_LENGTH
    win_size = WIN_LENGTH
    num_mels = NUM_MELS
    fmin = F_MIN
    fmax = F_MAX

    if torch.min(y) < -1.0 or torch.max(y) > 1.0:
        y = y / torch.max(torch.abs(y))

    key = f"{fmax}_{y.device}"
    if key not in mel_basis_cache:
        from librosa.filters import mel as librosa_mel_fn

        mel = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(y.device)
        hann_window_cache[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y_pad = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    spec = torch.stft(
        y_pad,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window_cache[str(y.device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec).pow(2).sum(-1)
    spec = torch.sqrt(spec + 1e-9)
    spec = torch.matmul(mel_basis_cache[key], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec.squeeze(0).T  # (mel_len, num_mels)


def load_audio(wav_path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load wav and resample to target_sr. Returns (T,) float32."""
    audio, sr = torchaudio.load(str(wav_path))
    if sr != target_sr:
        audio = F.resample(audio, sr, target_sr)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio.squeeze(0).float()


def parse_filelist(filelist_path: Path, data_root: Path) -> list[tuple[Path, str]]:
    entries = []
    with open(filelist_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wav_rel, text = line.split("|")
            wav_rel = wav_rel.replace("$DATA/", "")
            if wav_rel.startswith("LJSpeech-1.1/"):
                wav_rel = wav_rel.replace("LJSpeech-1.1/", "")
            if not wav_rel.startswith("wavs/"):
                wav_rel = "wavs/" + wav_rel
            if not wav_rel.endswith(".wav"):
                wav_rel += ".wav"
            entries.append((data_root / wav_rel, text))
    return entries


def process_split(
    split: str,
    model: MimiModel,
    feature_extractor: AutoFeatureExtractor,
    filelist: list[tuple[Path, str]],
    output_path: Path,
    num_codebooks: int,
):
    split_dir = output_path / "data" / split
    wav_dir = split_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    ids = []
    target_mels = []
    mimi_mels = []
    mimi_codes = []
    mel_lengths = []

    model.eval()
    device = next(model.parameters()).device

    for wav_path, _ in tqdm.tqdm(filelist, desc=f"{split}"):
        try:
            sid = wav_path.stem
            audio = load_audio(wav_path)  # (T,) float32, 24kHz

            target_mel = mel_spectrogram(audio)  # (mel_len, 80)
            mel_len = target_mel.shape[0]

            inputs = feature_extractor(
                raw_audio=audio.numpy(),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(device)

            with torch.no_grad():
                encoder_outputs = model.encode(input_values)
                audio_codes = encoder_outputs.audio_codes  # (1, K, T_code)
                audio_codes = audio_codes[:, :num_codebooks, :]

                audio_recon = model.decode(audio_codes, input_values)[
                    "audio_values"
                ]  # (1, 1, T_recon)
                audio_recon = audio_recon.squeeze(0).squeeze(0).cpu()  # (T_recon,)

            mimi_mel = mel_spectrogram(audio_recon)  # (mel_len_recon, 80)
            if mimi_mel.shape[0] != mel_len:
                mel_len = min(mel_len, mimi_mel.shape[0])
                target_mel = target_mel[:mel_len]
                mimi_mel = mimi_mel[:mel_len]
                audio_recon = audio_recon[: audio.shape[0]]
                audio = audio[: audio.shape[0]]

            ids.append(sid)
            target_mels.append(target_mel.numpy().astype(np.float32))
            mimi_mels.append(mimi_mel.numpy().astype(np.float32))
            mimi_codes.append(audio_codes.squeeze(0).cpu().numpy().astype(np.int64))
            mel_lengths.append(mel_len)

            torchaudio.save(
                str(wav_dir / f"{sid}.wav"),
                audio_recon.unsqueeze(0),
                SAMPLE_RATE,
            )

        except Exception as e:
            print(f"  Error: {wav_path}: {e}")

    print(f"  {split}: {len(ids)} samples")

    def to_object_array(lst):
        arr = np.empty(len(lst), dtype=object)
        arr[:] = lst
        return arr

    np.savez_compressed(
        split_dir / "target_mels.npz",
        to_object_array(target_mels),
    )
    np.savez_compressed(
        split_dir / "mimi_mels.npz",
        to_object_array(mimi_mels),
    )
    np.savez_compressed(
        split_dir / "mimi_codes.npz",
        to_object_array(mimi_codes),
    )
    np.save(split_dir / "ids.npy", np.array(ids, dtype=object))
    np.save(split_dir / "mel_lengths.npy", np.array(mel_lengths, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LJSpeech dataset with Mimi codes and HiFiGAN mels at 24kHz"
    )
    parser.add_argument("--data_root", required=True, help="LJSpeech root directory")
    parser.add_argument(
        "--filelists", required=True, help="Path to filelists directory"
    )
    parser.add_argument("--output", default="exp", help="Output directory")
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=8,
        help="Number of Mimi RVQ codebooks (default: 8)",
    )
    parser.add_argument(
        "--config",
        default="configs/ljspeech.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading Mimi model on {device}...")
    model = MimiModel.from_pretrained("kyutai/mimi").to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    print("Mimi model loaded.")

    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    filelists_path = Path(args.filelists)

    all_entries = {}
    for name in ["train", "valid", "test"]:
        fl = filelists_path / f"ljs_audio_text_{name}_filelist.txt"
        if not fl.exists():
            continue
        entries = parse_filelist(fl, data_root)
        all_entries[name] = entries
        print(f"{name}: {len(entries)} samples")

    for name, entries in all_entries.items():
        process_split(
            name, model, feature_extractor, entries, output_path, args.num_codebooks
        )

    print("Done!")


if __name__ == "__main__":
    main()
