#!/usr/bin/env python3
"""Extract text encoder features from Matcha-TTS model for LJSpeech dataset."""

import argparse
import sys
from pathlib import Path

import diffrax
import numpy as np
import torch
import tqdm
import yaml

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

sys.path.insert(0, str(Path(__file__).parent / "Matcha-TTS"))

from matcha.models.matcha_tts import MatchaTTS  # noqa: E402
from matcha.text import text_to_sequence  # noqa: E402
from matcha.utils.audio import mel_spectrogram  # noqa: E402
from matcha.utils.model import sequence_mask, normalize  # noqa: E402
from matcha.utils import monotonic_align as _local_ma  # noqa: E402

CONFIG_PATH = Path(__file__).parent / "configs" / "ljspeech.yaml"


def load_config(config_path: Path | None = None):
    if config_path is None:
        config_path = CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


_config = load_config()
N_FFT = _config["audio"]["n_fft"]
NUM_MELS = _config["audio"]["num_mels"]
HOP_LENGTH = _config["audio"]["hop_length"]
WIN_LENGTH = _config["audio"]["win_length"]
SAMPLE_RATE = _config["audio"]["sampling_rate"]
F_MIN = _config["audio"]["f_min"]
F_MAX = _config["audio"]["f_max"]

MEL_MEAN = _config["normalize"]["mel_mean"]
MEL_STD = _config["normalize"]["mel_std"]

MAX_LENGTH = _config["model"]["max_length"]
MU_LENGTH = 256


def parse_filelist(filelist_path: Path, data_root: Path):
    entries = []
    with open(filelist_path, "r") as f:
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
                wav_rel = wav_rel + ".wav"
            wav_path = data_root / wav_rel
            entries.append((wav_path, text))
    return entries


def load_audio(wav_path: Path):
    from scipy.io.wavfile import read

    sr, audio = read(wav_path)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    audio = audio / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio).float()


def compute_mel(audio: torch.Tensor):
    mel = mel_spectrogram(
        audio.unsqueeze(0),
        n_fft=N_FFT,
        num_mels=NUM_MELS,
        sampling_rate=SAMPLE_RATE,
        hop_size=HOP_LENGTH,
        win_size=WIN_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    mel = normalize(mel, MEL_MEAN, MEL_STD)
    return mel


def text_to_ids(text: str):
    ids, _ = text_to_sequence(text, ["english_cleaners2"])
    ids = ids + [0]  # intersperse with 0
    return ids


def extract_split(model, filelist, output_path, split):
    """Extract features for a split - unpadded variable length."""
    feats_path = output_path / "ljspeech_feats"
    feats_path.mkdir(parents=True, exist_ok=True)
    data_path = feats_path / f"{split}_data.npz"

    if data_path.exists():
        print(f"  {split} already extracted, skipping...")
        return

    coeffs_list = []
    mel_list = []
    mel_lengths_list = []
    ids_list = []

    model.eval()
    for wav_path, text in tqdm.tqdm(filelist, desc=f"Extracting {split}"):
        try:
            audio = load_audio(wav_path)
            mel = compute_mel(audio)
            mel_len = mel.shape[2]

            phoneme_ids = text_to_ids(text)
            x = torch.tensor(phoneme_ids, dtype=torch.long).unsqueeze(0)
            x_lengths = torch.tensor([x.shape[1]], dtype=torch.long)

            with torch.no_grad():
                mu_x, logw, x_mask = model.encoder(x, x_lengths, spks=None)

                y_max_length = mel.shape[2]
                y_lengths = torch.tensor([y_max_length], dtype=torch.long)
                y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
                attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

                const = -0.5 * np.log(2 * np.pi) * NUM_MELS
                factor = -0.5 * torch.ones(
                    mu_x.shape, dtype=mu_x.dtype, device=mu_x.device
                )
                y_square = torch.matmul(factor.transpose(1, 2), mel**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), mel)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = _local_ma.maximum_path(log_prior, attn_mask.squeeze(1))
                attn_trimmed = attn[:, :, :y_max_length]
                mas_durations = torch.sum(attn_trimmed, dim=-1).squeeze(0).long()

                mu_x_np = mu_x.squeeze(0).T.cpu().numpy()
                mel_np = mel.squeeze(0).T.cpu().numpy()

                # Compute phonetic timestamps: START of each phoneme (in mel frames)
                phn_knots = np.concatenate(
                    [[0], np.cumsum(mas_durations.cpu().numpy())]
                )

                mu_x_np = np.vstack([mu_x_np, mu_x_np[-1]])

                mu_x_with_time = np.append(mu_x_np, phn_knots[:, None], axis=-1)

                t_pad = np.linspace(
                    phn_knots[-1] + 1,
                    MAX_LENGTH,
                    int(MU_LENGTH - mu_x_with_time.shape[0]),
                )

                ts = np.append(phn_knots, t_pad)
                mu_pad = np.full((MU_LENGTH, 81), np.nan)
                mu_pad[: mu_x_with_time.shape[0]] = mu_x_with_time

                # Linear interpolation
                mu_x_clean = diffrax.linear_interpolation(
                    ts, mu_pad, fill_forward_nans_at_end=True
                )

                # Pad to MAX_LENGTH for valid/test
                if split in ("valid", "test"):
                    mel_padded = np.pad(
                        mel_np,
                        ((0, MAX_LENGTH - mel_len), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                    mu_x_clean = mu_x_clean.at[:, 80].set(ts)
                    coeffs_list.append(mu_x_clean)
                    mel_list.append(mel_padded.astype(np.float32))
                else:
                    coeffs_list.append(mu_x_clean)
                    mel_list.append(mel_np.astype(np.float32))
                mel_lengths_list.append(mel_len)
                ids_list.append(wav_path.stem)

        except Exception as e:
            print(f"  Error: {wav_path}: {e}")

    if not coeffs_list:
        print(f"  No samples for {split}")
        return

    print(f"  {split}: {len(coeffs_list)} samples")

    if split in ("valid", "test"):
        coeffs_array = np.stack(coeffs_list)
        mel_array = np.stack(mel_list)
    else:
        coeffs_array = np.array(coeffs_list, dtype=object)
        mel_array = np.array(mel_list, dtype=object)

    save_dict = {
        "coeffs_a": coeffs_array,
        "mel": mel_array,
        "mel_lengths": np.array(mel_lengths_list, dtype=np.int64),
        "ids": np.array(ids_list, dtype=object),
    }

    np.savez_compressed(data_path, **save_dict)
    print(f"  Saved {split} to {data_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--filelists", required=True)
    parser.add_argument("--output", default="exp")
    parser.add_argument(
        "--config", default="configs/ljspeech.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    global _config, N_FFT, NUM_MELS, HOP_LENGTH, WIN_LENGTH, SAMPLE_RATE, F_MIN, F_MAX
    global MEL_MEAN, MEL_STD, MAX_LENGTH

    _config = load_config(Path(args.config))
    N_FFT = _config["audio"]["n_fft"]
    NUM_MELS = _config["audio"]["num_mels"]
    HOP_LENGTH = _config["audio"]["hop_length"]
    WIN_LENGTH = _config["audio"]["win_length"]
    SAMPLE_RATE = _config["audio"]["sampling_rate"]
    F_MIN = _config["audio"]["f_min"]
    F_MAX = _config["audio"]["f_max"]
    MEL_MEAN = _config["normalize"]["mel_mean"]
    MEL_STD = _config["normalize"]["mel_std"]
    MAX_LENGTH = _config["model"]["max_length"]

    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    filelists_path = Path(args.filelists)

    print("Loading model...")
    model = MatchaTTS.load_from_checkpoint(
        str(Path.home() / ".local/share/matcha_tts/matcha_ljspeech.ckpt"),
        map_location="cpu",
    )
    model.eval()

    all_entries = {}

    for name in ["train", "valid", "test"]:
        filelist_file = filelists_path / f"ljs_audio_text_{name}_filelist.txt"
        if not filelist_file.exists():
            continue
        entries = parse_filelist(filelist_file, data_root)
        all_entries[name] = entries
        print(f"{name}: {len(entries)} samples")

    for name, entries in all_entries.items():
        extract_split(model, entries, output_path, name)

    print("Done!")


if __name__ == "__main__":
    main()
