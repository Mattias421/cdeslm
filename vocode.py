#!/usr/bin/env python3
"""Convert mel spectrograms to waveforms using HiFi-GAN vocoder."""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import tqdm

sys.path.insert(0, str(Path(__file__).parent / "Matcha-TTS"))

from matcha.hifigan.config import v1
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN


_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def load_vocoder(checkpoint_path, device):
    """Load HiFi-GAN vocoder from checkpoint."""
    h = AttrDict(v1)
    vocoder = HiFiGAN(h).to(device)
    vocoder.load_state_dict(
        torch.load(checkpoint_path, map_location=device)["generator"]
    )
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder


def vocode_mel(mel, vocoder, device):
    """Convert mel spectrogram to waveform.

    Args:
        mel: Mel spectrogram array of shape (80, time) or (time, 80)
        vocoder: HiFi-GAN vocoder
        device: torch device

    Returns:
        Waveform as numpy array
    """
    if isinstance(mel, np.ndarray):
        mel = torch.from_numpy(mel).float()

    if mel.shape[0] == 80:
        mel = mel.unsqueeze(0)
    elif mel.shape[-1] == 80:
        mel = mel.unsqueeze(0).transpose(1, 2)
    else:
        raise ValueError(f"Unexpected mel shape: {mel.shape}")

    mel = mel.to(device)

    with torch.no_grad():
        audio = vocoder(mel).clamp(-1, 1)

    return audio.squeeze().cpu().numpy()


def process_folder(input_dir, output_dir, vocoder_checkpoint, device):
    """Process all mel spectrogram files in a folder."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mel_files = sorted(input_path.glob("*.npy"))
    if not mel_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Loading vocoder from {vocoder_checkpoint}...")
    vocoder = load_vocoder(vocoder_checkpoint, device)
    print(f"Loaded vocoder to {device}")

    print(f"Processing {len(mel_files)} mel spectrograms...")
    for mel_file in tqdm.tqdm(mel_files):
        mel = np.load(mel_file)

        if mel.shape[-1] == 80:
            mel = mel.T

        audio = vocode_mel(mel, vocoder, device)

        wav_path = output_path / f"{mel_file.stem}.wav"
        sf.write(wav_path, audio, 22050, "PCM_24")

    print(f"Saved {len(mel_files)} waveforms to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert mel spectrograms to audio")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input folder with .npy mel files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output folder for .wav files"
    )
    parser.add_argument(
        "--vocoder",
        type=str,
        default="hifigan_T2_v1",
        choices=["hifigan_T2_v1", "hifigan_univ_v1"],
        help="Vocoder to use",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    vocoder_checkpoint = Path.home() / ".local/share/matcha_tts" / args.vocoder

    if not vocoder_checkpoint.exists():
        url = (
            "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1"
            if args.vocoder == "hifigan_T2_v1"
            else "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000"
        )
        print(f"Downloading {args.vocoder} from {url}...")
        import urllib.request

        vocoder_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, vocoder_checkpoint)
        print(f"Downloaded to {vocoder_checkpoint}")

    process_folder(args.input_dir, args.output_dir, vocoder_checkpoint, device)


if __name__ == "__main__":
    main()
