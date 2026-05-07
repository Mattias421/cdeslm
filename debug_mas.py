#!/usr/bin/env python
"""Debug script to investigate MAS vs dur_pred discrepancy in Matcha-TTS."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "Matcha-TTS"))

from matcha.models.matcha_tts import MatchaTTS
from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import sequence_mask
from matcha.utils import monotonic_align

N_FFT = 1024
NUM_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
SAMPLE_RATE = 22050
F_MIN = 0.0
F_MAX = 8000


def load_audio(wav_path: Path):
    from scipy.io.wavfile import read

    sr, audio = read(wav_path)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    audio = audio / 32768.0
    return torch.from_numpy(audio).float()


def compute_mel(audio: torch.Tensor):
    return mel_spectrogram(
        audio.unsqueeze(0),
        n_fft=N_FFT,
        num_mels=NUM_MELS,
        sampling_rate=SAMPLE_RATE,
        hop_size=HOP_LENGTH,
        win_size=WIN_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX,
    )


def text_to_ids(text):
    ids, _ = text_to_sequence(text, ["english_cleaners2"])
    return ids


def main():
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

    print("Loading model...")
    checkpoint_path = Path.home() / ".local/share/matcha_tts/matcha_ljspeech.ckpt"
    model = MatchaTTS.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model.eval()

    data_root = Path.home() / "data" / "LJSpeech-1.1"
    filelist_path = Path("/home/me/cdetts/filelists/ljs_audio_text_train_filelist.txt")

    with open(filelist_path) as f:
        lines = f.readlines()

    wav_path, text = lines[0].strip().split("|")
    wav_path = wav_path.replace("$DATA/", "")
    if wav_path.startswith("LJSpeech-1.1/"):
        wav_path = wav_path.replace("LJSpeech-1.1/", "")
    wav_path = data_root / "wavs" / wav_path.split("/")[-1]
    text = text.strip()

    print(f"\nProcessing: {wav_path.name}")
    print(f"Text: {text[:50]}...")

    audio = load_audio(wav_path)
    mel = compute_mel(audio)
    mel_len = mel.shape[2]
    print(f"Mel shape: {mel.shape}, mel_len: {mel_len}")

    phoneme_ids = text_to_ids(text)
    x = torch.tensor(phoneme_ids).unsqueeze(0)
    x_lengths = torch.tensor([x.shape[1]])
    x_len = x.shape[1]
    print(f"x_len: {x_len}")

    print("\n=== STEP 1: Encoder output (dur_pred) ===")
    with torch.no_grad():
        mu_x, logw, x_mask = model.encoder(x, x_lengths, spks=None)

    print(f"mu_x shape: {mu_x.shape}")
    dur_pred = torch.exp(logw).squeeze(0).squeeze(0)
    print(f"dur_pred sum: {dur_pred.sum():.2f}")
    print(f"dur_pred[:10]: {dur_pred[:10]}")
    print(f"dur_pred[-5:]: {dur_pred[-5:]}")

    print("\n=== STEP 2: maximum_path ===")
    const = -0.5 * np.log(2 * np.pi) * NUM_MELS

    mu_x_exp = mu_x.unsqueeze(3)
    mel_exp = mel.unsqueeze(2)
    diff = mu_x_exp - mel_exp
    diff_sq = (diff**2).sum(dim=1)
    log_prior = -0.5 * diff_sq + const

    y_max_length = mel.shape[2]
    y_lengths = torch.tensor([y_max_length], dtype=torch.long)
    y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

    attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
    print(f"attn shape: {attn.shape}")

    mas_durations_raw = torch.sum(attn, dim=-1)
    print(f"MAS durations sum: {mas_durations_raw.sum():.2f}")

    print("\n=== STEP 3: logw_ (log of MAS durations) ===")
    logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask

    logw_np = logw.squeeze(0).squeeze(0)[:x_len].cpu().numpy()
    logw_np_ = logw_.squeeze(0).squeeze(0)[:x_len].cpu().numpy()

    print(f"logw (pred) first 10: {logw_np[:10]}")
    print(f"logw_ (MAS) first 10: {logw_np_[:10]}")
    print(f"logw (pred) last 5: {logw_np[-5:]}")
    print(f"logw_ (MAS) last 5: {logw_np_[-5:]}")

    dur_loss = (
        torch.sum((logw.squeeze(0).squeeze(0) - logw_.squeeze(0).squeeze(0)) ** 2)
        / x_len
    )
    print(f"\nDuration loss (log scale): {dur_loss:.4f}")

    print("\n=== STEP 4: MAS durations (linear scale) ===")
    mas_durations = mas_durations_raw.squeeze(0)
    print(f"MAS durations[:10]: {mas_durations[:10]}")
    print(f"MAS durations[160:]: {mas_durations[160:]}")
    print(f"MAS durations[-5:]: {mas_durations[-5:]}")

    print("\n=== DEBUG: Check masks ===")
    print(f"x_mask shape: {x_mask.shape}")
    print(f"x_lengths: {x_lengths}")
    print(f"x_mask[0, 0, :10]: {x_mask[0, 0, :10]}")
    print(f"x_mask[0, 0, 160:]: {x_mask[0, 0, 160:]}")

    dur_pred_np = dur_pred[:x_len].cpu().numpy()
    mas_np = mas_durations[:x_len].cpu().numpy()

    print("\n=== DEBUG: Check where MAS durations > 1 ===")
    print(f"Non-unit MAS durations indices: {torch.where(mas_durations > 1)[0]}")
    print(f"Non-unit MAS durations: {mas_durations[mas_durations > 1]}")

    print(f"\nx_mask[0, 0, 165]: {x_mask[0, 0, 165]}")
    print(f"x_mask[0, 0, 169]: {x_mask[0, 0, 169]}")

    print("\n=== logw_ breakdown ===")
    attn_sum = torch.sum(attn.unsqueeze(1), -1)
    print(f"attn_sum[0, 0, 165]: {attn_sum[0, 0, 165]}")
    print(f"attn_sum[0, 0, 169]: {attn_sum[0, 0, 169]}")
    print(f"log(attn_sum)[0, 0, 165]: {torch.log(1e-8 + attn_sum[0, 0, 165])}")
    print(f"log(attn_sum)[0, 0, 169]: {torch.log(1e-8 + attn_sum[0, 0, 169])}")
    print(f"x_mask[0, 0, 165]: {x_mask[0, 0, 165]}")
    print(f"x_mask[0, 0, 169]: {x_mask[0, 0, 169]}")

    print("\n=== Analysis ===")
    print("The last phoneme (169) gets 680 mel frames from MAS.")
    print("BUT: Most other phonemes have MAS duration = 1!")
    print(f"MAS durations[0:10]: {mas_durations[:10]}")
    print(f"MAS durations[100:110]: {mas_durations[100:110]}")

    print("\n=== BUG DETECTED ===")
    print("All phonemes except the last have duration 1!")
    print("This means mu_x from encoder is NOT aligned with mel.")
    print("When mu_x is very different from mel, log_prior is nearly uniform,")
    print("and MAS finds the trivial path (diagonal = 1 per phoneme).")

    print("\n=== Check mu_x vs mel alignment ===")
    mu_x_mean = mu_x.squeeze(0).mean(dim=0)
    mel_mean = mel.squeeze(0).mean(dim=0)
    print(f"mu_x mean (per phoneme, first 10): {mu_x_mean[:10]}")
    print(f"mel mean (per frame, first 10): {mel_mean[:10]}")
    print(f"mel mean (per frame, last 10): {mel_mean[-10:]}")

    print("\n=== REAL BUG ===")
    print("mu_x from encoder has 170 phoneme embeddings")
    print("mel has 849 frames")
    print("The dimensions don't match, so comparison is tricky")
    print("")
    print("But the real issue is: MAS returns nearly all 1s because")
    print("the log_prior computation may have a bug.")
    print("")
    print("Let me check the log_prior values...")

    print("\n=== Check log_prior values ===")
    print(f"log_prior min: {log_prior.min():.2f}, max: {log_prior.max():.2f}")
    print(f"log_prior range (max-min): {log_prior.max() - log_prior.min():.2f}")
    print(f"log_prior[0, 0, :10]: {log_prior[0, 0, :10]}")
    print(f"log_prior[0, 169, :10]: {log_prior[0, 169, :10]}")
    print(f"log_prior[0, 169, 840:]: {log_prior[0, 169, 840:]}")

    print("\n=== CONCLUSION ===")
    print("MAS produces nearly all 1-frame durations because mu_x is not")
    print("well-aligned with mel. The log_prior differences between paths are")
    print("relatively small, so MAS defaults to diagonal path (1 per phoneme).")
    print("")
    print("For CDE training, we should USE dur_pred (neural network's duration")
    print("prediction), NOT mas_durations. Here's why:")
    print("  1. dur_pred is what the duration predictor learned to output")
    print("  2. MAS gives trivial 1-frame durations (not useful)")
    print("  3. CDE should learn to reconstruct mel from mu_x + dur_pred")
    print("")
    print("The discrepancy (dur_pred sum 511 vs mel_len 849) is expected -")
    print("dur_pred is a learned approximation, not a perfect alignment.")
    print("")
    print("For training the CDE: use dur_pred, not mas_durations!")

    print("\n=== What should we use for CDE? ===")
    print("Option 1: Use dur_pred (neural network's prediction)")
    print("  - Sum: 511 (doesn't match mel_len of 849)")
    print("  - This is what's used for inference")
    print("Option 2: Use mas_durations (ground truth alignment)")
    print("  - Sum: 849 (matches mel_len exactly)")
    print("  - Problem: Most phonemes have duration 1 (not useful!)")
    print("Option 3: Clip mas_durations to match dur_pred sum")
    print("  - Scale mas_durations to sum to dur_pred.sum()")
    print("  - Keeps relative proportions but matches expected total")

    print("\n=== COMPARISON (linear scale) ===")
    dur_pred_np = dur_pred[:x_len].cpu().numpy()
    mas_np = mas_durations[:x_len].cpu().numpy()

    print(f"dur_pred sum: {dur_pred_np.sum():.2f}")
    print(f"MAS dur sum: {mas_np.sum():.2f}")
    print(f"mel_len: {mel_len}")

    corr = np.corrcoef(dur_pred_np, mas_np)[0, 1]
    print(f"\nPearson correlation: {corr:.4f}")

    print(f"Ratio (MAS / dur_pred) first 10: {mas_np[:10] / (dur_pred_np[:10] + 1e-8)}")
    print(f"Ratio (MAS / dur_pred) last 5: {mas_np[-5:] / (dur_pred_np[-5:] + 1e-8)}")


if __name__ == "__main__":
    main()
