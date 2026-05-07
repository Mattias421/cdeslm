# cdetts

Speech LM cleanup dataset — LJSpeech encoded with Mimi neural audio codec (Kyutai), paired with HiFiGAN-style mel spectrograms at 24kHz.

## Project Structure

- `prepare_dataset.py` — Dataset preparation: resamples LJSpeech to 24kHz, computes mel spectrograms (target), encodes/decodes with Mimi, and computes mels from the reconstructed audio (mimi).
- `configs/ljspeech.yaml` — Audio configuration (24kHz, HiFiGAN mel params)
- `data.py` — Dataset loader with variable-length numpy arrays (WIP)
- `train.py` — Training script (WIP)
- `model.py` — Model definition (WIP)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

```bash
python prepare_dataset.py \
  --data_root ~/data/LJSpeech-1.1 \
  --filelists ~/cdetts/filelists \
  --output ~/exp \
  --num_codebooks 8
```

This produces:

```
<output>/data/{train,valid,test}/
  ids.npy              # (N,) object array of sample IDs
  target_mels.npz      # clean mel spectrograms (mel_len, 80) float32
  mimi_mels.npz        # Mimi-reconstructed mel spectrograms (mel_len, 80) float32
  mimi_codes.npz       # Mimi discrete tokens (8, code_len) int64
  mel_lengths.npy      # (N,) int64 mel frame lengths
  wavs/                # Mimi-decoded audio at 24kHz
```

**Arguments:**
- `--data_root` — LJSpeech root directory
- `--filelists` — Filelists directory (ljs_audio_text_*_filelist.txt)
- `--output` — Output root (default: exp)
- `--num_codebooks` — RVQ codebook count (default: 8)
- `--device` — cpu or cuda (default: auto-detect)
- `--config` — Config YAML path

### Audio Configuration

Mel spectrogram parameters (HiFiGAN-style, scaled from 256-hop@22050 → 24kHz):

| Param | Value |
|-------|-------|
| sampling_rate | 24000 |
| hop_length | 279 |
| n_fft | 1024 |
| win_length | 1024 |
| num_mels | 80 |
| f_min | 0 |
| f_max | 12000 |

## Testing

```bash
python -m pytest
```

## TODO

- [ ] dataset loader in data.py
- [ ] training loop
- [ ] model definition (JAX/Equinox)
- [ ] validation with Mimi reconstruction
