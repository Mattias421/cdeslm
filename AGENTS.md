# AGENTS.md - Agent Guidelines for cdetts

## Project Overview

Speech LM cleanup dataset preparation using Mimi neural audio codec (Kyutai) on LJSpeech at 24kHz. Produces discrete Mimi tokens (12.5 Hz, 8 RVQ codebooks) and paired HiFiGAN-style mel spectrograms (80-band, 86 Hz) for supervised cleanup: target_mels (clean) vs mimi_mels (from Mimi encode-decode).

## Build/Lint/Test Commands

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_prepare_dataset.py

# Run a single test
python -m pytest tests/test_prepare_dataset.py::TestMel::test_mel_shape -v

# Run tests matching a pattern
python -m pytest -k "test_mel" -v
```

### Linting

```bash
# Run ruff linter
cd /home/me/cdetts && ruff check .

# Run with auto-fix
cd /home/me/cdetts && ruff check --fix .
```

### Running the Project

```bash
# Dataset preparation
python prepare_dataset.py --data_root ~/data/LJSpeech-1.1 --filelists ~/cdetts/filelists --output ~/exp --num_codebooks 8
```

## Code Style Guidelines

### General Principles

1. **Be concise** - Answer directly without unnecessary preamble
2. **Be proactive** - Only take actions the user explicitly asks for
3. **Follow existing patterns** - Mimic the code style in the codebase
4. Don't edit anything within ./Matcha-TTS/, use the project root where possible
5. Functional programming is preferred over OOP; avoid unnecessary classes
6. Test driven development: write tests first, then implement
7. If a bug occurs, write a test to make sure the bug doesn't recur

### Python Style

1. **Imports**:
   - Standard library first, then third-party, then local
   - Use explicit imports (avoid `import *`)
   - Group by: stdlib, third-party, local
   ```python
   from pathlib import Path

   import numpy as np
   import torch
   import torchaudio
   import torchaudio.functional as F
   import tqdm
   import yaml
   from transformers import MimiModel, AutoFeatureExtractor
   ```

2. **Type Hints**:
   - Use type hints for function parameters and return values
   - Use `Path` from pathlib for file paths

3. **Naming Conventions**:
   - `snake_case` for functions, variables, and file names
   - `PascalCase` for classes
   - `UPPER_SNAKE_CASE` for constants

4. **Formatting**:
   - Line length: 100 characters
   - Use 4 spaces for indentation
   - One blank line between top-level definitions
   - Use ruff formatting

5. **Error Handling**:
   - Use specific exception types
   - Provide informative error messages

### Constants
   - Define at module level in UPPER_SNAKE_CASE
   - Group related constants
   ```python
   N_FFT = 1024
   NUM_MELS = 80
   HOP_LENGTH = 279
   SAMPLE_RATE = 24000
   NUM_CODEBOOKS = 8
   ```

## Testing Guidelines

1. **Test Organization**:
   - Use class-based test structure with pytest
   - Use fixtures for common setup
   - Group related tests in classes

2. **Test Naming**:
   - `test_<what_is_being_tested>`
   - Be descriptive: `test_target_mel_shape`

## Mimi Codec Conventions

1. **Model**: `MimiModel.from_pretrained("kyutai/mimi")` via transformers
2. **Feature extractor**: `AutoFeatureExtractor.from_pretrained("kyutai/mimi")`
3. **Input**: 24kHz mono audio, float32 in [-1, 1]
4. **Output codes**: shape `(1, num_codebooks, frame_len)` — int64
5. **Frame rate**: 12.5 Hz (80ms per frame)
6. **8 codebooks** default (out of 32 available)
7. **Typical usage**:
   ```python
   model = MimiModel.from_pretrained("kyutai/mimi").to(device)
   encoder_outputs = model.encode(input_values)
   codes = encoder_outputs.audio_codes[:, :num_codebooks, :]
   audio_recon = model.decode(codes, input_values)["audio_values"]
   ```

## Mel Spectrogram Convention

- HiFiGAN-style, computed at 24kHz
- hop_length=279 (scaled from 256@22050), n_fft=1024, win_length=1024, 80 mel bands
- STFT with reflect padding, hann window, log magnitude
- Returns `(mel_len, 80)` float32

## File Paths

1. **Use Pathlib**:
   ```python
   output_path = Path(args.output) / "data" / split
   ```

2. **Path Conventions**:
   - Data: `~/data/LJSpeech-1.1/`
   - Output: `~/exp/data/{train,valid,test}/`
   - Mimi model: loaded from HF hub `kyutai/mimi`

## Key Libraries

- **transformers** — MimiModel, AutoFeatureExtractor
- **torch/torchaudio** — Audio I/O, resampling, STFT
- **numpy** — Array storage and data loading
- **scipy** — Audio I/O (wavfile)
- **librosa** — Mel filterbank computation
- **pytest** — Testing framework
- **tqdm** — Progress bars
- **pyyaml** — YAML config files

## External Resources

- LJSpeech dataset: `~/data/LJSpeech-1.1/`
- Filelists: `~/cdetts/filelists/`
- Mimi model: `kyutai/mimi` on Hugging Face

## Important Notes

1. **Sampling rate conversion**: LJSpeech is 22050Hz, resample to 24000Hz for Mimi
2. **Mel hop_length scaling**: `round(256 * 24000 / 22050) = 279` to maintain ~86 Hz frame rate
3. **Pair alignment**: target_mels and mimi_mels should have the same frame count; trim to min when near-boundary samples produce slightly different lengths
4. **Mimi codebook count**: 8 codebooks balances quality and compression; 32 yields near-perfect reconstruction
5. **Variable-length storage**: Use numpy object arrays + np.savez_compressed for mel/code arrays; flat np.save for ids and lengths
6. **PyTorch 2.6+**: No special patches needed for transformers >= 4.45
