#!/bin/bash
# Vocode mel spectrograms to wav files
#
# Usage: ./vocode.sh <gen_dir> [output_dir]
# Example: ./vocode.sh /exp/euler_linear/h64_w128_d4_dt1024/gen_valid

INPUT_DIR="${1:?Usage: $0 <INPUT_DIR> [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$INPUT_DIR/wavs}"

echo "=== Vocode ==="
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

export JAX_PLATFORMS=cpu

apptainer run $EXP/apptainer/cdetts.sif \
    python vocode.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
