#!/bin/bash
#SBATCH --job-name=cdetts_gen
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen.out
#SBATCH --mem=64G

# Generate mel spectrograms from best checkpoint
#
# Usage: ./generate.sh <checkpoint_dir>
# Example: ./generate.sh /exp/euler_linear/h64_w128_d4_dt1024

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir>}"
echo "=== Generate ==="
echo "Checkpoint: $CHECKPOINT_DIR"

export JAX_PLATFORMS=cpu

apptainer run $EXP/apptainer/cdetts.sif \
    python generate.py \
    "$CHECKPOINT_DIR" \
    --exp_root $EXP \
    --split valid \
    --batch_size 50 \
