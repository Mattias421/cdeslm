#!/bin/bash
#SBATCH --job-name=cdetts_gen
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen-%a.out
#SBATCH --array=0-11
#SBATCH --mem=64G

declare -a LR_VALS=(0.2 0.23 0.26 0.16)
declare -a DEPTH_VALS=(1 2 4)

LR_IDX=$((SLURM_ARRAY_TASK_ID % 4))
DEPTH_IDX=$((SLURM_ARRAY_TASK_ID / 4))
LR="${LR_VALS[$LR_IDX]}"
DEPTH="${DEPTH_VALS[$DEPTH_IDX]}"

EXP_ROOT="${EXP:-/home/me/cdetts/exp}"
CHECKPOINT_DIR="${EXP_ROOT}/grid_lr${LR}_d${DEPTH}"
OUTPUT_DIR="${CHECKPOINT_DIR}/gen_${SAVE_MEL:-valid}"

echo "=== Generate Job ==="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "LR: $LR, Depth: $DEPTH"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Save mel: ${SAVE_MEL:-false}"

cd /home/me/cdetts

SAVE_MEL_FLAG=""
if [ "${SAVE_MEL:-false}" = "true" ]; then
    SAVE_MEL_FLAG="--save_mel"
fi

apptainer run --nv $EXP/apptainer/cdetts.sif \
    python generate.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split valid \
    --batch_size 50 \
    $SAVE_MEL_FLAG
