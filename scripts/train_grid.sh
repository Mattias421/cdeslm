#!/bin/bash
#SBATCH --job-name=cdetts_grid
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/grid-%a.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

# Grid search over LR and depth
# LR values: 0.2, 0.23, 0.26, 0.16
# Depth values: 1, 2, 4
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

declare -a LR_VALS=(0.001 0.0006 0.002  0.0048)
declare -a DEPTH_VALS=(1 2 4)

# Array index -> LR and depth mapping
# idx:  0  1  2  3  4  5  6  7  8  9 10 11
# lr:   0.2 0.2 0.2 0.23 0.23 0.23 0.26 0.26 0.26 0.16 0.16 0.16
# depth:1   2   4   1    2    4   1    2    4   1    2    4

LR_IDX=$((SLURM_ARRAY_TASK_ID % 4))
DEPTH_IDX=$((SLURM_ARRAY_TASK_ID / 4))

LR="${LR_VALS[$LR_IDX]}"
DEPTH="${DEPTH_VALS[$DEPTH_IDX]}"

EXP_ROOT="${EXP:-/home/me/cdetts/exp}"
OUTPUT_DIR="${EXP_ROOT}/cde_lrd_grid/lr${LR}_d${DEPTH}"

echo "=== Grid Search Job ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "LR: $LR"
echo "Depth: $DEPTH"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

cd /home/me/cdetts

apptainer run --nv $EXP/apptainer/cdetts.sif \
    python train.py \
    --exp_root "$EXP" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 2048 \
    --lr "$LR" \
    --depth "$DEPTH" \
    --steps 8000 \
    --save_every 6
