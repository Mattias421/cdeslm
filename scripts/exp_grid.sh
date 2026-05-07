#!/bin/bash
#SBATCH --job-name=cdetts_exp
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/exp-%a.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-17

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

# Detect server and set batch size
if [[ "$EXP" =~ ^/mnt ]]; then
    # Stanage
    BATCH_SIZE=2048
elif [[ "$EXP" =~ ^/exp ]]; then
    # Mimas
    BATCH_SIZE=64
else
    # Default fallback
    BATCH_SIZE=2048
fi

# Parameter arrays (18 elements each)
# Grid: depth(4,6,8) x use_tanh(t,f) x solver(euler_dt1k, euler_dt2k, tsit5)
# 3 depths x 2 tanh x 3 solvers = 18 experiments

DEPTH_VALS=(
    4 4 4 4 4 4
    6 6 6 6 6 6
    8 8 8 8 8 8
)

USE_TANH_VALS=(
    1 1 1 0 0 0
    1 1 1 0 0 0
    1 1 1 0 0 0
)

SOLVER_VALS=(
    euler euler tsit5
    euler euler tsit5
    euler euler tsit5
    euler euler tsit5
    euler euler tsit5
    euler euler tsit5
)

DT0_VALS=(
    0.0009765625 0.00048828125 0.0009765625
    0.0009765625 0.00048828125 0.0009765625
    0.0009765625 0.00048828125 0.0009765625
    0.0009765625 0.00048828125 0.0009765625
    0.0009765625 0.00048828125 0.0009765625
    0.0009765625 0.00048828125 0.0009765625
)

IDX=$SLURM_ARRAY_TASK_ID

DEPTH=${DEPTH_VALS[$IDX]}
USE_TANH=${USE_TANH_VALS[$IDX]}
SOLVER=${SOLVER_VALS[$IDX]}
DT0=${DT0_VALS[$IDX]}

if [[ $USE_TANH -eq 1 ]]; then
    TANH_STR="tanh"
    TANH_FLAG=""
else
    TANH_STR="softplus"
    TANH_FLAG="--no_tanh"
fi

OUTPUT_DIR="${EXP}/cde_exp/d${DEPTH}_${TANH_STR}_${SOLVER}"

echo "=== Experiment Grid Job ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Depth: $DEPTH"
echo "Use Tanh: $USE_TANH ($TANH_STR)"
echo "Solver: $SOLVER"
echo "dt0: $DT0"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

apptainer run --nv $EXP/apptainer/cdetts.sif \
    python train.py \
    --exp_root "$EXP" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --lr 0.0006 \
    --depth $DEPTH \
    $TANH_FLAG \
    --solver $SOLVER \
    --dt0 $DT0 \
    --steps 8000 \
    --save_every 6
