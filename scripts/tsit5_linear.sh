#!/bin/bash
#SBATCH --job-name=tsit5_linear_exp
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/exp_tsit5_linear-%a.out
#SBATCH --array=0-17

echo "CPU"
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

BATCH_SIZE=16

# Grid: depth(4,6,8) x rtol(3) x model(2) = 18 experiments
# But we do 12 like euler: depth(3) x rtol(2) x model(2) = 12
# Wait, user said default, fast, very_fast = 3 pairs
# depth(3) x rtol(3) x model(2) = 18 experiments
# Array 0-17

# depth: 4,6,8 for tiny and small
DEPTH_VALS=(
    4 4 4  # tiny d4
    6 6 6  # tiny d6
    8 8 8  # tiny d8
    4 4 4  # small d4
    6 6 6  # small d6
    8 8 8  # small d8
)

# rtol: default, fast, very_fast
RTOL_VALS=(
    0.01 0.1 1.0  # tiny d4
    0.01 0.1 1.0  # tiny d6
    0.01 0.1 1.0  # tiny d8
    0.01 0.1 1.0  # small d4
    0.01 0.1 1.0  # small d6
    0.01 0.1 1.0  # small d8
)

# atol matches rtol (paired)
ATOL_VALS=(
    0.0001 0.001 0.1  # tiny d4: 1e-4, 1e-3, 1e-1
    0.0001 0.001 0.1  # tiny d6
    0.0001 0.001 0.1  # tiny d8
    0.0001 0.001 0.1  # small d4
    0.0001 0.001 0.1  # small d6
    0.0001 0.001 0.1  # small d8
)

HIDDEN_VALS=(
    64 64 64     # tiny d4
    64 64 64     # tiny d6
    64 64 64     # tiny d8
    128 128 128  # small d4
    128 128 128  # small d6
    128 128 128  # small d8
)

WIDTH_VALS=(
    128 128 128   # tiny d4
    128 128 128   # tiny d6
    128 128 128   # tiny d8
    256 256 256   # small d4
    256 256 256   # small d6
    256 256 256   # small d8
)

IDX=$SLURM_ARRAY_TASK_ID

DEPTH=${DEPTH_VALS[$IDX]}
RTOL=${RTOL_VALS[$IDX]}
ATOL=${ATOL_VALS[$IDX]}
HIDDEN=${HIDDEN_VALS[$IDX]}
WIDTH=${WIDTH_VALS[$IDX]}

if [[ $(echo "$RTOL == 0.01" | bc -l) -eq 1 ]]; then
    TOL_STR="rtol1e2"  # default
elif [[ $(echo "$RTOL == 0.1" | bc -l) -eq 1 ]]; then
    TOL_STR="rtol1e1"  # fast
else
    TOL_STR="rtol1e0"  # very_fast (rtol=1)
fi

OUTPUT_DIR="${EXP}/tsit5_linear/h${HIDDEN}_w${WIDTH}_d${DEPTH}_${TOL_STR}"

echo "=== Tsit5 Linear Experiment ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Hidden: $HIDDEN"
echo "Width: $WIDTH"
echo "Depth: $DEPTH"
echo "rtol: $RTOL"
echo "atol: $ATOL"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

apptainer run --nv $EXP/apptainer/cdetts.sif \
    python train.py \
    --exp_root "$EXP" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --lr 0.0001 \
    --hidden_size $HIDDEN \
    --width_size $WIDTH \
    --depth $DEPTH \
    --rtol $RTOL \
    --atol $ATOL \
    --solver tsit5 \
    --steps 8000 \
    --save_every 500 \
    --loss mae
