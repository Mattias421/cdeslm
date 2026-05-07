#!/bin/bash
#SBATCH --job-name=euler_linear_exp
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/exp_euler_linear-%a.out
#SBATCH --array=0-11

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

# Grid: depth(4,6,8) x dt0(2) x model(2) = 12 experiments
# Model: tiny (h=64,w=128) or small (h=128,w=256)
# Indexed 0-11

DEPTH_VALS=(
    4 4  # tiny d4
    6 6  # tiny d6
    8 8  # tiny d8
    4 4  # small d4
    6 6  # small d6
    8 8  # small d8
)

DT0_VALS=(
    0.0009765625 0.00048828125  # tiny d4
    0.0009765625 0.00048828125  # tiny d6
    0.0009765625 0.00048828125  # tiny d8
    0.0009765625 0.00048828125  # small d4
    0.0009765625 0.00048828125  # small d6
    0.0009765625 0.00048828125  # small d8
)

HIDDEN_VALS=(
    64 64     # tiny d4
    64 64     # tiny d6
    64 64     # tiny d8
    128 128   # small d4
    128 128  # small d6
    128 128  # small d8
)

WIDTH_VALS=(
    128 128   # tiny d4
    128 128   # tiny d6
    128 128   # tiny d8
    256 256   # small d4
    256 256   # small d6
    256 256   # small d8
)

IDX=$SLURM_ARRAY_TASK_ID

DEPTH=${DEPTH_VALS[$IDX]}
DT0=${DT0_VALS[$IDX]}
HIDDEN=${HIDDEN_VALS[$IDX]}
WIDTH=${WIDTH_VALS[$IDX]}

if [[ $(echo "$DT0 == 0.0009765625" | bc -l) -eq 1 ]]; then
    DT_STR="dt1024"
else
    DT_STR="dt2048"
fi

OUTPUT_DIR="${EXP}/euler_linear/h${HIDDEN}_w${WIDTH}_d${DEPTH}_${DT_STR}"

echo "=== Euler Linear Experiment ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Hidden: $HIDDEN"
echo "Width: $WIDTH"
echo "Depth: $DEPTH"
echo "dt0: $DT0"
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
    --dt0 $DT0 \
    --steps 8000 \
    --save_every 500 \
    --loss mae
