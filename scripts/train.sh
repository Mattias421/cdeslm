#!/bin/bash
#SBATCH --job-name=cdetts_train
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%a.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8
bs=2048
apptainer run --nv $EXP/apptainer/cdetts.sif python train.py --exp_root $EXP --output_dir $EXP/cde_bsz$bs --batch_size $bs --steps 19500
