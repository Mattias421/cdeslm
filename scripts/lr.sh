#!/bin/bash

for lr in 1e-4 2e-4 6e-4 8e-4; do
  echo "=== Running lr=$lr ==="
  apptainer run --nv /exp/exp4/acq22mc/apptainer/cdetts.sif python train.py --exp_root $EXP --output_dir /exp/exp4/acq22mc/lr_grid/$lr --batch_size 512 --lr $lr --depth 2 --solver euler --steps 240 --save_every 24
done
