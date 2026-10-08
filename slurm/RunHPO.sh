#!/bin/bash
#SBATCH --job-name=DynaSurvHPO4Loss
#SBATCH --output=%x.%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --partition=ai

set -euo pipefail

PROJECT_DIR="/home/m-ben-salah/repos/dynasurv-ite"
CONTAINER_NAME="dynasurv.sif"
SIF="$PROJECT_DIR/$CONTAINER_NAME"

time srun --gpus-per-task=h100:1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    apptainer exec --nv --bind "$PROJECT_DIR:/workspace" "$SIF" \
    bash -c "cd /workspace/scripts && PYTHONPATH=/workspace/src python3 hyperopt/run_optuna.py"

time apptainer exec --nv --bind "$PROJECT_DIR:/workspace" "$SIF" \
    bash -c "cd /workspace/scripts && PYTHONPATH=/workspace/src python3 hyperopt/run_optuna.py --export-only"
