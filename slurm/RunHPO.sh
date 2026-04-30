#!/bin/bash
#SBATCH --job-name=RunHPO
#SBATCH --output=logs/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100

# Module and env setup
module load anaconda3/2023.09-0/none-none
module load cuda/13.0.2/none-none

PROJECT_DIR="/workdir/bensalama/DynaSurv"
cd "$PROJECT_DIR/scripts/hyperopt"

export PYTHONPATH="$PROJECT_DIR/src"

python3 run_optuna.py --n-trials 50
