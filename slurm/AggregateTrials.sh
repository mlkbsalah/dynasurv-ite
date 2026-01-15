#!/bin/bash
#SBATCH --job-name=aggregate_trials
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

module load anaconda3/2024.06/gcc-13.2.0
source activate pytorch_env

PROJECT_DIR="/workdir/bensalama/DynaSurv"
cd "$PROJECT_DIR/scripts"

#python aggregate_trials.py --seed $SPLIT_SEED
python aggregate_trials.py --seed 1768340925
