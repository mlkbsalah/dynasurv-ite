#!/bin/bash
#SBATCH --job-name=aggregate_trials
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

module anaconda3/2023.09-0/none-none
source activate pytorch_env

PROJECT_DIR="/workdir/bensalama/DynaSurv"
cd "$PROJECT_DIR/scripts"

python AggregateTrials.py --seed $SPLIT_SEED --date $DATE
