#!/bin/bash
#SBATCH --job-name=aggregate_trials
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

module load anaconda3
source activate pytorch_env

python aggregate_trials.py --seed $SPLIT_SEED