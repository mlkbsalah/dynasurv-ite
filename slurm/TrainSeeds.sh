#!/bin/bash
#SBATCH --job-name=TrainSeeds
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=1-8

# One ensemble member per array task, all on the same temporal split with a
# different seeded init -- the members scripts/RecommendEnsemble.py expects.
#   sbatch slurm/TrainSeeds.sh                  # seeds 1..8
#   sbatch --array=1-10 slurm/TrainSeeds.sh     # seeds 1..10
#   SEED_OFFSET=100 sbatch slurm/TrainSeeds.sh  # seeds 101..108
# Then `make sync` locally and point RecommendEnsemble.py at the new run dirs.

module load anaconda3/2023.09-0/none-none
module load cuda/13.0.2/none-none

source activate pytorch_env

PROJECT_DIR="/workdir/bensalama/DynaSurv"
cd "$PROJECT_DIR/scripts"

export PYTHONPATH="$PROJECT_DIR/src"

SEED=$(( ${SEED_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
python3 TrainDynasurvCausal.py --seed "$SEED"
