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
#
# Requires dynasurv.sif at the project root on the cluster (make build-docker,
# send, build-apptainer). The repo isn't baked into the image, so it's bind-
# mounted into /workspace at run time.

PROJECT_DIR="/workdir/bensalama/DynaSurv"
CONTAINER_NAME="dynasurv"
SIF="$PROJECT_DIR/$CONTAINER_NAME.sif"

SEED=$(( ${SEED_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))

apptainer exec --nv --bind "$PROJECT_DIR:/workspace" "$SIF" \
    bash -c "cd /workspace/scripts && PYTHONPATH=/workspace/src python3 TrainDynasurvCausal.py --seed $SEED"
