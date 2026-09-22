#!/bin/bash
#SBATCH --job-name=TrainSeeds
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=ai
#SBATCH --array=1-20

# One ensemble member per array task, all on the same temporal split with a
# different seeded init -- the members scripts/RecommendEnsemble.py expects.
# A single run takes ~1h25 on the gpu partition, so the array is the only cost
# driver; 20 members give p_best a resolution of 0.05 and make the default
# p_best_min = 0.7 an exact 14-of-20 vote (8 members would force 6-of-8 = 0.75).
#   sbatch slurm/TrainSeeds.sh                  # seeds 1..20
#   sbatch --array=1-10 slurm/TrainSeeds.sh     # seeds 1..10
#   sbatch --array=1-20%5 slurm/TrainSeeds.sh   # same seeds, at most 5 running at once
#   SEED_OFFSET=100 sbatch slurm/TrainSeeds.sh  # seeds 101..120
# Then `make sync` locally and point RecommendEnsemble.py at the new run dirs.
#
# Requires dynasurv.sif at the project root on the cluster (make build-docker,
# send, build-apptainer). The repo isn't baked into the image, so it's bind-
# mounted into /workspace at run time.

PROJECT_DIR="/workdir/bensalama/DynaSurv"
CONTAINER_NAME="dynasurv"
SIF="$PROJECT_DIR/$CONTAINER_NAME.sif"

SEED=$(( ${SEED_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))

time apptainer exec --nv --bind "$PROJECT_DIR:/workspace" "$SIF" \
    bash -c "cd /workspace/scripts && PYTHONPATH=/workspace/src python3 TrainDynasurvCausal.py --seed $SEED"
