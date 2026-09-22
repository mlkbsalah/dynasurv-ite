#!/bin/bash
#SBATCH --job-name=SemiSynthSweep
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=ai
#SBATCH --array=1-30

# One (sweep cell, replicate) per array task: generate the semi-synthetic replicate,
# train DynaSurv on it, then score every checkpoint kind against the simulated truth.
# Cells vary ONE knob of configs/semisynthetic/dgp.toml at a time; the others stay at
# the file's values (gamma = 1, hidden strength = 0, heterogeneity = 1):
#   gamma          0 0.25 0.5 1.0 1.5   assignment confounding (1.0 is the reference cell)
#   strength       0.5 1.0              hidden confounder -> assignment (0 = gamma/1.0)
#   heterogeneity  0 0.5 2.0            effect heterogeneity (1 = gamma/1.0)
# = 10 cells x 3 replicates = 30 tasks. Replicate r uses data seed r and training seed r.
# gamma 2 and 4 are left out on purpose: effective sample size falls to ~0.01, which
# measures extrapolation rather than confounding.
#   sbatch slurm/sweep.sh                       # all 30 tasks
#   sbatch --array=1-30%5 slurm/sweep.sh        # at most 5 running at once
#   sbatch --array=4,14,24 slurm/sweep.sh       # rerun single tasks
#   N_REPS=5 sbatch --array=1-50 slurm/sweep.sh # 5 replicates -> 10 x 5 = 50 tasks
#   EVAL_KINDS="bestCALIB" sbatch slurm/sweep.sh
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=7 bash slurm/sweep.sh   # print the task, run nothing
# Every checkpoint kind is scored by default (val_loss, bestCI, bestCALIB, final_epoch):
# on the first local run the kind changed the counterfactual error by a factor of two,
# so no kind is privileged before the sweep. A missing checkpoint kind only skips its
# own evaluation.
# Outputs, all under the project directory:
#   data/semisynthetic/{axis}/{level}/rep{r}/        expanded data, truth.parquet, manifest.json
#   models/semisynthetic/{axis}/{level}/rep{r}/seed_{r}/eval_{kind}/*.csv
# `make sync` pulls models/ (the eval CSVs); data/ stays on the cluster, so rsync
# data/semisynthetic/*/*/rep*/manifest.json too if you want the achieved rates locally.
# Set WANDB_MODE=offline in the submitting shell if the nodes have no internet.
#
# Requires dynasurv.sif at the project root on the cluster (make build-docker,
# send, build-apptainer). The repo isn't baked into the image, so it's bind-
# mounted into /workspace at run time.

PROJECT_DIR="/home/m-ben-salah/repos/dynasurv-ite"
CONTAINER_NAME="dynasurv"
SIF="$PROJECT_DIR/$CONTAINER_NAME.sif"

CELLS=(
    "gamma 0.0" "gamma 0.25" "gamma 0.5" "gamma 1.0" "gamma 1.5"
    "strength 0.5" "strength 1.0"
    "heterogeneity 0.0" "heterogeneity 0.5" "heterogeneity 2.0"
)
N_REPS=${N_REPS:-3}
EVAL_KINDS=${EVAL_KINDS:-"val_loss bestCI bestCALIB final_epoch"}

TASK=$(( SLURM_ARRAY_TASK_ID - 1 ))
if [ "$TASK" -ge $(( ${#CELLS[@]} * N_REPS )) ]; then
    echo "task $SLURM_ARRAY_TASK_ID is beyond ${#CELLS[@]} cells x $N_REPS replicates" >&2
    exit 1
fi
read -r AXIS LEVEL <<< "${CELLS[$(( TASK / N_REPS ))]}"
REP=$(( TASK % N_REPS ))
SEED=$REP
CELL="$AXIS/$LEVEL/rep$REP"

STEPS="python3 semisynthetic/generate.py --out ../data/semisynthetic/$CELL --replicate $REP --$AXIS $LEVEL"
STEPS="$STEPS && python3 semisynthetic/train.py --axis $AXIS --level $LEVEL --rep $REP --seed $SEED"
for KIND in $EVAL_KINDS; do
    STEPS="$STEPS; python3 semisynthetic/evaluate.py --axis $AXIS --level $LEVEL --rep $REP --seed $SEED --kind $KIND"
done

if [ -n "$DRY_RUN" ]; then
    echo "task $SLURM_ARRAY_TASK_ID -> $CELL (seed $SEED)"
    echo "$STEPS"
    exit 0
fi

time apptainer exec --nv --bind "$PROJECT_DIR:/workspace" "$SIF" \
    bash -c "cd /workspace/scripts && export PYTHONPATH=/workspace/src && $STEPS"
