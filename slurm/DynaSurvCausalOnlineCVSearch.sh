#!/bin/bash
#SBATCH --job-name=DynaSurvCausalOnlineCV
#SBATCH --output=logs/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Module and env setup
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.8.0/gcc-13.2.0

source activate pytorch_env

PROJECT_DIR="/workdir/bensalama/DynaSurv"
cd "$PROJECT_DIR/scripts"

export PYTHONPATH="$PROJECT_DIR/src"

FAST_DEV_FLAG=""
if [[ "$DEV_MODE" -eq 1 ]]; then
    FAST_DEV_FLAG="--fast_dev_run"
fi

# Script
python3 DynaSurvCausalOnlineCVSearch.py \
        --trial_id ${SLURM_ARRAY_TASK_ID} \
        --split_seed $SPLIT_SEED \
        --n_folds $N_FOLDS \
        $FAST_DEV_FLAG \
        ${PROJECT_NAME:+--project_name "$PROJECT_NAME"}
