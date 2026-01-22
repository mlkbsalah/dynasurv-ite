#!/bin/bash

N_TRIALS=100
N_FOLDS=5
PROJECT_NAME=""

# Generate a default random seed if none is provided
SPLIT_SEED=$(( $(date +%s) % 2147483647 ))

# Create log file with timestamp
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cv_search_${SPLIT_SEED}_$(date +%Y%m%d_%H%M%S).log"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --n_trials      Number of trials (default: 100)"
    echo "  -f, --n_folds       Number of folds (default: 5)"
    echo "  -s, --split_seed    Split seed (default: random)"
    echo "  -p, --project_name  Project name for logging"
    echo "  -dev, --dev_mode    Enable fast development run mode"
    echo "  -h, --help          Display this help message"
    exit 0
}

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--n_trials)     N_TRIALS="$2"; shift 2 ;;
    -f|--n_folds)      N_FOLDS="$2";  shift 2 ;;
    -s|--split_seed)   SPLIT_SEED="$2"; shift 2 ;;
    -p|--project_name) PROJECT_NAME="$2"; shift 2 ;;
    -dev|--dev_mode)   DEV_MODE=1; shift 1 ;;
    -h|--help)         usage ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

log_message "Starting CV run with seed: $SPLIT_SEED, trials: $N_TRIALS, folds: $N_FOLDS, dev mode: ${DEV_MODE:-0}"

#Submit trials
ARRAY_JOB_ID=$(./LaunchCVSearch.sh \
    --n_trials "$N_TRIALS" \
    --n_folds "$N_FOLDS" \
    --split_seed "$SPLIT_SEED" \
    ${PROJECT_NAME:+--project_name "$PROJECT_NAME"} \
    ${DEV_MODE:+--dev_mode})

log_message "CV Search submitted with job ID: $ARRAY_JOB_ID"

#Aggregate results
AGG_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$ARRAY_JOB_ID \
    --export=ALL,SPLIT_SEED="$SPLIT_SEED",PROJECT_NAME="$PROJECT_NAME" \
    AggregateTrials.sh)

log_message "Aggregation job submitted with job ID: $AGG_JOB_ID"

#Train final model on full data and validate on hold out test set
TRAIN_FINAL_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$AGG_JOB_ID \
    --export=ALL,SPLIT_SEED="$SPLIT_SEED",PROJECT_NAME="$PROJECT_NAME" \
    TrainDynaSurvCausalOnline.sh)

log_message "Final model training submitted with job ID: $TRAIN_FINAL_JOB_ID"
