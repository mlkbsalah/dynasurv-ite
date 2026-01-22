#!/bin/bash

N_TRIALS=1
N_FOLDS=2
SPLIT_SEED=0
PROJECT_NAME=""
DEV_MODE=0

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--n_trials)
      N_TRIALS="$2"
      shift 2
      ;;
    -f|--n_folds)
      N_FOLDS="$2"
      shift 2
      ;;
    -s|--split_seed)
      SPLIT_SEED="$2"
      shift 2
      ;;
    -p|--project_name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    -dev|--dev_mode)
      DEV_MODE=1
      shift 1
      ;;
    -h|--help)
      echo "Usage: ./launch [options]"
      echo "Options:"
      echo "  -t, --n_trials      Number of trials (default: 1)"
      echo "  -f, --n_folds       Number of folds (required if not in dev mode)"
      echo "  -s, --split_seed    Split seed (required if not in dev mode)"
      echo "  -p, --project_name  Project name"
      echo "  -dev, --dev_mode    Enable fast development run mode"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

#Arguments parsing
if [[ "$DEV_MODE" -eq 0 ]] && { [[ -z "$N_FOLDS" ]] || [[ -z "$SPLIT_SEED" ]]; }; then
	echo "Usage: sbatch script.sh [n_trials] [n_folds] [split_seed] [project_name]"
	echo "Error: Both n_folds and split_seed are required arguments"
	exit 1
fi

if [[ -z "$PROJECT_NAME" ]]; then 
	echo "Warning: unspecified project name, logging disabled" >&2
fi

if [[ "$DEV_MODE" -eq 1 ]]; then
    N_TRIALS=2
    N_FOLDS=2
    echo "Development mode enabled: overriding n_trials to $N_TRIALS, n_folds to $N_FOLDS, and training epochs to 5"
fi

sbatch --parsable --export=ALL,N_FOLDS=$N_FOLDS,SPLIT_SEED=$SPLIT_SEED,PROJECT_NAME=$PROJECT_NAME,DEV_MODE=$DEV_MODE --array=0-$((N_TRIALS-1)) DynaSurvCausalOnlineCVSearch.sh
