#!/bin/bash

N_TRIALS=1
PROJECT_NAME=""

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
    -h|--help)
      echo "Usage: ./launch [options]"
      echo "Options:"
      echo "  -t, --n_trials      Number of trials (default: 1)"
      echo "  -f, --n_folds       Number of folds (required)"
      echo "  -s, --split_seed    Split seed (required)"
      echo "  -p, --project_name  Project name"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

#Arguments parsing
if [[ -z "$N_FOLDS" || -z "$SPLIT_SEED" ]]; then
	echo "Usage: sbatch script.sh [n_trials] [n_folds] [split_seed] [project_name]"
	echo "Error: Both n_folds and split_seed are required arguments"
	exit 1
fi

if [[ -z "$PROJECT_NAME" ]]; then 
	echo "Warning: unspecified project name, logging disabled" >&2
fi

sbatch --parsable --export=ALL,N_FOLDS=$N_FOLDS,SPLIT_SEED=$SPLIT_SEED,PROJECT_NAME=$PROJECT_NAME --array=0-$((N_TRIALS-1)) DynaSurvCausalOnlineCVSearch.sh
