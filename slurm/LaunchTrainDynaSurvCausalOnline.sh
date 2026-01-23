#!/bin/bash

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    -s|--split_seed)
      SPLIT_SEED="$2"
      shift 2
      ;;
    -dev|--dev_mode)
      DEV_MODE=1
      shift 1
      ;;
    -h|--help)
      echo "Usage: ./launch [options]"
      echo "Options:"
      echo "  -s, --split_seed      Split seed (required)"
      echo "  -dev, --dev_mode      Enable fast development run mode"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

#Arguments parsing
if [[ -z "$SPLIT_SEED" ]]; then
	echo "Usage: sbatch script.sh [split_seed]"
	echo "Error: split_seed is a required argument"
	exit 1
fi

sbatch --parsable --export=ALL,SPLIT_SEED=$SPLIT_SEED,DEV_MODE=$DEV_MODE TrainDynaSurvCausalOnline.sh