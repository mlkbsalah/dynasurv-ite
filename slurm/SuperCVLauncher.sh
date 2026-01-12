#!/bin/bash

N_TRIALS=10

#Generate a random split seed
SPLIT_SEED=$(( $(date +%s) % 2147483647 ))
echo "Using 32-bit CV split seed: $SPLIT_SEED"

#Submit trials
ARRAY_JOB_ID=$(sbatch --parsable --export=ALL,SPLIT_SEED=$SPLIT_SEED --array=0-$((N_TRIALS-1)) slurm_DynaSurvCausalOnlineCV.sh)
echo "Submitted job array ID: $ARRAY_JOB_ID"

#Aggregate results after all trials complete
AGG_JOB_ID=$(sbatch --dependency=afterok:$ARRAY_JOB_ID --export=ALL,SPLIT_SEED=$SPLIT_SEED aggregate_trials.sh)
echo "Submitted aggregation job ID: $AGG_JOB_ID (will run after all trials finish)"