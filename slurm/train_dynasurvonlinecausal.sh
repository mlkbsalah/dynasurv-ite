#!/bin/bash
#SBATCH --job-name=dynasurvonline_training
#SBATCH --output=%x.o%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpua100

module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.8.0/gcc-13.2.0

source activate pytorch_env 

PROJECT_DIR="/workdir/bensalama/DynaSurv"
cd "$PROJECT_DIR/scripts"
echo $PWD

export PYTHONPATH="$PROJECT_DIR/src"

python3 train_DynaSurvCausalOnline_sweep.py
