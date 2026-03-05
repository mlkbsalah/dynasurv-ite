#!/bin/bash
module purge

module load anaconda3/2023.09-0/none-none
module load cuda/13.0.2/none-none
module load vim/8.1.2141/gcc-9.2.0

source activate pytorch_env

PROJECT_DIR="/workdir/bensalama/DynaSurv"
cd "$PROJECT_DIR/scripts"
echo $PWD

export PYTHONPATH="$PROJECT_DIR/src"
