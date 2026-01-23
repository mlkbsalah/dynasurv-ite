#!/bin/bash
module purge

module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.8.0/gcc-13.2.0
module load vim/8.1.2141/gcc-9.2.0

source activate pytorch_env

PROJECT_DIR="/workdir/bensalama/DynaSurv"
cd "$PROJECT_DIR/scripts"
echo $PWD

export PYTHONPATH="$PROJECT_DIR/src"
