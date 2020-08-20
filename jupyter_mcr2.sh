#!/bin/bash
#SBATCH --job-name=mcr2_jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --partition=default_gpu

. /home/zw287/anaconda3/etc/profile.d/conda.sh
conda activate mcr2

cd $HOME/mcr2
XDG_RUNTIME_DIR=/tmp/zw287 jupyter-notebook --ip=0.0.0.0 --port=8895
