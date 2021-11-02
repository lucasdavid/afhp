#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_small
#SBATCH -J afhp-setup
#SBATCH -o /scratch/lerdl/lucas.david/afhp/experiments/logs/%j-setup.out
#SBATCH --time=01:00:00


echo "[train.baseline.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST
module load gcc/7.4 cudnn/8.1_cuda-11.2

CODE_DIR=$SCRATCH/afhp
BUILD_DIR=$SCRATCH/afhp/build

PIP=pip3.9
PY=python3.9

cd $CODE_DIR

$PIP install virtualenv

virtualenv $SCRATCH/envs/tf2.9 --python python3.9

source $SCRATCH/envs/tf2.9/bin/activate
pip install -r requirements.txt

mkdir -p ./experiments/logs ./experiments/weights ./experiments/predictions
