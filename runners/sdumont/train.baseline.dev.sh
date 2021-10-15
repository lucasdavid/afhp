#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_dev
#SBATCH -J pbn_train_baseline_dev
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/painting-by-numbers/logs/train-baseline-dev-%j.out


echo "[train.baseline.dev.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

CODE_DIR=$SCRATCH/painting-by-numbers/code
BUILD_DIR=$SCRATCH/painting-by-numbers/build

cd $CODE_DIR

OPT=momentum PATCHES=5 BATCH=64                                                                               \
  PERFORM_T=true  EPOCHS=2    OPT=momentum    LR=0.1     INITIAL_EPOCH=0     TRAIN_STEPS=2    VALID_STEPS=2    \
  PERFORM_FT=true EPOCHS_FT=2 OPT_FT=momentum LR_FT=0.001 INITIAL_EPOCH_FT=2 TRAIN_STEPS_FT=2 VALID_STEPS_FT=2 \
  python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
