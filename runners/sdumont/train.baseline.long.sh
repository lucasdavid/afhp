#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_long
#SBATCH -J pbn_train_baseline
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/painting-by-numbers/logs/train-baseline-long-%j.out


echo "[train.baseline.long.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

CODE_DIR=$SCRATCH/painting-by-numbers/code
BUILD_DIR=$SCRATCH/painting-by-numbers/build

cd $CODE_DIR

# Optimizers
## SGD Momentum Nesterov
PATCHES=20 BATCH=64 PERFORM_T=true  EPOCHS=100 OPT=momentum LR=0.01  INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.0001  INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=64 PERFORM_T=false EPOCHS=100 OPT=momentum LR=0.01  INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.001   INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=64 PERFORM_T=false EPOCHS=100 OPT=momentum LR=0.01  INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.01    INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py

## Adam Optimizer
PATCHES=20 BATCH=64 PERFORM_T=true  EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.00001 INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=64 PERFORM_T=false EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.0001  INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=64 PERFORM_T=false EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.001   INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py

# Batch sizes
PATCHES=20 BATCH=32 PERFORM_T=true  EPOCHS=100 OPT=momentum LR=0.01  INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.01    INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=32 PERFORM_T=true  EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.0001  INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py

PATCHES=20 BATCH=96 PERFORM_T=true  EPOCHS=100 OPT=momentum LR=0.01  INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.01    INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=96 PERFORM_T=true  EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.0001  INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
