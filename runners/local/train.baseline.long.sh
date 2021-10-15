#!/bin/bash

echo "[train.baseline.long.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

export CODE_DIR=/tf/code
export BUILD_DIR=/tf/build
export SCRATCH=/tf
export DATA_DIR=/tf/data/painter-by-numbers
export LOGS_DIR=/tf/logs
export WEIGHTS_DIR=/tf/logs/models

cd $CODE_DIR


PATCHES=20 BATCH=64 PERFORM_T=true  EPOCHS=100 OPT=momentum LR=0.1   INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.001   INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=64 PERFORM_T=false EPOCHS=100 OPT=momentum LR=0.1   INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.01    INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=64 PERFORM_T=false EPOCHS=100 OPT=momentum LR=0.1   INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.1     INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py

## Adam Optimizer
PATCHES=20 BATCH=64 PERFORM_T=true  EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.0001  INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=64 PERFORM_T=false EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.00001 INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=64 PERFORM_T=false EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.001   INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py

# Batch sizes
PATCHES=20 BATCH=32 PERFORM_T=true  EPOCHS=100 OPT=momentum LR=0.1   INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.01    INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=32 PERFORM_T=true  EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.0001  INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py

PATCHES=20 BATCH=96 PERFORM_T=true  EPOCHS=100 OPT=momentum LR=0.1   INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=momentum LR_FT=0.01    INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
PATCHES=20 BATCH=96 PERFORM_T=true  EPOCHS=100 OPT=adam     LR=0.001 INITIAL_EPOCH=0 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam     LR_FT=0.0001  INITIAL_EPOCH_FT=100 python3.9 -X pycache_prefix=$BUILD_DIR baseline.py
