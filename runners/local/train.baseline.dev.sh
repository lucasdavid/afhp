#!/bin/bash

echo "[train.baseline.dev.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

export CODE_DIR=/tf/code
export BUILD_DIR=/tf/build
export SCRATCH=/tf
export DATA_DIR=/tf/data/painter-by-numbers
export LOGS_DIR=/tf/logs
export WEIGHTS_DIR=/tf/logs/models

cd $CODE_DIR


OPT=momentum PATCHES=5 BATCH=128                                                                               \
  PERFORM_T=true  EPOCHS=2    OPT=momentum    LR=0.1     INITIAL_EPOCH=0     TRAIN_STEPS=2    VALID_STEPS=2    \
  PERFORM_FT=true EPOCHS_FT=2 OPT_FT=momentum LR_FT=0.001 INITIAL_EPOCH_FT=2 TRAIN_STEPS_FT=2 VALID_STEPS_FT=2 \
  python3 -X pycache_prefix=$BUILD_DIR baseline.py
