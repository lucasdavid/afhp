#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J pbn-supcon
#SBATCH -o /scratch/lerdl/lucas.david/afhp/experiments/logs/%j-pbn-ce.out
#SBATCH --time=48:00:00
# SBATCH --exclusive


echo "[train.baseline.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."


if [[ "`hostname`" == "sdumont"* ]]; then
  nodeset -e $SLURM_JOB_NODELIST

  echo "Loading modules gcc/7.4 cudnn/8.1_cuda-11.2"
  module load gcc/7.4 cudnn/8.1_cuda-11.2

  ENV=sdumont
  WORK_DIR=$SCRATCH/afhp
  DATA_DIR=$SCRATCH/datasets
  BUILD_DIR=$WORK_DIR/build

  PIP=pip
  PY=python

  echo "Loading virtual env $SCRATCH/envs/tf"
  source $SCRATCH/envs/tf/bin/activate
else
  ENV=local
  WORK_DIR=/home/ldavid/workspace/repos/research/afhp
  DATA_DIR=/home/ldavid/workspace/datasets
  BUILD_DIR=$WORK_DIR/build

  PIP=pip
  PY=python
fi

cd $WORK_DIR

DATA_SPLIT=frequent  # frequent, original
STRATEGY=ce    # ce, supcon, supcon_mh
BACKBONE=ResNet50V2  # ResNet50
BACKBONE_FEATURES_LAYER=avg_pool

PATCHES_TRAIN=2   # 2 patches for each sample, ensuring two samples with the class in a batch, necessary for SupCon.
POSITIVES_TRAIN=2 # 2 paintings for each painter, ensuring two samples of the same class in a batch, necessary for SupCon.
PATCHES_INFER=20  # patches extracted for each sample during inference.
PATCH_SIZE=224
FEATURES_PARTS=4

BATCH_TRAIN=128   # True batch = BATCH_TRAIN * PATCHES_TRAIN * POSITIVES_TRAIN = 32 * 2 * 2 = 128
BATCH_TEST=2      # True batch = BATCH_TRAIN * PATCHES_INFER = 2 * 20 = 40

EPOCHS_HE=0
EPOCHS_FT=100
LR_HE=0.01
LR_FT=0.01

EPOCHS_AF=20
LR_AF=0.00001

WORKERS=8
DEVICES=0,1  # 0,1
GPUS=all

MIXED_PRECISION=false  # true
JIT_COMPILE=false
OVERRIDE=false

function build_arguments() {
  echo "--override $OVERRIDE --mixed_precision $MIXED_PRECISION --jit_compile $JIT_COMPILE \
    --data_split $DATA_SPLIT --patch_size $PATCH_SIZE --batch_size $BATCH_TRAIN --batch_test $BATCH_TEST \
    --patches_train $PATCHES_TRAIN --patches_test $PATCHES_INFER --positives_train $POSITIVES_TRAIN \
    --features_parts $FEATURES_PARTS --backbone_train_workers $WORKERS \
    --strategy $STRATEGY --backbone_architecture $BACKBONE --backbone_features_layer $BACKBONE_FEATURES_LAYER \
    --backbone_train_epochs    $EPOCHS_HE --backbone_train_lr    $LR_HE \
    --backbone_finetune_epochs $EPOCHS_FT --backbone_finetune_lr $LR_FT \
    --afhp_epochs $EPOCHS_AF --afhp_lr $LR_AF \
    --step_preprocess_reduce false \
    --step_features_train true --step_features_infer true \
    --step_afhp_train true --step_afhp_test true";
}

function run_local() {
  echo "==========================================================="
  echo "[Run] $BACKBONE $STRATEGY"
  echo "==========================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES $PY run.py `build_arguments`
}

function run_docker() {

  docker rm -f $(docker ps -aq)
  docker build -t tensorflow/tensorflow:2.9.3-personal .

  docker run \
    -v $(pwd):/workdir -v $DATA_DIR:$DATA_DIR \
    --rm --gpus $GPUS \
    tensorflow/tensorflow:2.9.3-personal \
    $PY run.py `build_arguments`
}

function run_all_local() {
  STRATEGY=ce
  EPOCHS_HE=0
  EPOCHS_FT=0
  # run_local

  EPOCHS_HE=5
  EPOCHS_FT=100
  # run_local

  BACKBONE_FEATURES_LAYER=painter_proj1
  EPOCHS_HE=5
  EPOCHS_FT=100
  
  STRATEGY=supcon
  # run_local

  STRATEGY=supcon_mh
  run_local
}

# Frequent Split (Most frequent painters are used to train the feature extractor).
DATA_SPLIT=frequent run_all_local

# # Original Split (Painter by Numbers Competition)
# DATA_SPLIT=original run_all_local
