#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J pbn-supcon
#SBATCH -o /scratch/lerdl/lucas.david/afhp/experiments/logs/%j-pbn-supconmh.out
#SBATCH --time=24:00:00
#SBATCH --exclusive


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
BACKBONE=InceptionV3  # ResNet50

PATCHES_TRAIN=2   # 2 for each sample, ensuring two samples with the class in a batch, necessary for SupCon.
PATCHES_INFER=20  # patches extracted for each sample during inference.
PATCH_SIZE=299
FEATURES_PARTS=4

BATCH_TRAIN=128      # True batch = BATCH_TRAIN * PATCHES_TRAIN = 32 * 2 = 64
BATCH_TEST=2  # True batch = BATCH_TRAIN * PATCHES_INFER = 2 * 20 = 40

EPOCHS_HE=5
EPOCHS_FT=100
LR_HE=0.05
LR_FT=0.01

EPOCHS_AF=100
LR_AF=0.00001

GPUS=all
WORKERS=8

MIXED_PRECISION=false  # true
OVERRIDE=false

USE_DOCKER=true

function build_arguments() {
  echo "--override $OVERRIDE --mixed_precision $MIXED_PRECISION \
    --data_split $DATA_SPLIT --patch_size $PATCH_SIZE --batch_size $BATCH_TRAIN --batch_test $BATCH_TEST \
    --patches_train $PATCHES_TRAIN --patches_test $PATCHES_INFER \
    --features_parts $FEATURES_PARTS --backbone_train_workers $WORKERS \
    --strategy $STRATEGY --backbone_architecture $BACKBONE \
    --backbone_train_epochs    $EPOCHS_HE --backbone_train_lr    $LR_HE \
    --backbone_finetune_epochs $EPOCHS_FT --backbone_finetune_lr $LR_FT \
    --afhp_epochs              $EPOCHS_AF --afhp_lr              $LR_AF \
    --step_preprocess_reduce true \
    --step_features_train false \
    --step_features_infer false \
    --step_afhp_train true";
}

function run_local() {
  $PY run.py `build_arguments`
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

  STRATEGY=supcon
  EPOCHS_HE=0
  EPOCHS_FT=100
  run_local

  STRATEGY=supcon_mh
  EPOCHS_HE=0
  EPOCHS_FT=100
  run_local
}

PATCHES_INFER=4  # patches extracted for each sample during inference.

# Frequent Split (Most frequent painters are used to train the feature extractor).
DATA_SPLIT=frequent run_all_local

# # Original Split (Painter by Numbers Competition)
# DATA_SPLIT=original run_all_local
