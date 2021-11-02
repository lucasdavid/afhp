#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p nvidia_long
#SBATCH -J pbn-supcon
#SBATCH -o /scratch/lerdl/lucas.david/afhp/experiments/logs/%j-pbn-supconmh.out
#SBATCH --time=48:00:00
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
BACKBONE=InceptionV3
PATCHES_TRAIN=2   # 2 for each sample, ensuring two samples with the class in a batch, necessary for SupCon.
PATCHES_INFER=20  # patches extracted for each sample during inference.
PATCH_SIZE=299
FEATURES_PARTS=4

BATCH=128      # True batch = BATCH * PATCHES_TRAIN = 32 * 2 = 64
BATCH_INFER=2  # True batch = BATCH * PATCHES_INFER = 2 * 20 = 40

EPOCHS_HE=5
EPOCHS_FT=100
LR_HE=0.05
LR_FT=0.01

GPUS=all
WORKERS=8

MIXED_PRECISION=false  # true
OVERRIDE=false

USE_DOCKER=true

function build_arguments() {
  echo "--data_split $DATA_SPLIT --patch_size $PATCH_SIZE --batch_size $BATCH --patches_train $PATCHES_TRAIN \
    --batch_size_infer $BATCH_INFER --patches_infer $PATCHES_INFER --features_parts $FEATURES_PARTS \
    --backbone_train_workers $WORKERS --override $OVERRIDE --mixed_precision $MIXED_PRECISION \
    --backbone_model $STRATEGY --backbone_architecture $BACKBONE \
    --backbone_train_epochs_head     $EPOCHS_HE --backbone_train_lr    $LR_HE \
    --backbone_train_epochs_finetune $EPOCHS_FT --backbone_finetune_lr $LR_FT \
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

FEATURES_PARTS=1

DATA_SPLIT=frequent  # frequent, original
EPOCHS_HE=0
EPOCHS_FT=0
# run_docker

EPOCHS_HE=5
EPOCHS_FT=100
run_local
# run_docker

# STRATEGY=supcon
# EPOCHS_HE=0
# EPOCHS_FT=100
# run_docker

# STRATEGY=supcon_mh
# EPOCHS_HE=0
# EPOCHS_FT=100
# run_docker

# DATA_SPLIT=original
# EPOCHS_HE=5
# EPOCHS_FT=100
# run_docker

# STRATEGY=supcon
# EPOCHS_HE=0
# EPOCHS_FT=100
# run_docker

# STRATEGY=supcon_mh
# EPOCHS_HE=0
# EPOCHS_FT=100
# run_docker
