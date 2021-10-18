#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_long
#SBATCH -J ml_train_multi_gpu
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/logs/cityscapes-resnet101-ce-evaluate-slurm-%j.out

nodeset -e $SLURM_JOB_NODELIST

module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

cd ./experiments/

# KUR=false LR=0.1 LRF=0.01 EXTRA=false python3.9 multilabel_cityscapes_resnet101_evaluate.py
# KUR=true  LR=1.0 LRF=1.0  EXTRA=false python3.9 multilabel_cityscapes_resnet101_evaluate.py
KUR=false LR=0.1 LRF=0.01 EXTRA=true python3.9 multilabel_cityscapes_resnet101_evaluate.py
KUR=true  LR=1.0 LRF=1.0  EXTRA=true python3.9 multilabel_cityscapes_resnet101_evaluate.py
