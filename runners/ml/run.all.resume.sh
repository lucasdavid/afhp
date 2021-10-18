#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_long
#SBATCH -J ml_train_multi_gpu
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/logs/cityscapes/resnet101-ce-slurm-%j.out

nodeset -e $SLURM_JOB_NODELIST

module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

cd ./experiments/

KUR=false T_PERFORM=false LR=0.1 FT_INITIAL_EPOCH=57 LRF=0.01 python3.9 multilabel_cityscapes_resnet101.py
KUR=true  T_PERFORM=false LR=1.0 FT_INITIAL_EPOCH=80 LRF=1.0  python3.9 multilabel_cityscapes_resnet101.py
