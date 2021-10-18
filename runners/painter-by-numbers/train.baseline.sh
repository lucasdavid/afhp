#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_long
#SBATCH -J pbn_train_baseline
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/logs/painter-by-numbers/baseline-%j.out

nodeset -e $SLURM_JOB_NODELIST

module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

cd ./painting-by-numbers/

OPT=momentum PATCHES=20 BATCH=128 PERFORM_T=true  EPOCHS=100 LR=0.1   PERFORM_FT=true EPOCHS_FT=200 LR_FT=0.001 python3.9 baseline.py
OPT=momentum PATCHES=20 BATCH=128 PERFORM_T=false EPOCHS=100 LR=0.1   PERFORM_FT=true EPOCHS_FT=200 LR_FT=0.01  python3.9 baseline.py
OPT=momentum PATCHES=20 BATCH=128 PERFORM_T=false EPOCHS=100 LR=0.1   PERFORM_FT=true EPOCHS_FT=200 LR_FT=0.1   python3.9 baseline.py

OPT=adam     PATCHES=20 BATCH=128 PERFORM_T=true  EPOCHS=100 LR=0.001 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam LR_FT=0.0001 python3.9 baseline.py
OPT=adam     PATCHES=20 BATCH=128 PERFORM_T=false EPOCHS=100 LR=0.001 PERFORM_FT=true EPOCHS_FT=200 OPT_FT=adam LR_FT=0.001  python3.9 baseline.py
