#!/bin/bash
#SBATCH --nodes=1           #Numero de Nós
#SBATCH --ntasks-per-node=1 #Numero de tarefas por Nó
#SBATCH --ntasks=1          #Numero de tarefas
#SBATCH -p nvidia_long      #Fila (partition) a ser utilizada
#SBATCH -J kd_parallel_test #Nome job
#SBATCH --exclusive         #Utilização exclusiva dos nós durante a execução do job

echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

module load gcc/7.4 python/3.8.2 cudnn/8.0_cuda-10.1

mkdir -p $SCRATCH/logs/


cd $SCRATCH/experiments/kd

CUDA_VISIBLE_DEVICES=0 python vanilla.py > $SCRATCH/logs/vanilla.log & 
CUDA_VISIBLE_DEVICES=1 python fitnet.py >  $SCRATCH/logs/fitnet.log & 
