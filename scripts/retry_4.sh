#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=100GB  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 2:00:00  # Job time limit
#SBATCH -o out/slurm-%j-example0.out  # %j = job ID for output log
#SBATCH -e out/slurm-%j-example0.err  # %j = job ID for error log
#SBATCH --constraint=a100  # Constraint to use A100 GPU

module load miniconda/22.11.1-1
conda activate finetuning
python retry_generic-Copy4.py