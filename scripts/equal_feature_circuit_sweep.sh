#!/bin/bash

# Loop over example numbers 0 through 4 to submit jobs
for i in {1..4}; do
    sbatch << EOF
#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=100GB  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 2:00:00  # Job time limit
#SBATCH --constraint=a100  # Constraint to use A100 GPU
#SBATCH --output=out/slurm-%j-example$i.out  # Output file
#SBATCH --error=out/slurm-%j-example$i.err   # Error file

module load miniconda/22.11.1-1
conda activate finetuning
python generic_equal_feature_patching.py --example_number $i
EOF
done
