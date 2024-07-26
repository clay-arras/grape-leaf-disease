#!/bin/bash

#SBATCH --account=st-sielmann-1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=grape-ld
#SBATCH --mail-type=ALL
#SBATCH --mail-user=astrollin.neil@gmail.com
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=0:30:00

module load gcc python
module load http_proxy

source ~/.bashrc

cd /scratch/st-sielmann-1/agrobot/grape-ld/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

~/miniconda3/envs/grape-ld/bin/python __main__.py
