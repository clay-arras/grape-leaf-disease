#!/bin/bash

#SBATCH --account=st-sielmann-1-gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --job-name=cosine_similarity
#SBATCH --mail-type=ALL
#SBATCH --mail-user=astrollin.neil@gmail.com
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=oCosineSim.txt
#SBATCH --error=eCosineSim.txt
#SBATCH --time=2:00:00

module load gcc python miniconda3 cuda cudnn

source ~/.bashrc

cd /scratch/st-sielmann-1/agrobot/grape-ld/data/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

~/miniconda3/envs/grape-ld/bin/python run.py
