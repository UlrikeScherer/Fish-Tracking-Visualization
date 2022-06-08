#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=res_%j.txt     # output file
#SBATCH --error=res_%j.err      # error file
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000      # memory in MB per cpu allocated
#SBATCH --partition=ex_scioi_gpu  # partition to submit to
#SBATCH --gres=gpu:1

#module load nvidia/cuda/10.0    # load required modules (depends upon your code which modules are needed)
#module load comp/gcc/7.2.0

#source ./venv/bin/activate      # activate your python environment
conda activate rapids-22.04

python src/hpc_clustering.py trace_size=200 n_clusters=7

deactivate 