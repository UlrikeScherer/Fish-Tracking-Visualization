#!/bin/bash -l

#SBATCH --job-name=tsne_clustering
#SBATCH --output=res_%j.txt     # output file
#SBATCH --error=res_%j.err      # error file
#SBATCH --ntasks=1
#SBATCH --time=0-02:00 
#SBATCH --mem-per-cpu=10000      # memory in MB per cpu allocated
#SBATCH --partition=ex_scioi_gpu  # partition to submit to
#SBATCH --gres=gpu:1

#module load nvidia/cuda/10.0    # load required modules (depends upon your code which modules are needed)
#module load comp/gcc/7.2.0

#source ./venv/bin/activate      # activate your python environment
#source ~/.condarc
#source ~/miniconda/etc/profile.d/conda.sh
conda activate rapids-22.04
echo "JOB START"
#python hpc_clustering.py 200 6 4
python hpc_clustering.py 20 6 4
echo "JOB DONE"
conda deactivate 
