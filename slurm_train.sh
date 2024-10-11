#!/bin/bash
#SBATCH --job-name=urbannav          # Job name
#SBATCH --output=logs/urbannav_%j.out   # Standard output and error log
#SBATCH --error=logs/urbannav_%j.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=48                  # Number of CPU cores per task
#SBATCH --mem=256G                          # Total memory
#SBATCH --time=48:00:00                    # Time limit hrs:min:sec
#SBATCH --account=pr_117_tandon_advanced
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinhao.liu@nyu.edu
#SBATCH --gres=gpu:2                       # Number of GPUs per node
#SBTCH --constraint=h100

cd $SCRATCH/UrbanNav

# Define paths
CONFIG=config/slurm_sunny.yaml

# Create logs directory if not exists
mkdir -p logs

singularity exec --nv \
    --overlay $SCRATCH/environments/nav.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh;
    conda activate UrbanNav;
    python train.py --config ${CONFIG}"
