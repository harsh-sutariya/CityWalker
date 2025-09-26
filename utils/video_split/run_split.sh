#!/bin/bash
#SBATCH --job-name=split          # Job name
#SBATCH --output=/scratch/hs5580/citywalker/logs/split_%A_%a.out   # Standard output and error log
#SBATCH --error=/scratch/hs5580/citywalker/logs/split_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task
#SBATCH --mem=8G                          # Total memory
#SBATCH --time=4:00:00                    # Time limit hrs:min:sec
#SBATCH --array=0-499                       # Array range (e.g., 100 jobs)

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

REPO=/scratch/hs5580/citywalker/CityWalker
OVERLAY=/scratch/hs5580/singularity/citywalker.ext3
TEMP_OVERLAY=/tmp/temp_overlay_${SLURM_JOB_ID}.ext3
CONDA_IMAGE=/scratch/work/public/singularity/anaconda3-2024.06-1.sqf
OS_IMAGE=/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif

singularity overlay create --size 1024 $TEMP_OVERLAY

# Create logs directory if not exists
mkdir -p /scratch/hs5580/citywalker/logs

singularity exec \
  --overlay $OVERLAY:ro \
  --overlay $TEMP_OVERLAY:rw \
  --overlay $CONDA_IMAGE:ro \
  $OS_IMAGE /bin/bash -c "
        source /ext3/env.sh
        conda activate citywalker
        cd $REPO/utils/video_split
        python split_slurm.py
        "

rm -f $TEMP_OVERLAY
