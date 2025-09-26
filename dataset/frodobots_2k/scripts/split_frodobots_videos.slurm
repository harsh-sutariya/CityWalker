#!/bin/bash

#SBATCH --job-name=frodobots_split         # Job name
#SBATCH --output=/scratch/hs5580/citywalker/logs/frodobots_split_%A_%a.out   # Standard output and error log
#SBATCH --error=/scratch/hs5580/citywalker/logs/frodobots_split_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task
#SBATCH --mem=8G                          # Total memory
#SBATCH --time=6:00:00                    # Time limit hrs:min:sec (increased for large dataset)
#SBATCH --array=0-499                       # Array range for 17,098+ videos (500 tasks)

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

REPO=/scratch/hs5580/citywalker/CityWalker
SCRIPTS_DIR=/scratch/hs5580/citywalker/CityWalker/dataset/frodobots_2k/scripts
OVERLAY=/scratch/hs5580/singularity/citywalker.ext3
TEMP_OVERLAY=/tmp/temp_overlay_${SLURM_JOB_ID}.ext3
CONDA_IMAGE=/scratch/work/public/singularity/anaconda3-2024.06-1.sqf
OS_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

# Define paths
BASE_PATH="/vast/hs5580/data/frodobots_2k/extracted"

echo "Starting space-efficient video splitting for Frodobots dataset"
echo "Base path: ${BASE_PATH}"
echo "Mode: Split videos, verify segments, delete originals to save space"

# Create temporary overlay
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
        cd $SCRIPTS_DIR
        
        echo 'Splitting Frodobots videos into 2-minute segments (space-efficient mode)...'
        python frodobots_split_videos.py --base-path ${BASE_PATH}
        
        echo 'Space-efficient video splitting completed'
        "

# Clean up
rm -f $TEMP_OVERLAY

echo "Video splitting job completed"
