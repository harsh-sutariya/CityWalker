#!/bin/bash

#SBATCH --job-name=frodobots_hls_convert    # Job name
#SBATCH --output=/scratch/hs5580/citywalker/logs/frodobots_convert_%A_%a.out   # Standard output and error log
#SBATCH --error=/scratch/hs5580/citywalker/logs/frodobots_convert_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --mem=16G                         # Memory for video processing
#SBATCH --time=6:00:00                    # Time limit hrs:min:sec
#SBATCH --array=0-23                       # Array range for 24 parts (0-23)

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Define variables
PART_ID=${SLURM_ARRAY_TASK_ID}

REPO=/scratch/hs5580/citywalker/CityWalker
SCRIPTS_DIR=/scratch/hs5580/citywalker/CityWalker/dataset/frodobots_2k/scripts
OVERLAY=/scratch/hs5580/singularity/citywalker.ext3
TEMP_OVERLAY=/tmp/temp_overlay_${SLURM_JOB_ID}.ext3
CONDA_IMAGE=/scratch/work/public/singularity/anaconda3-2024.06-1.sqf
OS_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

# Define paths
BASE_PATH="/vast/hs5580/data/frodobots_2k/extracted"

echo "Starting HLS to MP4 conversion for part ${PART_ID}"
echo "Base path: ${BASE_PATH}"

# Create temporary overlay
singularity overlay create --size 1024 $TEMP_OVERLAY

# Create logs directory if not exists
mkdir -p /scratch/hs5580/citywalker/logs

singularity exec --nv \
  --overlay $OVERLAY:ro \
  --overlay $TEMP_OVERLAY:rw \
  --overlay $CONDA_IMAGE:ro \
  $OS_IMAGE /bin/bash -c "
        source /ext3/env.sh
        conda activate citywalker
        cd $SCRIPTS_DIR
        
        echo 'Converting HLS streams to MP4 for part ${PART_ID}...'
        python frodobots_hls_to_mp4.py --base-path ${BASE_PATH}
        
        echo 'HLS to MP4 conversion completed for part ${PART_ID}'
        "

# Clean up
rm -f $TEMP_OVERLAY

echo "HLS to MP4 conversion job for part ${PART_ID} completed"
