#!/bin/bash
#SBATCH --job-name=frodobots_dpvo          # Job name
#SBATCH --output=/scratch/hs5580/citywalker/logs/frodobots_dpvo_%A_%a.out   # Standard output and error log
#SBATCH --error=/scratch/hs5580/citywalker/logs/frodobots_dpvo_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --mem=8G                          # Total memory
#SBATCH --gres=gpu:1                       # Number of GPUs per node
#SBATCH --partition=rtx8000
#SBATCH --time=4:00:00                    # Time limit hrs:min:sec (increased for large dataset)
#SBATCH --array=0-199                      # Array range (adjust based on number of video segments)

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Define variables
TOTAL_JOBS=200
JOB_INDEX=${SLURM_ARRAY_TASK_ID}

REPO=/scratch/hs5580/citywalker/CityWalker
OVERLAY=/scratch/hs5580/singularity/citywalker.ext3
TEMP_OVERLAY=/tmp/temp_overlay_${SLURM_JOB_ID}.ext3
CONDA_IMAGE=/scratch/work/public/singularity/anaconda3-2024.06-1.sqf
OS_IMAGE=/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif

# Define paths for Frodobots
PYTHON_SCRIPT=frodobots_dpvo_slurm.py
VIDEO_DIR=/vast/hs5580/data/frodobots_2k/extracted
CALIB_FILE=$REPO/thirdparty/DPVO/calib/frodobots.txt
OUTPUT_DIR=/vast/hs5580/data/frodobots_2k/extracted
CAMERA_TYPE=${CAMERA_TYPE:-front} # Default to front camera, can be overridden

echo "Starting Frodobots DPVO processing"
echo "Video directory: ${VIDEO_DIR}"
echo "Camera type: ${CAMERA_TYPE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Calibration file: ${CALIB_FILE}"

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
        conda activate dpvo_legacy
        cd $REPO/dataset/frodobots_2k/scripts/dpvo
        
        echo 'Starting DPVO processing for Frodobots ${CAMERA_TYPE} camera videos...'
        python $PYTHON_SCRIPT \
          --total_jobs $TOTAL_JOBS \
          --job_index $JOB_INDEX \
          --videodir $VIDEO_DIR \
          --calib $CALIB_FILE \
          --output_dir $OUTPUT_DIR \
          --camera_type $CAMERA_TYPE \
          --stride 6 \
          --save_trajectory
        
        echo 'Frodobots DPVO processing completed'
        "

# Clean up
rm -f $TEMP_OVERLAY

echo "Frodobots DPVO job completed"
