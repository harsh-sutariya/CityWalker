#!/bin/bash

#SBATCH --job-name=frodobots_2k_download          # Job name
#SBATCH --output=/scratch/hs5580/citywalker/logs/frodobots_download_%A_%a.out   # Standard output and error log
#SBATCH --error=/scratch/hs5580/citywalker/logs/frodobots_download_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH --mem=4G                          # Total memory
#SBATCH --time=8:00:00                    # Time limit hrs:min:sec (large files need more time)
#SBATCH --array=0-23                       # Array range for parts 0-23 (24 total parts)

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Define variables
PART_ID=${SLURM_ARRAY_TASK_ID}
BASE_URL="https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com"
FILENAME="output_rides_${PART_ID}.zip"
DOWNLOAD_URL="${BASE_URL}/${FILENAME}"

# Define download directory
DOWNLOAD_DIR="/vast/hs5580/data/frodobots_2k"
OUTPUT_FILE="${DOWNLOAD_DIR}/${FILENAME}"

# Create logs directory if not exists
mkdir -p /scratch/hs5580/citywalker/logs

# Create download directory if not exists
mkdir -p $DOWNLOAD_DIR

echo "Starting download of part ${PART_ID}"
echo "URL: ${DOWNLOAD_URL}"
echo "Output: ${OUTPUT_FILE}"

# Change to download directory
cd $DOWNLOAD_DIR

# Download with wget (with resume capability in case of interruptions)
wget -c -t 5 -T 300 --progress=bar:force "${DOWNLOAD_URL}" -O "${OUTPUT_FILE}"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Successfully downloaded part ${PART_ID}: ${FILENAME}"
    
    # Verify file size (optional)
    FILE_SIZE=$(stat -c%s "${OUTPUT_FILE}")
    echo "Downloaded file size: ${FILE_SIZE} bytes"
    
    # Create a success marker file
    touch "${OUTPUT_FILE}.complete"
    
else
    echo "Failed to download part ${PART_ID}: ${FILENAME}"
    
    # Remove incomplete file if it exists
    rm -f "${OUTPUT_FILE}"
    
    exit 1
fi

echo "Download job for part ${PART_ID} completed successfully"
