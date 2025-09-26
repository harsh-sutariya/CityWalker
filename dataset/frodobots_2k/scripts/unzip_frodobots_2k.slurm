#!/bin/bash

# Script to unzip FrodoBots 2K dataset parts
# Refer to ../docs/helpercode.ipynb for dataset structure and usage examples
# This script will extract zip files and delete them after successful verification

#SBATCH --job-name=frodobots_2k_unzip          # Job name
#SBATCH --output=/scratch/hs5580/citywalker/logs/frodobots_unzip_%A_%a.out   # Standard output and error log
#SBATCH --error=/scratch/hs5580/citywalker/logs/frodobots_unzip_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task (unzip can use multiple cores)
#SBATCH --mem=16G                         # More memory for large file operations
#SBATCH --time=6:00:00                    # Time limit hrs:min:sec (large files need time)
#SBATCH --array=0-23                       # Array range for parts 0-23 (24 total parts)

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Define variables
PART_ID=${SLURM_ARRAY_TASK_ID}
FILENAME="output_rides_${PART_ID}.zip"

# Define directories
DOWNLOAD_DIR="/vast/hs5580/data/frodobots_2k"
EXTRACT_DIR="/vast/hs5580/data/frodobots_2k/extracted"
SOURCE_FILE="${DOWNLOAD_DIR}/${FILENAME}"
COMPLETE_MARKER="${DOWNLOAD_DIR}/${FILENAME}.complete"
EXTRACT_COMPLETE="${EXTRACT_DIR}/${FILENAME}.extracted"

# Create logs directory if not exists
mkdir -p /scratch/hs5580/citywalker/logs

# Create extraction directory if not exists
mkdir -p $EXTRACT_DIR

echo "Starting extraction of part ${PART_ID}"
echo "Source file: ${SOURCE_FILE}"
echo "Extract to: ${EXTRACT_DIR}"

# Check if source file exists and download was completed
if [ ! -f "$SOURCE_FILE" ]; then
    echo "ERROR: Source file $SOURCE_FILE does not exist"
    exit 1
fi

if [ ! -f "$COMPLETE_MARKER" ]; then
    echo "WARNING: Download marker $COMPLETE_MARKER not found. Proceeding anyway..."
fi

# Check if already extracted
if [ -f "$EXTRACT_COMPLETE" ]; then
    echo "Part ${PART_ID} already extracted successfully. Skipping."
    exit 0
fi

# Get file size for progress tracking
FILE_SIZE=$(stat -c%s "$SOURCE_FILE")
echo "Source file size: ${FILE_SIZE} bytes ($(echo "scale=1; $FILE_SIZE / 1024^3" | bc)GB)"

# Start extraction
echo "Starting unzip operation at $(date)"
cd $EXTRACT_DIR

# Use unzip with progress and create directory for this part
PART_DIR="part_${PART_ID}"
mkdir -p $PART_DIR
cd $PART_DIR

# Extract with verbose output and overwrite protection
unzip -o "$SOURCE_FILE" 2>&1

# Check extraction status
UNZIP_EXIT_CODE=$?

if [ $UNZIP_EXIT_CODE -eq 0 ]; then
    echo "Successfully extracted part ${PART_ID}: ${FILENAME}"
    
    # Count extracted files
    EXTRACTED_FILES=$(find . -type f | wc -l)
    echo "Number of extracted files: ${EXTRACTED_FILES}"
    
    # Get total extracted size
    EXTRACTED_SIZE=$(du -sb . | cut -f1)
    echo "Total extracted size: ${EXTRACTED_SIZE} bytes ($(echo "scale=1; $EXTRACTED_SIZE / 1024^3" | bc)GB)"
    
    # Create extraction completion marker
    touch "$EXTRACT_COMPLETE"
    echo "Created completion marker: $EXTRACT_COMPLETE"
    
    # Verify extraction was successful by checking if we have files and the marker exists
    if [ $EXTRACTED_FILES -gt 0 ] && [ -f "$EXTRACT_COMPLETE" ]; then
        echo "Extraction verified successfully. Proceeding to delete source zip file."
        
        # Delete the source zip file to save space
        echo "Deleting source file: ${SOURCE_FILE}"
        rm "$SOURCE_FILE"
        
        if [ $? -eq 0 ]; then
            echo "Successfully deleted source zip file: ${FILENAME}"
            
            # Also remove the download completion marker since zip is deleted
            if [ -f "$COMPLETE_MARKER" ]; then
                rm "$COMPLETE_MARKER"
                echo "Removed download completion marker: $COMPLETE_MARKER"
            fi
        else
            echo "ERROR: Failed to delete source zip file: ${SOURCE_FILE}"
            exit 1
        fi
    else
        echo "ERROR: Extraction verification failed. Not deleting source file."
        exit 1
    fi
    
    echo "Extraction completed successfully at $(date)"
    
else
    echo "ERROR: Failed to extract part ${PART_ID}: ${FILENAME}"
    echo "Unzip exit code: $UNZIP_EXIT_CODE"
    
    # Clean up any partial extraction
    cd $EXTRACT_DIR
    rm -rf $PART_DIR
    
    exit $UNZIP_EXIT_CODE
fi

echo "Unzip job for part ${PART_ID} completed successfully"
