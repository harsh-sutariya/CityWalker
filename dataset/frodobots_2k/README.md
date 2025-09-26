# FrodoBots 2K Dataset Processing Tools

This directory contains all utilities for processing the FrodoBots 2K dataset, from download to final 2-minute video segments.

## Directory Structure

```
frodobots_2k/
├── scripts/           # Processing scripts and SLURM jobs
├── docs/             # Dataset documentation and examples  
├── config/           # Configuration files
└── README.md         # This file
```

## Processing Pipeline

The complete processing pipeline consists of these steps:

### 1. Download Dataset
```bash
cd scripts/
sbatch download_frodobots_2k.slurm
```

### 2. Extract Archives
```bash
sbatch unzip_frodobots_2k.slurm
```

### 3. Convert HLS to MP4
```bash
sbatch convert_frodobots_hls.slurm
```

### 4. Split into 2-minute Segments
```bash
sbatch split_frodobots_videos.slurm
```

## Scripts Overview

### SLURM Job Scripts
- `download_frodobots_2k.slurm` - Downloads the dataset from S3
- `unzip_frodobots_2k.slurm` - Extracts zip archives and cleans up
- `convert_frodobots_hls.slurm` - Converts HLS streams to MP4 format
- `split_frodobots_videos.slurm` - Splits videos into 2-minute segments

### Python Processing Scripts
- `frodobots_hls_to_mp4.py` - HLS to MP4 conversion with verification and cleanup
- `frodobots_split_videos.py` - Space-efficient video splitting with verification

### Documentation
- `docs/helpercode.ipynb` - Dataset structure and usage examples

## Dataset Information

- **Total Size**: ~1TB after processing
- **Video Segments**: 51,100 total (2 minutes each)
- **Format**: MP4 video segments
- **Structure**: Organized by ride sessions with sensor data
- **Cameras**: Front and rear camera streams (separate processing)
- **Additional Data**: GPS, IMU, control data, audio timestamps

## Features

- **Space-efficient processing**: Automatically deletes source files after verification
- **Parallel processing**: Uses SLURM array jobs for scalability
- **Verification**: Ensures data integrity throughout the pipeline
- **Resume capability**: Can restart interrupted jobs safely
