# Frodobots DPVO Processing

This directory contains scripts for running DPVO (Deep Patch Visual Odometry) on the Frodobots 2K dataset video segments.

## Overview

DPVO processes video sequences to estimate camera trajectories and generate 3D point clouds. This implementation is adapted from the CityWalker DPVO pipeline to work with the Frodobots dataset structure.

## Files

- **`frodobots_dpvo_slurm.py`** - Main Python script for parallel DPVO processing
- **`run_frodobots_dpvo.slurm`** - SLURM batch script for job submission
- **`test_frodobots_dpvo.py`** - Test script to verify setup before running full jobs
- **`submit_frodobots_dpvo.sh`** - Helper script for easy job submission
- **`README.md`** - This documentation

## Prerequisites

1. **Video Segments**: Ensure video splitting has completed successfully
   ```bash
   # Check if video segments exist
   find /vast/hs5580/data/frodobots_2k/extracted -name "*_seg_*.mp4" | wc -l
   ```

2. **DPVO Environment**: The `dpvo_legacy` conda environment should be available in the singularity container

3. **Network Weights**: DPVO network weights should be available at:
   ```
   /scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO/dpvo.pth
   ```

## Usage

### Step 1: Test the Setup

Before running the full processing, test that everything works:

```bash
cd /scratch/hs5580/citywalker/CityWalker/dataset/frodobots_2k/scripts/dpvo
python test_frodobots_dpvo.py
```

This will:
- Verify DPVO environment setup
- Test video discovery functionality  
- Process a sample video to ensure everything works

### Step 2: Submit Processing Jobs

Use the helper script for easy job submission:

```bash
./submit_frodobots_dpvo.sh
```

This will:
- Show available video counts for front/rear cameras
- Let you choose which cameras to process
- Submit appropriate SLURM jobs

**Or manually submit jobs:**

```bash
# Process front camera videos
sbatch --export=CAMERA_TYPE=front run_frodobots_dpvo.slurm

# Process rear camera videos  
sbatch --export=CAMERA_TYPE=rear run_frodobots_dpvo.slurm
```

### Step 3: Monitor Progress

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f /scratch/hs5580/citywalker/logs/frodobots_dpvo_*_0.out

# Check output progress
find /vast/hs5580/data/frodobots_2k/extracted -name "*.txt" -path "*/dpvo_poses/*" | wc -l
```

## Output Structure

DPVO generates trajectory files organized by the original ride structure:

```
/vast/hs5580/data/frodobots_2k/extracted/
├── part_0/
│   └── ride_16446_20240115041752/
│       └── dpvo_poses/
│           ├── ride_16446_20240115041752_front_camera_seg_0000.txt
│           ├── ride_16446_20240115041752_front_camera_seg_0001.txt
│           └── ...
└── ...
```

Each `.txt` file contains the camera trajectory in TUM format:
```
timestamp tx ty tz qx qy qz qw
```

## Configuration

### Camera Calibration

Frodobots camera intrinsics are defined in:
```
/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO/calib/frodobots.txt
```

Current values: `410 410 320 180` (fx, fy, cx, cy)

### Processing Parameters

Key parameters in the SLURM script:
- **Array size**: `0-199` (200 parallel tasks)
- **Stride**: `6` (process every 6th frame)
- **Camera type**: `front` or `rear`
- **Time limit**: `4:00:00` (4 hours)

## Troubleshooting

### Common Issues

1. **"No videos found"**
   - Ensure video splitting completed: check for `*_seg_*.mp4` files
   - Verify paths in the scripts match your data location

2. **"Network weights not found"**
   - Check if DPVO weights exist at the expected path
   - Ensure the singularity environment has access to the weights

3. **CUDA errors**
   - Ensure jobs are submitted to GPU nodes (`--gres=gpu:1`)
   - Check GPU availability: `nvidia-smi`

4. **Import errors**
   - Verify the `dpvo_legacy` conda environment is properly configured
   - Check that all DPVO dependencies are installed

### Performance Tuning

- **Reduce stride** (e.g., from 6 to 12) for faster processing but lower quality
- **Adjust array size** based on the number of video segments
- **Increase time limit** if jobs are timing out

## Expected Runtime

- **Per video segment**: ~2-5 minutes depending on length and stride
- **Total dataset**: 4-8 hours with 200 parallel tasks
- **Output size**: ~1-2 KB per trajectory file

## Next Steps

After DPVO processing completes, the trajectory files can be used for:
- Path reconstruction and visualization
- SLAM evaluation and benchmarking  
- Multi-modal data fusion with GPS/IMU
- Dataset analysis and statistics
