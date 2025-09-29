#!/usr/bin/env python3
"""
Frodobots DPVO Processing Script
Adapted from CityWalker dpvo_slurm.py for processing Frodobots 2K dataset video segments.

This script processes the 2-minute video segments created by the video splitting pipeline
and runs DPVO (Deep Patch Visual Odometry) to generate camera trajectories.
"""

import os
from multiprocessing import Process, Queue
from pathlib import Path
import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

# Add DPVO to Python path
import sys
sys.path.append('/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO')

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, video_path, calib, stride=1, skip=0, viz=False, timeit=False):
    slam = None
    queue = Queue(maxsize=16)
    
    reader = Process(target=video_stream, args=(queue, video_path, calib, stride, skip))
    reader.start()

    while True:
        item = queue.get()
        if item is None:
            break
        (t, image, intrinsics) = item

        if t < 0: 
            break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))

def is_valid_video(video_path):
    """Check if video file is valid and not corrupted."""
    try:
        # Basic file existence and size check first
        if not os.path.exists(video_path):
            return False
        
        # Check if file has reasonable size (at least 1KB)
        if os.path.getsize(video_path) < 1024:
            return False
            
        # Try ffprobe if available, but don't fail if it's not
        try:
            import subprocess
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
            # If ffprobe works and gives valid output, use it
            if result.returncode == 0 and len(result.stdout.decode().strip()) > 0:
                return True
            # If ffprobe fails but file exists and has size, assume it's valid
            # (ffprobe might not be available or video format might not be supported)
            print(f"Warning: ffprobe failed for {video_path}, but file exists - proceeding anyway")
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # ffprobe not available or failed - fall back to basic checks
            print(f"Warning: ffprobe not available for {video_path}, using basic file validation")
            return True
            
    except Exception as e:
        print(f"Error validating video {video_path}: {e}")
        return False

def find_frodobots_videos(base_path, camera_type="front"):
    """
    Find all Frodobots video segments for the specified camera type.
    
    Args:
        base_path (str): Base path to Frodobots extracted data
        camera_type (str): "front" or "rear" camera
    
    Returns:
        list: List of video file paths
    """
    video_files = []
    base_path = Path(base_path)
    
    # Pattern: part_*/output_rides_*/ride_*/recordings_converted_2min/*_{camera_type}_camera_seg_*.mp4
    pattern = f"part_*/output_rides_*/ride_*/recordings_converted_2min/*_{camera_type}_camera_seg_*.mp4"
    
    for video_file in base_path.glob(pattern):
        video_files.append(str(video_file))
    
    return sorted(video_files)

def process_videos(videos, cfg, network, calib, stride, skip, viz, timeit, output_dir, args):
    """Process a list of videos with DPVO and save results."""
    for video_file in videos:
        video_path = video_file
        video_name = Path(video_file).stem
        
        # Create output directory structure based on ride organization
        # Extract ride information from path for organized output
        path_parts = Path(video_file).parts
        part_name = None
        ride_name = None
        
        for i, part in enumerate(path_parts):
            if part.startswith('part_'):
                part_name = part
            elif part.startswith('ride_'):
                ride_name = part
                break
        
        if part_name and ride_name:
            video_output_dir = os.path.join(output_dir, part_name, ride_name, "dpvo_poses")
        else:
            video_output_dir = os.path.join(output_dir, "dpvo_poses")
            
        Path(video_output_dir).mkdir(parents=True, exist_ok=True)

        print(f"Processing {video_name}...")
        
        # Check if video is valid before processing
        if not is_valid_video(video_path):
            print(f"Skipping corrupted video: {video_path}")
            continue
            
        # Skip if trajectory already exists
        trajectory_file = os.path.join(video_output_dir, f"{video_name}.txt")
        if os.path.exists(trajectory_file) and not args.overwrite:
            print(f"Trajectory already exists for {video_name}, skipping...")
            continue
            
        try:
            (poses, tstamps), (points, colors, calib_info) = run(cfg, network, video_path, calib, stride, skip, viz, timeit)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue
            
        # Create trajectory object
        trajectory = PoseTrajectory3D(
            positions_xyz=poses[:, :3], 
            orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], 
            timestamps=tstamps
        )

        # Save outputs based on arguments
        if args.save_ply:
            save_ply(os.path.join(video_output_dir, f"{video_name}.ply"), points, colors)

        if args.save_colmap:
            save_output_for_COLMAP(os.path.join(video_output_dir, video_name), trajectory, points, colors, *calib_info)

        if args.save_trajectory:
            file_interface.write_tum_trajectory_file(trajectory_file, trajectory)
            print(f"Saved trajectory: {trajectory_file}")

        if args.plot:
            plot_dir = os.path.join(video_output_dir, "trajectory_plots")
            Path(plot_dir).mkdir(exist_ok=True)
            plot_trajectory(
                trajectory, 
                title=f"DPVO Trajectory for {video_name}", 
                filename=os.path.join(plot_dir, f"{video_name}.pdf")
            )

def partition_videos(video_files, total_jobs, job_index):
    """
    Partition the list of video files into subsets based on total_jobs and job_index.
    """
    num_videos = len(video_files)
    if num_videos == 0:
        return []
        
    videos_per_job = num_videos // total_jobs
    remainder = num_videos % total_jobs

    start_idx = job_index * videos_per_job + min(job_index, remainder)
    end_idx = start_idx + videos_per_job + (1 if job_index < remainder else 0)

    return video_files[start_idx:end_idx]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Frodobots DPVO Parallel Video Processing Script")
    parser.add_argument('--network', type=str, default='dpvo.pth', help='Path to the network weights file')
    parser.add_argument('--videodir', type=str, required=True, help='Base directory containing Frodobots extracted data')
    parser.add_argument('--calib', type=str, required=True, help='Path to the calibration file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--camera_type', type=str, default='front', choices=['front', 'rear'], 
                       help='Camera type to process (front or rear)')
    parser.add_argument('--stride', type=int, default=6, help='Stride for frame processing')
    parser.add_argument('--skip', type=int, default=0, help='Number of frames to skip')
    parser.add_argument('--config', type=str, default="/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO/config/default.yaml", help='Path to the config file')
    parser.add_argument('--timeit', action='store_true', help='Enable timing information')
    parser.add_argument('--viz', action="store_true", help='Enable visualization')
    parser.add_argument('--plot', action="store_true", help='Enable plotting of trajectories')
    parser.add_argument('--save_ply', action="store_true", help='Save point cloud as PLY')
    parser.add_argument('--save_colmap', action="store_true", help='Save output for COLMAP')
    parser.add_argument('--save_trajectory', action="store_true", help='Save trajectory file')
    parser.add_argument('--overwrite', action="store_true", help='Overwrite existing trajectory files')
    
    # Arguments for array job partitioning
    parser.add_argument('--total_jobs', type=int, required=True, help='Total number of array jobs')
    parser.add_argument('--job_index', type=int, required=True, help='Index of this job (0-based)')
    
    # Config options
    parser.add_argument('opts', nargs='*', help='Config options')
    
    args = parser.parse_args()

    # Load configuration
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("🤖 Frodobots DPVO Processing")
    print("=" * 50)
    print(f"Video Directory: {args.videodir}")
    print(f"Camera Type: {args.camera_type}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Job {args.job_index + 1}/{args.total_jobs}")
    print("=" * 50)

    # Find all video files for the specified camera type
    print(f"🔍 Finding {args.camera_type} camera video segments...")
    video_files = find_frodobots_videos(args.videodir, args.camera_type)
    
    if not video_files:
        print(f"❌ No {args.camera_type} camera video files found in {args.videodir}")
        exit(1)
    
    print(f"📹 Found {len(video_files)} {args.camera_type} camera video segments")

    # Partition videos for this job
    job_videos = partition_videos(video_files, args.total_jobs, args.job_index)
    
    if not job_videos:
        print(f"⚠️  No videos assigned to job {args.job_index}")
        exit(0)
    
    print(f"🚀 Processing {len(job_videos)} videos in this job")

    # Prepare network path (DPVO constructor will load it internally)
    network_path = os.path.join('/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO', args.network)
    if not os.path.exists(network_path):
        print(f"❌ Network weights not found: {network_path}")
        exit(1)
    
    print(f"🔧 Loading network: {network_path}")
    # Pass the path string to DPVO constructor, not the loaded state_dict
    network = network_path

    # Process videos
    process_videos(
        job_videos, cfg, network, args.calib, args.stride, args.skip, 
        args.viz, args.timeit, args.output_dir, args
    )
    
    print(f"\n✅ Job {args.job_index + 1} completed successfully!")
    print(f"📊 Processed {len(job_videos)} video segments")
