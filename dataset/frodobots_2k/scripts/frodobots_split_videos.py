#!/usr/bin/env python3
"""
Frodobots 2K Dataset Video Splitter (Space-Efficient Version)
This script splits the converted MP4 videos from the Frodobots dataset into 2-minute segments,
similar to the CityWalk video processing pipeline.

Key Features:
- Splits videos into 2-minute segments using FFmpeg
- Verifies all segments were created successfully
- Automatically deletes original videos after successful verification 
- Space-efficient processing to avoid storage issues
- Parallel processing support via SLURM array jobs
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import time

# Configuration
SEGMENT_DURATION = 120  # Duration of each segment in seconds (2 minutes)

def ensure_directories(output_dir):
    """Ensure that output directory exists."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory '{output_dir}'.")

def verify_video_file(video_path):
    """
    Verify that a video file is valid using ffprobe.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        bool: True if valid, False otherwise
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,duration',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, check=True, timeout=10)
        output = result.stdout.strip()
        return len(output) > 0 and ',' in output
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

def verify_split_segments(video_path, output_dir, expected_segments):
    """
    Verify that all expected split segments were created successfully.
    
    Args:
        video_path (Path): Original video file path
        output_dir (Path): Directory containing split segments
        expected_segments (list): List of expected segment file paths
        
    Returns:
        tuple: (success: bool, details: str)
    """
    if not output_dir.exists():
        return False, f"Output directory {output_dir} does not exist"
    
    missing_segments = []
    invalid_segments = []
    
    for segment_path in expected_segments:
        segment_file = Path(segment_path)
        
        # Check if segment file exists
        if not segment_file.exists():
            missing_segments.append(segment_file.name)
            continue
        
        # Check if segment file has reasonable size (at least 1KB)
        if segment_file.stat().st_size < 1024:
            invalid_segments.append(f"{segment_file.name} (too small)")
            continue
        
        # Verify segment is a valid video file
        if not verify_video_file(segment_file):
            invalid_segments.append(f"{segment_file.name} (invalid video)")
    
    if missing_segments or invalid_segments:
        issues = []
        if missing_segments:
            issues.append(f"Missing: {', '.join(missing_segments)}")
        if invalid_segments:
            issues.append(f"Invalid: {', '.join(invalid_segments)}")
        return False, "; ".join(issues)
    
    return True, f"All {len(expected_segments)} segments verified successfully"

def safe_delete_video(video_path):
    """
    Safely delete original video file after successful splitting.
    
    Args:
        video_path (str): Path to the video file to delete
        
    Returns:
        bool: True if deletion succeeded, False otherwise
    """
    try:
        video_file = Path(video_path)
        if video_file.exists():
            file_size = video_file.stat().st_size
            os.remove(video_path)
            print(f"🗑️  Deleted original video: {video_file.name} ({file_size // (1024*1024)}MB)")
            return True
        else:
            print(f"⚠️  Original video not found: {video_file.name}")
            return False
    except OSError as e:
        print(f"❌ Failed to delete {video_path}: {e}")
        return False

def get_video_duration(input_path):
    """Get the total duration of the video in seconds using ffprobe."""
    cmd_duration = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', str(input_path)
    ]
    try:
        result = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        total_duration = float(result.stdout.strip())
        return total_duration
    except subprocess.CalledProcessError as e:
        print(f"Error getting duration of '{input_path}': {e.stderr}")
        return None

def find_converted_videos(base_path):
    """
    Find all converted MP4 videos in the Frodobots dataset structure.
    
    Args:
        base_path (str): Base path to search
    
    Returns:
        list: List of video file paths
    """
    video_files = []
    base_path = Path(base_path)
    
    # Pattern: part_*/output_rides_*/ride_*/recordings_converted/*.mp4
    for mp4_file in base_path.glob("part_*/output_rides_*/ride_*/recordings_converted/*.mp4"):
        video_files.append(str(mp4_file))
    
    return video_files

def split_video_completely(video_path, output_dir, base_name):
    """
    Split a complete video into all its segments.
    
    Args:
        video_path (Path): Path to the video file
        output_dir (Path): Directory to save segments
        base_name (str): Base name for segment files
        
    Returns:
        tuple: (success: bool, segment_paths: list, message: str)
    """
    # Get video duration
    total_duration = get_video_duration(video_path)
    if total_duration is None:
        return False, [], f"Could not determine duration of {video_path.name}"
    
    # Skip very short videos
    if total_duration < 10:  # Less than 10 seconds
        return False, [], f"Video too short ({total_duration:.1f}s): {video_path.name}"
    
    # Calculate number of segments
    num_segments = int(total_duration // SEGMENT_DURATION) + (1 if total_duration % SEGMENT_DURATION > 0 else 0)
    
    # Ensure output directory exists
    ensure_directories(output_dir)
    
    # Generate all segments for this video
    segment_paths = []
    failed_segments = []
    
    print(f"  📹 Splitting {video_path.name} into {num_segments} segments...")
    
    for segment_idx in range(num_segments):
        start_time = segment_idx * SEGMENT_DURATION
        segment_name = f"{base_name}_seg_{segment_idx:04d}.mp4"
        segment_path = output_dir / segment_name
        segment_paths.append(str(segment_path))
        
        # Skip if segment already exists and is valid
        if segment_path.exists() and verify_video_file(segment_path):
            continue
        
        # Split this segment
        cmd_split = [
            'ffmpeg',
            '-y',                 # Overwrite without asking
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(SEGMENT_DURATION),
            '-c:v', 'libx264',    # Use libx264 codec
            '-an',                # Discard audio
            str(segment_path)
        ]
        
        try:
            subprocess.run(cmd_split, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            failed_segments.append(f"seg_{segment_idx:04d}")
            print(f"    ❌ Failed to create segment {segment_idx}: {e}")
    
    if failed_segments:
        return False, segment_paths, f"Failed segments: {', '.join(failed_segments)}"
    
    return True, segment_paths, f"Successfully created {num_segments} segments"

def process_frodobots_videos(base_path, task_id=0, num_tasks=1):
    """
    Process converted MP4 videos for splitting into segments with space-efficient cleanup.
    
    Args:
        base_path (str): Base path to the extracted dataset
        task_id (int): Current task ID for parallel processing
        num_tasks (int): Total number of tasks
    """
    print(f"🔍 Scanning for converted MP4 files in {base_path}...")
    video_files = find_converted_videos(base_path)
    
    if not video_files:
        print("❌ No converted MP4 video files found!")
        print("💡 Hint: Run the HLS to MP4 conversion script first")
        return
    
    print(f"📹 Found {len(video_files)} converted MP4 videos")
    
    # Distribute videos among tasks (not segments)
    total_videos = len(video_files)
    videos_per_task = total_videos // num_tasks
    remainder = total_videos % num_tasks
    
    # Compute the start and end indices for this task
    if task_id < remainder:
        start_idx = task_id * (videos_per_task + 1)
        end_idx = start_idx + videos_per_task + 1
    else:
        start_idx = task_id * videos_per_task + remainder
        end_idx = start_idx + videos_per_task
    
    # Get the videos for this task
    task_videos = video_files[start_idx:end_idx]
    
    if not task_videos:
        print(f"No videos assigned to task {task_id}.")
        return
    
    print(f"🚀 Task {task_id}: Processing {len(task_videos)} videos out of {total_videos} total videos.")
    
    # Process videos one by one with verification and cleanup
    successful_videos = 0
    failed_videos = 0
    deleted_videos = 0
    space_saved = 0
    
    with tqdm(total=len(task_videos), desc=f"Task {task_id}", ncols=100) as pbar:
        for video_path_str in task_videos:
            video_path = Path(video_path_str)
            pbar.set_description(f"Task {task_id}: {video_path.name}")
            
            # Create output directory for this video's segments
            # Structure: part_X/output_rides_X/ride_X/recordings_converted_2min/
            output_dir = video_path.parent.parent / "recordings_converted_2min"
            base_name = video_path.stem  # filename without extension
            
            try:
                # Step 1: Split the video completely
                success, segment_paths, message = split_video_completely(video_path, output_dir, base_name)
                
                if not success:
                    print(f"  ❌ Splitting failed: {message}")
                    failed_videos += 1
                    pbar.update(1)
                    continue
                
                # Step 2: Verify all segments were created successfully
                verification_success, verification_details = verify_split_segments(
                    video_path, output_dir, segment_paths
                )
                
                if not verification_success:
                    print(f"  ❌ Verification failed: {verification_details}")
                    failed_videos += 1
                    pbar.update(1)
                    continue
                
                # Step 3: Delete original video after successful verification
                original_size = video_path.stat().st_size
                deletion_success = safe_delete_video(video_path)
                
                if deletion_success:
                    deleted_videos += 1
                    space_saved += original_size
                    successful_videos += 1
                    print(f"  ✅ {message}")
                else:
                    print(f"  ⚠️  Splitting successful but deletion failed for {video_path.name}")
                    successful_videos += 1
                
            except Exception as e:
                print(f"  ❌ Unexpected error processing {video_path.name}: {e}")
                failed_videos += 1
            
            pbar.update(1)
    
    # Final summary
    print(f"\n✅ Task {task_id} completed:")
    print(f"   📹 Videos processed: {len(task_videos)}")
    print(f"   ✅ Successful: {successful_videos}")
    print(f"   ❌ Failed: {failed_videos}")
    print(f"   🗑️  Videos deleted: {deleted_videos}")
    print(f"   💾 Space saved: {space_saved // (1024*1024*1024):.1f} GB")

def main():
    parser = argparse.ArgumentParser(description="Split Frodobots converted videos into 2-minute segments")
    parser.add_argument('--base-path', type=str, 
                       default="/vast/hs5580/data/frodobots_2k/extracted",
                       help="Base path to extracted Frodobots dataset")
    parser.add_argument('--task-id', type=int, default=None,
                       help="Task ID for parallel processing")
    parser.add_argument('--num-tasks', type=int, default=None,
                       help="Total number of parallel tasks")
    
    args = parser.parse_args()
    
    # Get task_id and num_tasks from SLURM environment or arguments
    if args.task_id is not None and args.num_tasks is not None:
        task_id = args.task_id
        num_tasks = args.num_tasks
    else:
        # Try to get from environment variables
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
        num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', '1'))
    
    # Adjust task_id to be zero-based
    task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', '0'))
    task_id = task_id - task_min
    
    print("🤖 Frodobots Video Splitter")
    print("=" * 50)
    print(f"Base Path: {args.base_path}")
    print(f"Segment Duration: {SEGMENT_DURATION} seconds")
    print(f"Task: {task_id + 1}/{num_tasks}")
    print("=" * 50)
    
    process_frodobots_videos(args.base_path, task_id, num_tasks)

if __name__ == "__main__":
    main()
