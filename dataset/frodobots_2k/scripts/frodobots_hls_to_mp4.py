#!/usr/bin/env python3
"""
Frodobots 2K Dataset HLS to MP4 Converter
This script converts HLS video streams (.m3u8 + .ts segments) to MP4 format
for easier processing and splitting.

Features:
- Converts HLS playlists to MP4 using ffmpeg
- Verifies MP4 integrity after conversion
- Automatically cleans up original HLS files (.m3u8 and .ts) after successful conversion
- Preserves storage space by removing redundant files
- Parallel processing support via SLURM array jobs
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import glob
import re

def verify_mp4_file(mp4_path):
    """
    Verify that the MP4 file is valid using ffprobe.
    
    Args:
        mp4_path (str): Path to the MP4 file to verify
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,duration',
        '-of', 'csv=p=0',
        str(mp4_path)
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, check=True, timeout=30)
        # Check if we got codec info (indicates valid video)
        output = result.stdout.strip()
        return len(output) > 0 and ',' in output  # Should have codec,duration
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

def get_ts_files_from_m3u8(m3u8_path):
    """
    Parse M3U8 file to extract list of .ts segment files.
    
    Args:
        m3u8_path (str): Path to the .m3u8 playlist file
        
    Returns:
        list: List of absolute paths to .ts files referenced in the playlist
    """
    ts_files = []
    m3u8_dir = Path(m3u8_path).parent
    
    try:
        with open(m3u8_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Handle both relative and absolute paths
                    if line.endswith('.ts'):
                        ts_path = m3u8_dir / line if not os.path.isabs(line) else Path(line)
                        if ts_path.exists():
                            ts_files.append(str(ts_path))
    except (IOError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse M3U8 file {m3u8_path}: {e}")
    
    return ts_files

def cleanup_hls_files(m3u8_path, output_path):
    """
    Delete original HLS files after successful conversion.
    
    Args:
        m3u8_path (str): Path to the original .m3u8 file
        output_path (str): Path to the converted MP4 file
        
    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    try:
        # First verify the MP4 is valid
        if not verify_mp4_file(output_path):
            print(f"❌ MP4 verification failed for {output_path}, keeping original files")
            return False
        
        # Get list of .ts files to delete
        ts_files = get_ts_files_from_m3u8(m3u8_path)
        
        files_deleted = 0
        files_failed = 0
        
        # Delete .ts files
        for ts_file in ts_files:
            try:
                os.remove(ts_file)
                files_deleted += 1
            except OSError as e:
                print(f"Warning: Could not delete {ts_file}: {e}")
                files_failed += 1
        
        # Delete .m3u8 file
        try:
            os.remove(m3u8_path)
            files_deleted += 1
        except OSError as e:
            print(f"Warning: Could not delete {m3u8_path}: {e}")
            files_failed += 1
        
        if files_failed > 0:
            print(f"⚠️  Deleted {files_deleted} files, failed to delete {files_failed} files")
        else:
            print(f"🗑️  Successfully deleted {files_deleted} HLS files")
        
        return files_failed == 0
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False

def convert_hls_to_mp4(m3u8_path, output_path):
    """
    Convert HLS stream to MP4 using ffmpeg, then verify and cleanup original files.
    
    Args:
        m3u8_path (str): Path to the .m3u8 playlist file
        output_path (str): Path for the output MP4 file
        
    Returns:
        bool: True if conversion and cleanup succeeded, False otherwise
    """
    if os.path.exists(output_path):
        print(f"Skipping {output_path} as it already exists.")
        # If MP4 exists, check if we should still cleanup HLS files
        if os.path.exists(m3u8_path):
            print(f"🔍 Checking if HLS cleanup needed for {m3u8_path}...")
            cleanup_hls_files(m3u8_path, output_path)
        return True
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-y',                    # Overwrite without asking
        '-i', str(m3u8_path),    # Input HLS playlist
        '-c:v', 'copy',          # Copy video stream without re-encoding (faster)
        '-an',                   # Remove audio (consistent with split script)
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, 
                              text=True, check=True)
        
        # Verify the output file was created and has reasonable size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:  # At least 1KB
            print(f"✅ Conversion successful: {os.path.basename(output_path)}")
            
            # Now cleanup original HLS files
            cleanup_success = cleanup_hls_files(m3u8_path, output_path)
            if not cleanup_success:
                print(f"⚠️  Conversion succeeded but cleanup had issues for {m3u8_path}")
            
            return True
        else:
            print(f"❌ Output file {output_path} was not created or is too small")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {m3u8_path}: {e.stderr}")
        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def find_hls_files(base_path, part_id=None):
    """
    Find all HLS video files in the Frodobots dataset structure.
    
    Args:
        base_path (str): Base path to search (e.g., /vast/hs5580/data/frodobots_2k/extracted)
        part_id (int): Specific part to process (0-23), or None for all parts
    
    Returns:
        list: List of tuples (m3u8_path, ride_info)
    """
    hls_files = []
    base_path = Path(base_path)
    
    if part_id is not None:
        # Process only specific part for efficiency
        part_pattern = f"part_{part_id}/output_rides_{part_id}/ride_*/recordings/*.m3u8"
        search_pattern = base_path / f"part_{part_id}" / f"output_rides_{part_id}" / "ride_*" / "recordings" / "*.m3u8"
        glob_pattern = f"part_{part_id}/output_rides_{part_id}/ride_*/recordings/*.m3u8"
    else:
        # Process all parts (for single-task mode)
        glob_pattern = "part_*/output_rides_*/ride_*/recordings/*.m3u8"
    
    for m3u8_file in base_path.glob(glob_pattern):
        # Extract ride information from path
        parts = m3u8_file.parts
        part_name = next(p for p in parts if p.startswith('part_'))
        ride_name = next(p for p in parts if p.startswith('ride_'))
        
        # Only process video streams (not audio)
        if 'video' in m3u8_file.name:
            hls_files.append((str(m3u8_file), {
                'part': part_name,
                'ride': ride_name,
                'stream_id': m3u8_file.stem,
                'recordings_dir': m3u8_file.parent
            }))
    
    return hls_files

def process_ride_videos(base_path, task_id=0, num_tasks=1):
    """
    Process HLS videos for conversion to MP4.
    
    Args:
        base_path (str): Base path to the extracted dataset
        task_id (int): Current task ID for parallel processing
        num_tasks (int): Total number of tasks
    """
    # Efficient approach: Each task processes only its corresponding part
    if num_tasks == 24:
        # Standard mode: task_id corresponds to part_id (0-23)
        part_id = task_id
        print(f"🔍 Scanning for HLS files in part_{part_id}...")
        hls_files = find_hls_files(base_path, part_id=part_id)
        task_files = hls_files  # Process all files in this part
        
        if not hls_files:
            print(f"❌ No HLS video files found in part_{part_id}!")
            return
        
        print(f"📹 Found {len(hls_files)} HLS video streams in part_{part_id}")
        print(f"🚀 Task {task_id}: Processing all {len(task_files)} files from part_{part_id}")
        
    else:
        # Fallback mode: distribute all files among arbitrary number of tasks
        print(f"🔍 Scanning for HLS files in {base_path}...")
        hls_files = find_hls_files(base_path)
        
        if not hls_files:
            print("❌ No HLS video files found!")
            return
        
        print(f"📹 Found {len(hls_files)} HLS video streams")
        
        # Distribute work among tasks
        total_files = len(hls_files)
        files_per_task = total_files // num_tasks
        remainder = total_files % num_tasks
        
        # Calculate start and end indices for this task
        if task_id < remainder:
            start_idx = task_id * (files_per_task + 1)
            end_idx = start_idx + files_per_task + 1
        else:
            start_idx = task_id * files_per_task + remainder
            end_idx = start_idx + files_per_task
        
        task_files = hls_files[start_idx:end_idx]
        
        if not task_files:
            print(f"No files assigned to task {task_id}")
            return
        
        print(f"🚀 Task {task_id}: Processing {len(task_files)} files")
    
    # Process each HLS file
    successful_conversions = 0
    failed_conversions = 0
    total_files_cleaned = 0
    
    with tqdm(total=len(task_files), desc=f"Task {task_id} Converting", ncols=100) as pbar:
        for m3u8_path, ride_info in task_files:
            # Create output path: recordings_converted/{stream_id}.mp4
            recordings_dir = Path(ride_info['recordings_dir'])
            converted_dir = recordings_dir.parent / "recordings_converted"
            
            # Extract camera info from stream ID for cleaner naming
            stream_id = ride_info['stream_id']
            if 'uid_s_1000' in stream_id:
                camera_name = "front_camera"
            elif 'uid_s_1001' in stream_id:
                camera_name = "rear_camera"  
            else:
                camera_name = "camera"
            
            output_filename = f"{ride_info['ride']}_{camera_name}.mp4"
            output_path = converted_dir / output_filename
            
            pbar.set_description(f"Task {task_id}: {ride_info['part']}/{ride_info['ride']} ({camera_name})")
            
            # Count original files before conversion
            ts_files_before = get_ts_files_from_m3u8(m3u8_path)
            original_file_count = len(ts_files_before) + 1  # +1 for .m3u8 file
            
            success = convert_hls_to_mp4(m3u8_path, output_path)
            if success:
                successful_conversions += 1
                # Check if files were actually cleaned up
                if not os.path.exists(m3u8_path):
                    total_files_cleaned += original_file_count
            else:
                failed_conversions += 1
            
            pbar.update(1)
    
    print(f"✅ Task {task_id} completed:")
    print(f"   📹 Conversions: {successful_conversions} successful, {failed_conversions} failed")
    print(f"   🗑️  Original files cleaned: {total_files_cleaned}")

def main():
    parser = argparse.ArgumentParser(description="Convert Frodobots HLS videos to MP4")
    parser.add_argument('--base-path', type=str, 
                       default="/vast/hs5580/data/frodobots_2k/extracted",
                       help="Base path to extracted Frodobots dataset")
    parser.add_argument('--task-id', type=int, default=0,
                       help="Task ID for parallel processing")
    parser.add_argument('--num-tasks', type=int, default=1,
                       help="Total number of parallel tasks")
    
    args = parser.parse_args()
    
    # Try to get SLURM environment variables if not provided
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', '0'))
        task_id = task_id - task_min
    else:
        task_id = args.task_id
    
    if 'SLURM_ARRAY_TASK_COUNT' in os.environ:
        num_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    else:
        num_tasks = args.num_tasks
    
    print("🤖 Frodobots HLS to MP4 Converter")
    print("=" * 50)
    print(f"Base Path: {args.base_path}")
    print(f"Task: {task_id + 1}/{num_tasks}")
    print("=" * 50)
    
    process_ride_videos(args.base_path, task_id, num_tasks)

if __name__ == "__main__":
    main()
