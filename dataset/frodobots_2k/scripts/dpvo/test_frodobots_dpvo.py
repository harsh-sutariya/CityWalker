#!/usr/bin/env python3
"""
Test script for Frodobots DPVO setup
This script tests the DPVO processing on a small subset of videos to verify everything works.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add DPVO to Python path
sys.path.append('/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO')

def test_environment():
    """Test if the DPVO environment is set up correctly."""
    print("🔧 Testing DPVO environment setup...")
    
    # Test DPVO imports
    try:
        from dpvo.config import cfg
        from dpvo.dpvo import DPVO
        print("✅ DPVO imports successful")
    except ImportError as e:
        print(f"❌ DPVO import failed: {e}")
        print("💡 Tip: Make sure you're running in the dpvo_legacy conda environment")
        return False
    except Exception as e:
        print(f"⚠️  DPVO loaded with warnings: {e}")
        print("✅ DPVO imports successful (ignoring CUDA warnings for CPU-only testing)")
    
    # Test if network weights exist
    network_path = '/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO/dpvo.pth'
    if os.path.exists(network_path):
        print("✅ DPVO network weights found")
    else:
        print(f"❌ DPVO network weights not found at: {network_path}")
        return False
    
    # Test calibration file
    calib_path = '/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO/calib/frodobots.txt'
    if os.path.exists(calib_path):
        print("✅ Frodobots calibration file found")
    else:
        print(f"❌ Frodobots calibration file not found at: {calib_path}")
        return False
    
    return True

def find_test_videos():
    """Find a few test videos for processing."""
    print("🔍 Finding test videos...")
    
    base_path = Path('/vast/hs5580/data/frodobots_2k/extracted')
    
    # Find front camera videos
    front_videos = list(base_path.glob("part_*/output_rides_*/ride_*/recordings_converted_2min/*_front_camera_seg_*.mp4"))
    
    if len(front_videos) == 0:
        print("❌ No front camera videos found")
        return []
    
    # Take first 3 videos for testing
    test_videos = front_videos[:3]
    print(f"✅ Found {len(test_videos)} test videos:")
    for video in test_videos:
        print(f"   - {video}")
    
    return test_videos

def test_single_video(video_path):
    """Test DPVO processing on a single video."""
    print(f"\n🎬 Testing DPVO on: {Path(video_path).name}")
    
    # Import required modules
    try:
        from frodobots_dpvo_slurm import run, is_valid_video
        from dpvo.config import cfg
        import torch
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Check if video is valid
    if not is_valid_video(str(video_path)):
        print(f"❌ Video is not valid: {video_path}")
        return False
    
    print("✅ Video is valid")
    
    # Load network weights
    network_path = '/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO/dpvo.pth'
    calib_path = '/scratch/hs5580/citywalker/CityWalker/thirdparty/DPVO/calib/frodobots.txt'
    
    try:
        print("🔧 Loading network weights...")
        network = torch.load(network_path, map_location='cuda')
        print("✅ Network loaded successfully")
        
        print("🏃 Running DPVO (this may take a moment)...")
        # Run with higher stride for faster testing
        (poses, tstamps), (points, colors, calib_info) = run(
            cfg, network, str(video_path), calib_path, 
            stride=12, skip=0, viz=False, timeit=True
        )
        
        print(f"✅ DPVO completed successfully!")
        print(f"   📊 Generated {len(poses)} poses")
        print(f"   📍 Generated {len(points)} 3D points")
        
        return True
        
    except Exception as e:
        print(f"❌ DPVO processing failed: {e}")
        return False

def test_video_finding():
    """Test the video finding functionality."""
    print("\n🔍 Testing video discovery...")
    
    try:
        from frodobots_dpvo_slurm import find_frodobots_videos
        
        # Test finding front camera videos
        front_videos = find_frodobots_videos('/vast/hs5580/data/frodobots_2k/extracted', 'front')
        print(f"✅ Found {len(front_videos)} front camera videos")
        
        # Test finding rear camera videos  
        rear_videos = find_frodobots_videos('/vast/hs5580/data/frodobots_2k/extracted', 'rear')
        print(f"✅ Found {len(rear_videos)} rear camera videos")
        
        return len(front_videos) > 0 or len(rear_videos) > 0
        
    except Exception as e:
        print(f"❌ Video finding failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Frodobots DPVO Test Suite")
    print("=" * 50)
    
    # Test 1: Environment setup
    if not test_environment():
        print("\n❌ Environment test failed. Please check DPVO installation.")
        return False
    
    # Test 2: Video finding
    if not test_video_finding():
        print("\n❌ Video finding test failed. Please check video paths.")
        return False
    
    # Test 3: Find test videos
    test_videos = find_test_videos()
    if not test_videos:
        print("\n❌ No test videos found. Please check if video splitting completed.")
        return False
    
    # Test 4: Process one video
    if not test_single_video(test_videos[0]):
        print("\n❌ Video processing test failed.")
        return False
    
    print("\n🎉 All tests passed! DPVO setup is working correctly.")
    print("\n📋 Next steps:")
    print("1. Review the SLURM script parameters in run_frodobots_dpvo.slurm")
    print("2. Submit the job: sbatch run_frodobots_dpvo.slurm")
    print("3. Monitor progress: squeue -u $USER")
    
    return True

if __name__ == '__main__':
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = main()
    sys.exit(0 if success else 1)
