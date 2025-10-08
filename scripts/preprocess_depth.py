#!/usr/bin/env python3
"""
Preprocess Depth Maps for CityWalker DBR Training

This script uses Depth-Anything-V2 to generate depth maps for all videos
in the dataset. The depth maps are saved as numpy arrays for efficient
loading during training.

Usage:
    python scripts/preprocess_depth.py --config config/citywalk_dbr.yaml --checkpoint <path_to_depth_model>

Example:
    python scripts/preprocess_depth.py \
        --config config/citywalk_dbr.yaml \
        --checkpoint checkpoints/depth_anything_v2_metric_vits.pth \
        --model_size small
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from model.depth_teacher import load_depth_teacher, precompute_depth_for_video


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute depth maps for CityWalker training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to depth model checkpoint')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'],
                       help='Depth model size')
    parser.add_argument('--max_depth', type=float, default=20.0, help='Maximum depth in meters')
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'png16'],
                       help='Format to save depth maps')
    parser.add_argument('--force', action='store_true', help='Overwrite existing depth files')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    
    # Get data directories
    video_dir = cfg['data']['video_dir']
    depth_dir = cfg['data']['depth_dir']
    video_fps = cfg['data']['video_fps']
    target_fps = cfg['data']['target_fps']
    
    # Create depth directory
    os.makedirs(depth_dir, exist_ok=True)
    print(f"Depth maps will be saved to {depth_dir}")
    
    # Load depth teacher model
    print(f"\nLoading Depth-Anything-V2 model ({args.model_size})...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        print("Please download the checkpoint from:")
        print("https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models")
        return
    
    depth_teacher = load_depth_teacher(
        model_size=args.model_size,
        max_depth=args.max_depth,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    print("Model loaded successfully!")
    
    # Get list of video files
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    print(f"\nFound {len(video_files)} videos to process")
    
    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, video_file)
        output_name = video_file.replace('.mp4', '_depth.npy')
        output_path = os.path.join(depth_dir, output_name)
        
        # Skip if already exists and not forcing
        if os.path.exists(output_path) and not args.force:
            skipped += 1
            continue
        
        try:
            # Precompute depth for this video
            precompute_depth_for_video(
                video_path=video_path,
                output_path=output_path,
                depth_teacher=depth_teacher,
                target_fps=target_fps,
                video_fps=video_fps,
                save_format=args.save_format
            )
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {video_file}: {e}")
            failed += 1
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("Depth Preprocessing Summary")
    print("="*60)
    print(f"Total videos:     {len(video_files)}")
    print(f"Processed:        {processed}")
    print(f"Skipped:          {skipped}")
    print(f"Failed:           {failed}")
    print("="*60)
    
    if failed > 0:
        print("\nSome videos failed to process. Check the error messages above.")
    else:
        print("\nAll videos processed successfully!")
        print(f"\nDepth maps saved to: {depth_dir}")
        print("\nYou can now train with DBR using:")
        print(f"    python train.py --config {args.config}")


if __name__ == '__main__':
    main()

