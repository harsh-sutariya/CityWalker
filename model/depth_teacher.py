"""
Depth Teacher Module - Wrapper for Depth-Anything-V2 metric depth estimation.

This module provides utilities for:
1. Loading pre-trained Depth-Anything-V2 models
2. Batch depth inference on video frames
3. Depth map preprocessing and caching
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add Depth-Anything-V2 to path
DEPTH_ANYTHING_PATH = Path(__file__).parent.parent / 'thirdparty' / 'Depth-Anything-V2' / 'metric_depth'
sys.path.insert(0, str(DEPTH_ANYTHING_PATH))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print(f"Warning: Could not import DepthAnythingV2 from {DEPTH_ANYTHING_PATH}")
    DepthAnythingV2 = None


class DepthTeacher(nn.Module):
    """
    Wrapper for Depth-Anything-V2 metric depth estimation.
    
    Args:
        model_size (str): Model size - 'small', 'base', or 'large'
        max_depth (float): Maximum depth in meters
        checkpoint_path (str): Path to model checkpoint
        device (str): Device to run inference on
    """
    
    def __init__(self, model_size='small', max_depth=20.0, checkpoint_path=None, device='cuda'):
        super().__init__()
        
        if DepthAnythingV2 is None:
            raise ImportError("DepthAnythingV2 not available. Check installation.")
        
        self.model_size = model_size
        self.max_depth = max_depth
        self.device = device
        
        # Model configurations
        model_configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        if model_size not in model_configs:
            raise ValueError(f"Invalid model size: {model_size}. Choose from {list(model_configs.keys())}")
        
        config = model_configs[model_size]
        
        # Initialize model
        self.model = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels'],
            max_depth=max_depth
        )
        
        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading depth model checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: No checkpoint provided or file not found at {checkpoint_path}")
            print("Using randomly initialized weights (not recommended for inference)")
        
        self.model.to(device)
        self.model.eval()
        
    @torch.no_grad()
    def forward(self, images):
        """
        Predict depth for batch of images.
        
        Args:
            images: (B, 3, H, W) tensor of RGB images, normalized to [0, 1]
            
        Returns:
            depth: (B, H, W) tensor of depth values in meters
        """
        # Ensure images are on correct device
        images = images.to(self.device)
        
        # Depth-Anything-V2 expects images normalized with ImageNet stats
        # If images are in [0, 1], apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_normalized = (images - mean) / std
        
        # Forward pass
        depth = self.model(images_normalized)
        
        return depth
    
    @torch.no_grad()
    def infer_batch_with_padding(self, images):
        """
        Infer depth for a batch of images, handling size constraints.
        
        Args:
            images: (B, 3, H, W) tensor of images
            
        Returns:
            depth: (B, H, W) tensor of depth maps
        """
        B, C, H, W = images.shape
        
        # Ensure dimensions are multiples of 14
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        
        if pad_h > 0 or pad_w > 0:
            # Pad images to make dimensions multiples of 14
            images_padded = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            images_padded = images
        
        # Forward pass
        depth_padded = self.forward(images_padded)
        
        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            depth = depth_padded[:, :H, :W]
        else:
            depth = depth_padded
        
        return depth
    
    @torch.no_grad()
    def infer_video_frames(self, video_frames, batch_size=8):
        """
        Predict depth for a sequence of video frames.
        
        Args:
            video_frames: (T, 3, H, W) tensor of video frames
            batch_size: Batch size for inference
            
        Returns:
            depth_maps: (T, H, W) tensor of depth maps
        """
        T = video_frames.shape[0]
        depth_maps = []
        
        for i in range(0, T, batch_size):
            batch = video_frames[i:i+batch_size]
            depth_batch = self.forward(batch)
            depth_maps.append(depth_batch.cpu())
        
        depth_maps = torch.cat(depth_maps, dim=0)
        return depth_maps


def load_depth_teacher(model_size='small', max_depth=20.0, checkpoint_path=None, device='cuda'):
    """
    Convenience function to load a depth teacher model.
    
    Args:
        model_size (str): 'small', 'base', or 'large'
        max_depth (float): Maximum depth in meters
        checkpoint_path (str): Path to checkpoint file
        device (str): Device for inference
        
    Returns:
        DepthTeacher instance
    """
    return DepthTeacher(
        model_size=model_size,
        max_depth=max_depth,
        checkpoint_path=checkpoint_path,
        device=device
    )


def precompute_depth_for_video(video_path, output_path, depth_teacher, target_fps=5, 
                                video_fps=30, save_format='npy'):
    """
    Precompute and save depth maps for a video file.
    
    Args:
        video_path (str): Path to input video
        output_path (str): Path to save depth maps
        depth_teacher (DepthTeacher): Depth estimation model
        target_fps (int): Target FPS for depth extraction
        video_fps (int): Original video FPS
        save_format (str): Format to save depth ('npy' or 'png16')
    """
    from decord import VideoReader, cpu
    import cv2
    
    # Load video
    video_reader = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(video_reader)
    
    # Compute frame indices at target FPS
    frame_multiplier = video_fps // target_fps
    frame_indices = list(range(0, num_frames, frame_multiplier))
    
    print(f"Processing {len(frame_indices)} frames from {video_path}")
    
    # Process in batches
    depth_maps = []
    batch_size = 8
    
    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i:i+batch_size]
        frames = video_reader.get_batch(batch_indices).asnumpy()
        
        # Convert to tensor and normalize
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Predict depth with padding to handle size constraints
        depth_batch = depth_teacher.infer_batch_with_padding(frames_tensor)
        depth_maps.append(depth_batch.cpu().numpy())
    
    # Concatenate all batches
    depth_maps = np.concatenate(depth_maps, axis=0)
    
    # Save depth maps
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if save_format == 'npy':
        # Save as numpy array (preserves float precision)
        np.save(output_path, depth_maps)
        print(f"Saved depth maps to {output_path}")
    elif save_format == 'png16':
        # Save as 16-bit PNG (one file per frame)
        output_dir = output_path.replace('.npy', '')
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, depth in enumerate(depth_maps):
            # Scale to 16-bit range
            depth_scaled = (depth / depth_teacher.max_depth * 65535).astype(np.uint16)
            png_path = os.path.join(output_dir, f'{idx:06d}.png')
            cv2.imwrite(png_path, depth_scaled)
        
        print(f"Saved {len(depth_maps)} depth PNGs to {output_dir}")
    else:
        raise ValueError(f"Unknown save format: {save_format}")
    
    return depth_maps


if __name__ == '__main__':
    """
    Example usage for preprocessing depth maps.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Precompute depth maps for videos')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save depth maps')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to depth model checkpoint')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'])
    parser.add_argument('--max_depth', type=float, default=20.0, help='Maximum depth in meters')
    parser.add_argument('--target_fps', type=int, default=5, help='Target FPS for depth extraction')
    parser.add_argument('--video_fps', type=int, default=30, help='Original video FPS')
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'png16'])
    
    args = parser.parse_args()
    
    # Load depth teacher
    print(f"Loading depth teacher model ({args.model_size})...")
    depth_teacher = load_depth_teacher(
        model_size=args.model_size,
        max_depth=args.max_depth,
        checkpoint_path=args.checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process all videos in directory
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos to process")
    
    for video_file in video_files:
        video_path = os.path.join(args.video_dir, video_file)
        output_name = video_file.replace('.mp4', '_depth.npy')
        output_path = os.path.join(args.output_dir, output_name)
        
        if os.path.exists(output_path):
            print(f"Skipping {video_file} (already processed)")
            continue
        
        try:
            precompute_depth_for_video(
                video_path=video_path,
                output_path=output_path,
                depth_teacher=depth_teacher,
                target_fps=args.target_fps,
                video_fps=args.video_fps,
                save_format=args.save_format
            )
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue
    
    print("Depth preprocessing complete!")

