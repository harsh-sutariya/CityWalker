import os
import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import torch.nn.functional as F
from tqdm import tqdm

class VideoJEPADataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.video_dir = cfg.data.video_dir
        self.context_size = cfg.model.obs_encoder.context_size  # Number of input frames
        self.target_size = cfg.model.obs_encoder.target_size  # Number of target frames
        self.video_fps = cfg.data.video_fps
        self.target_fps = cfg.data.target_fps
        self.frame_multiplier = self.video_fps // self.target_fps

        # Load video paths
        self.video_paths = [
            os.path.join(self.video_dir, f)
            for f in sorted(os.listdir(self.video_dir))
            if f.endswith('.mp4')
        ]
        print(f"Total videos found: {len(self.video_paths)}")

        # Split videos according to mode
        if mode == 'train':
            self.video_paths = self.video_paths[:cfg.data.num_train]
        elif mode == 'val':
            self.video_paths = self.video_paths[cfg.data.num_train: cfg.data.num_train + cfg.data.num_val]
        elif mode == 'test':
            self.video_paths = self.video_paths[cfg.data.num_train + cfg.data.num_val:
                                               cfg.data.num_train + cfg.data.num_val + cfg.data.num_test]
        else:
            raise ValueError(f"Invalid mode {mode}")

        print(f"Number of videos for {mode}: {len(self.video_paths)}")

        # Build the look-up table (lut) and video_ranges
        self.lut = []
        self.video_ranges = []
        idx_counter = 0
        for video_idx, video_path in enumerate(tqdm(self.video_paths, desc="Building LUT")):
            # Initialize VideoReader to get number of frames
            vr = VideoReader(video_path, ctx=cpu(0))
            num_frames = len(vr)
            usable_frames = num_frames // self.frame_multiplier - (self.context_size + self.target_size)
            if usable_frames <= 0:
                continue  # Skip videos that are too short
            start_idx = idx_counter
            for frame_start in range(0, usable_frames, self.context_size):
                self.lut.append((video_idx, frame_start))
                idx_counter += 1
            end_idx = idx_counter
            self.video_ranges.append((start_idx, end_idx))
        assert len(self.lut) > 0, "No usable samples found."

        print(f"Total samples in LUT: {len(self.lut)}")
        print(f"Total video ranges: {len(self.video_ranges)}")

        # Initialize the video reader cache per worker
        self.video_reader_cache = {'video_idx': None, 'video_reader': None}

    def __len__(self):
        return len(self.lut)

    def __getitem__(self, index):
        video_idx, frame_start = self.lut[index]

        # Retrieve or create the VideoReader for the current video
        if self.video_reader_cache['video_idx'] != video_idx:
            # Replace the old VideoReader with the new one
            self.video_reader_cache['video_reader'] = VideoReader(self.video_paths[video_idx], ctx=cpu(0))
            self.video_reader_cache['video_idx'] = video_idx
        video_reader = self.video_reader_cache['video_reader']

        # Compute actual frame indices for input and target
        actual_frame_start = frame_start * self.frame_multiplier
        frame_indices_input = actual_frame_start + np.arange(self.context_size) * self.frame_multiplier
        frame_indices_target = actual_frame_start + (self.context_size + np.arange(self.target_size)) * self.frame_multiplier

        # Ensure frame indices are within the video length
        num_frames = len(video_reader)
        frame_indices_input = [min(idx, num_frames - 1) for idx in frame_indices_input]
        frame_indices_target = [min(idx, num_frames - 1) for idx in frame_indices_target]

        # Load the frames
        input_frames = video_reader.get_batch(frame_indices_input).asnumpy()
        target_frames = video_reader.get_batch(frame_indices_target).asnumpy()

        # Process frames
        input_frames = self.process_frames(input_frames)
        target_frames = self.process_frames(target_frames)

        sample = {
            'input_frames': input_frames,     # Shape: (context_size, 3, H, W)
            'target_frames': target_frames    # Shape: (target_size, 3, H, W)
        }

        return sample

    def process_frames(self, frames):
        """
        Convert frames to tensor, normalize, and resize if necessary.

        Args:
            frames (numpy.ndarray): Array of frames with shape (N, H, W, C).

        Returns:
            torch.Tensor: Processed frames with shape (N, 3, H, W).
        """
        # Convert frames to tensor and normalize
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0  # Shape: (N, 3, H, W)

        # # Optional resizing
        # if self.cfg.data.do_resize:
        #     frames = F.interpolate(frames, size=(self.cfg.data.resize_height, self.cfg.data.resize_width),
        #                            mode='bilinear', align_corners=False)

        # # Optional normalization
        # if self.cfg.data.do_rgb_normalize:
        #     mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(frames.device)
        #     std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(frames.device)
        #     frames = (frames - mean) / std

        return frames
