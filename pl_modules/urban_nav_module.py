import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from model.urban_nav import UrbanNav
import matplotlib.pyplot as plt
import os
from time import time

class UrbanNavModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = UrbanNav(cfg)
        self.save_hyperparameters(cfg)
        self.num_visualize = cfg.validation.num_visualize
        self.current_visualization_count = 0
        self.result_dir = cfg.project.result_dir
        self.automatic_optimization = False
        self.batch_size = cfg.training.batch_size

    def forward(self, obs, cord):
        return self.model(obs, cord)

    def training_step(self, batches, batch_idx):
        opt = self.optimizers()
        for batch in batches:
            opt.zero_grad()
            obs = batch['video_frames']
            cord = batch['input_positions']
            wp_pred, arrive_pred = self(obs, cord)
            losses = self.compute_loss(wp_pred, arrive_pred, batch)
            waypoints_loss = losses['waypoints_loss']
            arrived_loss = losses['arrived_loss']
            total_loss = waypoints_loss + arrived_loss
            self.log('train/waypoints_loss', waypoints_loss, batch_size=self.batch_size)
            self.log('train/arrived_loss', arrived_loss, batch_size=self.batch_size)
            self.log('train/total_loss', total_loss, batch_size=self.batch_size)
            self.manual_backward(total_loss)
            opt.step()
        # return loss

    def validation_step(self, batches, batch_idx):
        video_loss = 0
        for batch in batches:
            obs = batch['video_frames']
            cord = batch['input_positions']
            wp_pred, arrive_pred = self(obs, cord)
            losses = self.compute_loss(wp_pred, arrive_pred, batch)
            waypoints_loss = losses['waypoints_loss']
            arrived_loss = losses['arrived_loss']
            total_loss = waypoints_loss + arrived_loss
            self.log('val/waypoints_loss', waypoints_loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
            self.log('val/arrived_loss', arrived_loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
            self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
            video_loss += total_loss

            if self.current_visualization_count < self.num_visualize:
                idx = 0
                original_input_positions = batch['original_input_positions'][idx].cpu().numpy()
                noisy_input_positions = batch['noisy_input_positions'][idx].cpu().numpy()
                gt_waypoints = batch['gt_waypoints'][idx].cpu().numpy()
                pred_waypoints = wp_pred[idx].detach().cpu().numpy()
                target_transformed = batch['target_transformed'][idx].cpu().numpy()
                # Get the last frame from the sequence
                frame = obs[idx, -1].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)  # Convert back to uint8 for visualization
                self.visualize_sample(
                    frame, original_input_positions, noisy_input_positions,
                    gt_waypoints, pred_waypoints, target_transformed,
                    self.current_epoch, self.current_visualization_count
                )
                self.current_visualization_count += 1

        return video_loss / len(batches)

    def on_validation_epoch_start(self):
        self.current_visualization_count = 0

    def compute_loss(self, wp_pred, arrive_pred, batch):
        waypoints_target = batch['waypoints']
        arrived_target = batch['arrived']
        wp_loss = F.mse_loss(wp_pred, waypoints_target)
        arrived_loss = F.binary_cross_entropy_with_logits(arrive_pred.flatten(), arrived_target)
        return {'waypoints_loss': wp_loss, 'arrived_loss': arrived_loss}

    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name
        lr = float(self.cfg.optimizer.lr)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler
        scheduler_cfg = self.cfg.scheduler
        if scheduler_cfg.name == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.max_epochs)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name == 'none':
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")

    def visualize_sample(self, frame, original_input_positions, noisy_input_positions,
                         gt_waypoints, pred_waypoints, target_transformed, epoch, sample_idx):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Left axis: plot the current observation (frame)
        ax1.imshow(frame)
        ax1.axis('off')
        ax1.set_title('Current Observation')

        # Right axis: plot the coordinates
        ax2.plot(original_input_positions[:, 0], original_input_positions[:, 1],
                 'o-', label='Original Input Positions', color='blue')
        ax2.plot(noisy_input_positions[:, 0], noisy_input_positions[:, 1],
                 'o-', label='Noisy Input Positions', color='orange')
        ax2.plot(gt_waypoints[:, 0], gt_waypoints[:, 1],
                 'x-', label='GT Waypoints', color='green')
        ax2.plot(pred_waypoints[:, 0], pred_waypoints[:, 1],
                 's-', label='Predicted Waypoints', color='red')
        ax2.plot(target_transformed[0], target_transformed[1],
                 marker='*', markersize=15, label='Target Coordinate', color='purple')
        ax2.legend()
        ax2.set_title('Coordinates')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.grid(True)

        # Optionally, customize color palette here based on cfg if needed

        # Save the plot
        output_dir = os.path.join(self.result_dir, f'epoch_{epoch}')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'sample_{sample_idx}.png')
        plt.savefig(output_path)
        plt.close(fig)
