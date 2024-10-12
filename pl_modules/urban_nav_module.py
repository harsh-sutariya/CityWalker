import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from model.urban_nav import UrbanNav
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os

class UrbanNavModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = UrbanNav(cfg)
        self.save_hyperparameters(cfg)
        self.do_normalize = cfg.training.normalize_step_length
        
        # Visualization settings
        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.current_val_visualization_count = 0
        self.current_test_visualization_count = 0
        
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])
        
        # Initialize list to store test metrics
        self.test_metrics = []
        
    def forward(self, obs, cord):
        return self.model(obs, cord)

    def training_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        wp_pred, arrive_pred = self(obs, cord)
        losses = self.compute_loss(wp_pred, arrive_pred, batch)
        waypoints_loss = losses['waypoints_loss']
        arrived_loss = losses['arrived_loss']
        total_loss = waypoints_loss + arrived_loss
        self.log('train/l_wp', waypoints_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/l_arvd', arrived_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        wp_pred, arrive_pred = self(obs, cord)
        
        # Compute L1 loss for waypoints
        l1_loss = self.compute_loss(wp_pred, arrive_pred, batch)['waypoints_loss']

        # Compute accuracy for "arrived" prediction
        arrived_target = batch['arrived']
        arrived_logits = arrive_pred.flatten()
        arrived_probs = torch.sigmoid(arrived_logits)
        arrived_pred_binary = (arrived_probs >= 0.5).float()
        correct = (arrived_pred_binary == arrived_target).float()
        accuracy = correct.sum() / correct.numel()

        # Log the metrics
        self.log('val/l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/arrived_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Handle visualization
        self.process_visualization(
            mode='val',
            batch=batch,
            obs=obs,
            wp_pred=wp_pred,
            arrive_pred=arrive_pred
        )
        
        return l1_loss

    def test_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        wp_pred, arrive_pred = self(obs, cord)

        # Compute L1 loss for waypoints
        waypoints_target = batch['waypoints']
        l1_loss = F.l1_loss(wp_pred, waypoints_target, reduction='mean').item()

        # Compute accuracy for "arrived" prediction
        arrived_target = batch['arrived']
        arrived_logits = arrive_pred.flatten()
        arrived_probs = torch.sigmoid(arrived_logits)
        arrived_pred_binary = (arrived_probs >= 0.5).float()
        correct = (arrived_pred_binary == arrived_target).float()
        accuracy = correct.sum().item() / correct.numel()

        # Store the metrics
        self.test_metrics.append({
            'l1_loss': l1_loss,
            'arrived_accuracy': accuracy
        })

        # Handle visualization
        self.process_visualization(
            mode='test',
            batch=batch,
            obs=obs,
            wp_pred=wp_pred,
            arrive_pred=arrive_pred
        )

    def on_test_epoch_end(self):
        # Save the test metrics to a .npy file
        test_metrics_array = np.array(self.test_metrics, dtype=object)
        test_metrics_path = os.path.join(self.result_dir, 'test_metrics.npy')
        np.save(test_metrics_path, test_metrics_array)
        print(f"Test metrics saved to {test_metrics_path}")

    def on_validation_epoch_start(self):
        self.current_val_visualization_count = 0

    def on_test_epoch_start(self):
        self.current_test_visualization_count = 0

    def compute_loss(self, wp_pred, arrive_pred, batch):
        waypoints_target = batch['waypoints']
        arrived_target = batch['arrived']
        wp_loss = F.l1_loss(wp_pred, waypoints_target)
        arrived_loss = F.binary_cross_entropy_with_logits(arrive_pred.flatten(), arrived_target)

        if self.do_normalize:
            wp_scale = waypoints_target[:, 0, :].norm(p=2, dim=1).mean()
            wp_loss = wp_loss / wp_scale
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

    def process_visualization(self, mode, batch, obs, wp_pred, arrive_pred):
        """
        Handles visualization for both validation and testing.

        Args:
            mode (str): 'val' or 'test'
            batch (dict): Batch data
            obs (torch.Tensor): Observation frames
            wp_pred (torch.Tensor): Predicted waypoints
            arrive_pred (torch.Tensor): Predicted arrival logits
        """
        if mode == 'val':
            num_visualize = self.val_num_visualize
            current_count = self.current_val_visualization_count
            vis_dir = os.path.join(self.result_dir, 'val_vis', f'epoch_{self.current_epoch}')
        elif mode == 'test':
            num_visualize = self.test_num_visualize
            current_count = self.current_test_visualization_count
            vis_dir = os.path.join(self.result_dir, 'test_vis')
        else:
            raise ValueError("Mode should be either 'val' or 'test'.")

        os.makedirs(vis_dir, exist_ok=True)

        batch_size = obs.size(0)
        for idx in range(batch_size):
            if mode == 'val':
                if current_count >= num_visualize:
                    break
            elif mode == 'test':
                if current_count >= num_visualize:
                    break

            # Extract necessary data
            arrived_target = batch['arrived'][idx].item()
            arrived_logits = arrive_pred[idx].flatten()
            arrived_probs = torch.sigmoid(arrived_logits).item()
            # arrived_pred_binary = (arrived_probs >= 0.5).float().item()

            original_input_positions = batch['original_input_positions'][idx].cpu().numpy()
            noisy_input_positions = batch['noisy_input_positions'][idx].cpu().numpy()
            gt_waypoints = batch['gt_waypoints'][idx].cpu().numpy()
            pred_waypoints = wp_pred[idx].detach().cpu().numpy()
            target_transformed = batch['target_transformed'][idx].cpu().numpy()

            if self.do_normalize:
                step_length = np.linalg.norm(gt_waypoints, axis=1).mean()
                original_input_positions = original_input_positions / step_length
                noisy_input_positions = noisy_input_positions / step_length
                gt_waypoints = gt_waypoints / step_length
                pred_waypoints = pred_waypoints / step_length
                target_transformed = target_transformed / step_length

            # Get the last frame from the sequence
            frame = obs[idx, -1].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8 for visualization

            # Visualization title
            arrive_title = f"Arrived GT: {'True' if arrived_target else 'False'}, Pred: {arrived_probs:.2f}"

            # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Left axis: plot the current observation (frame) with arrived info in title
            ax1.imshow(frame)
            ax1.axis('off')
            ax1.set_title(arrive_title)

            # Right axis: plot the coordinates
            ax2.axis('equal')
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
            ax2.set_xlabel('X (step length)')
            ax2.set_ylabel('Y (step length)')
            ax2.grid(True)

            # Save the plot
            output_path = os.path.join(vis_dir, f'sample_{current_count}.png')
            plt.savefig(output_path)
            plt.close(fig)

            # Update visualization count
            if mode == 'val':
                self.current_val_visualization_count += 1
            elif mode == 'test':
                self.current_test_visualization_count += 1
