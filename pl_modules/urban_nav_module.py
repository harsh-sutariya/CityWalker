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
        
        # Coordinate representation
        self.output_coordinate_repr = cfg.model.output_coordinate_repr
        if self.output_coordinate_repr not in ["euclidean", "polar"]:
            raise ValueError(f"Unsupported coordinate representation: {self.output_coordinate_repr}")
        
        self.decoder = cfg.model.decoder.type
        if self.decoder not in ["diff_policy", "attention"]:
            raise ValueError(f"Unsupported decoder: {self.decoder}")
        
        # Direction loss weight (you can adjust this value in your cfg)
        self.direction_loss_weight = cfg.training.direction_loss_weight
        
        # Visualization settings
        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.vis_count = 0
        
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])
        
        # If polar, define additional loss weights
        if self.output_coordinate_repr == "polar":
            self.distance_loss_weight = cfg.training.distance_loss_weight
            self.angle_loss_weight = cfg.training.angle_loss_weight

    def forward(self, obs, cord, gt_action=None):
        return self.model(obs, cord, gt_action)
    
    def training_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        
        if self.decoder == "attention":
            if self.output_coordinate_repr == "euclidean":
                wp_pred, arrive_pred = self(obs, cord)
                losses = self.compute_loss(wp_pred, arrive_pred, batch)
                waypoints_loss = losses['waypoints_loss']
                arrived_loss = losses['arrived_loss']
                direction_loss = losses['direction_loss']
                total_loss = waypoints_loss + arrived_loss + self.direction_loss_weight * direction_loss
                self.log('train/l_wp', waypoints_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            elif self.output_coordinate_repr == "polar":
                wp_pred_euclidean, arrive_pred, distance_pred, angle_pred = self(obs, cord)
                losses = self.compute_loss_polar(wp_pred_euclidean, distance_pred, angle_pred, arrive_pred, batch)
                distance_loss = losses['distance_loss']
                angle_loss = losses['angle_loss']
                arrived_loss = losses['arrived_loss']
                direction_loss = losses['direction_loss']
                total_loss = (self.distance_loss_weight * distance_loss +
                            self.angle_loss_weight * angle_loss +
                            arrived_loss +
                            self.direction_loss_weight * direction_loss)
                self.log('train/l_distance', distance_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('train/l_angle', angle_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        elif self.decoder == "diff_policy":
            wp_pred, noise_pred, arrived_pred, noise = self(obs, cord, batch['waypoints'])
            losses = self.compute_loss_diff_policy(wp_pred, noise_pred, arrived_pred, noise, batch)
            noise_loss = losses['noise_loss']
            arrived_loss = losses['arrived_loss']
            direction_loss = losses['direction_loss']
            total_loss = noise_loss + arrived_loss + self.direction_loss_weight * direction_loss
            self.log('train/l_noise', noise_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Common logs
        self.log('train/l_arvd', arrived_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/l_dir', direction_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        if self.decoder == "attention":
            if self.output_coordinate_repr == "euclidean":
                wp_pred, arrive_pred = self(obs, cord)
                losses = self.compute_loss(wp_pred, arrive_pred, batch)
                l1_loss = losses['waypoints_loss']
                direction_loss = losses['direction_loss']
                self.log('val/l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                
            elif self.output_coordinate_repr == "polar":
                wp_pred, arrive_pred, distance_pred, angle_pred = self(obs, cord)
                losses = self.compute_loss_polar(wp_pred, distance_pred, angle_pred, arrive_pred, batch)
                direction_loss = losses['direction_loss']
                self.log('val/distance_loss', losses['distance_loss'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('val/angle_loss', losses['angle_loss'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        elif self.decoder == "diff_policy":
            wp_pred, noise_pred, arrive_pred, noise = self(obs, cord, batch['waypoints'])
            losses = self.compute_loss_diff_policy(wp_pred, noise_pred, arrive_pred, noise, batch)
            noise_loss = losses['noise_loss']
            direction_loss = losses['direction_loss']
            self.log('val/noise_loss', noise_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Compute accuracy for "arrived" prediction
        arrived_target = batch['arrived']
        arrived_logits = arrive_pred.flatten()
        arrived_probs = torch.sigmoid(arrived_logits)
        arrived_pred_binary = (arrived_probs >= 0.5).float()
        correct = (arrived_pred_binary == arrived_target).float()
        accuracy = correct.sum() / correct.numel()
        
        # Log the metrics
        self.log('val/arrived_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/direction_loss', direction_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Handle visualization
        self.process_visualization(
            mode='val',
            batch=batch,
            obs=obs,
            wp_pred=wp_pred,
            arrive_pred=arrive_pred
        )
        
        return direction_loss

    def test_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        
        if self.output_coordinate_repr == "euclidean":
            wp_pred, arrive_pred = self(obs, cord)
            # Compute L1 loss for waypoints
            waypoints_target = batch['waypoints']
            l1_loss = F.l1_loss(wp_pred, waypoints_target, reduction='mean').item()
        elif self.output_coordinate_repr == "polar":
            wp_pred, arrive_pred, distance_pred, angle_pred = self(obs, cord)
            waypoints_target = batch['waypoints']
            distance_loss, angle_loss, _, _ = self.compute_loss_polar(wp_pred, distance_pred, angle_pred, arrive_pred, batch)
            
        
        # Compute accuracy for "arrived" prediction
        arrived_target = batch['arrived']
        arrived_logits = arrive_pred.flatten()
        arrived_probs = torch.sigmoid(arrived_logits)
        arrived_pred_binary = (arrived_probs >= 0.5).float()
        correct = (arrived_pred_binary == arrived_target).float()
        accuracy = correct.sum().item() / correct.numel()

        # wp_pred_last = wp_pred[:, -1, :]  # shape [batch_size, 2]
        # waypoints_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

        # Compute cosine similarity
        B, T, _ = wp_pred.shape
        wp_pred_view = wp_pred.view(-1, 2)
        waypoints_target_view = waypoints_target.view(-1, 2)
        dot_product = (wp_pred_view * waypoints_target_view).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
        norm_target = waypoints_target_view.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero
        
        # Compute angle in degrees
        angle = torch.acos(cos_sim.clamp(-1+1e-7, 1-1e-7)) * 180 / torch.pi  # shape [batch_size]
        angle = angle.view(B, T)
        
        # Take mean angle
        mean_angle = angle.mean(dim=0).cpu().numpy()
        
        # Store the metrics
        if self.output_coordinate_repr == "euclidean":
            self.test_metrics['l1_loss'].append(l1_loss)
        elif self.output_coordinate_repr == "polar":
            self.test_metrics['distance_loss'].append(distance_loss)
            self.test_metrics['angle_loss'].append(angle_loss)
        self.test_metrics['arrived_accuracy'].append(accuracy)
        self.test_metrics['mean_angle'].append(mean_angle)
        
        # Handle visualization
        if self.output_coordinate_repr == "euclidean":
            self.process_visualization(
                mode='test',
                batch=batch,
                obs=obs,
                wp_pred=wp_pred,
                arrive_pred=arrive_pred
            )
        elif self.output_coordinate_repr == "polar":
            self.process_visualization(
                mode='test',
                batch=batch,
                obs=obs,
                wp_pred=wp_pred,
                arrive_pred=arrive_pred
            )

    def on_test_epoch_end(self):
        for metric in self.test_metrics:
            metric_array = np.array(self.test_metrics[metric])
            save_path = os.path.join(self.result_dir, f'test_{metric}.npy')
            np.save(save_path, metric_array)
            if not metric == "mean_angle":
                print(f"Test mean {metric} {metric_array.mean():.4f} saved to {save_path}")
            else:
                mean_angle = metric_array.mean(axis=0)
                for i in range(len(mean_angle)):
                    print(f"Test mean angle at step {i} {mean_angle[i]:.4f}")

    def on_validation_epoch_start(self):
        self.vis_count = 0

    def on_test_epoch_start(self):
        self.vis_count = 0
        if self.output_coordinate_repr == "euclidean":
            self.test_metrics = {'l1_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
        elif self.output_coordinate_repr == "polar":
            self.test_metrics = {'distance_loss': [], 'angle_loss': [], 'arrived_accuracy': [], 'mean_angle': []}

    def compute_loss(self, wp_pred, arrive_pred, batch):
        waypoints_target = batch['waypoints']
        arrived_target = batch['arrived']
        wp_loss = F.l1_loss(wp_pred, waypoints_target)
        arrived_loss = F.binary_cross_entropy_with_logits(arrive_pred.flatten(), arrived_target)

        # Compute direction loss
        wp_pred_last = wp_pred[:, -1, :]  # shape [batch_size, 2]
        wp_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

        # Compute cosine similarity
        dot_product = (wp_pred_last * wp_target_last).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_last.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_last.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = 1 - cos_sim.mean()

        if self.do_normalize:
            wp_scale = waypoints_target[:, 0, :].norm(p=2, dim=1).mean()
            wp_loss = wp_loss / wp_scale
            # Optionally, normalize direction_loss as well
            # direction_loss = direction_loss / wp_scale

        return {'waypoints_loss': wp_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss}
    
    def compute_loss_polar(self, wp_pred_euclidean, distance_pred, angle_pred, arrive_pred, batch):
        waypoints_target = batch['waypoints']
        arrived_target = batch['arrived']
        
        # Compute distance and angle targets
        distance_target, angle_target = self.waypoints_to_polar(waypoints_target)
        
        # Compute L1 loss for distance and angle
        distance_loss = F.l1_loss(distance_pred, distance_target)
        angle_loss = F.l1_loss(angle_pred, angle_target)
        
        # Compute arrived loss
        arrived_loss = F.binary_cross_entropy_with_logits(arrive_pred.flatten(), arrived_target)
        
        # Compute direction loss using Euclidean waypoints
        wp_pred_last = wp_pred_euclidean[:, -1, :]  # shape [batch_size, 2]
        wp_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

        # Compute cosine similarity
        dot_product = (wp_pred_last * wp_target_last).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_last.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_last.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = (1 - cos_sim.mean()) ** 2

        if self.do_normalize:
            distance_loss = distance_loss / distance_target.mean()

        return {'distance_loss': distance_loss, 'angle_loss': angle_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss}

    def compute_loss_diff_policy(self, wp_pred, noise_pred, arrived_pred, noise, batch):
        # Compute loss for noise prediction
        waypoints_target = batch['waypoints']
        noise_loss = F.mse_loss(noise_pred, noise)
        
        # Compute loss for arrived prediction
        arrived_target = batch['arrived']
        arrived_loss = F.binary_cross_entropy_with_logits(arrived_pred.flatten(), arrived_target)

        # Compute direction loss
        wp_pred_last = wp_pred[:, -1, :]  # shape [batch_size, 2]
        wp_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

        # Compute cosine similarity
        dot_product = (wp_pred_last * wp_target_last).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_last.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_last.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = (1 - cos_sim.mean()) ** 2
        
        return {'noise_loss': noise_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss}
    
    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
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
        if scheduler_cfg.name.lower() == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.max_epochs)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'none':
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
            vis_dir = os.path.join(self.result_dir, 'val_vis', f'epoch_{self.current_epoch}')
        elif mode == 'test':
            num_visualize = self.test_num_visualize
            vis_dir = os.path.join(self.result_dir, 'test_vis')
        else:
            raise ValueError("Mode should be either 'val' or 'test'.")

        os.makedirs(vis_dir, exist_ok=True)

        batch_size = obs.size(0)
        for idx in range(batch_size):
            if self.vis_count >= num_visualize:
                break

            # Extract necessary data
            arrived_target = batch['arrived'][idx].item()
            arrived_logits = arrive_pred[idx].flatten()
            arrived_probs = torch.sigmoid(arrived_logits).item()

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
            output_path = os.path.join(vis_dir, f'sample_{self.vis_count}.png')
            plt.savefig(output_path)
            plt.close(fig)

            self.vis_count += 1

    def waypoints_to_polar(self, waypoints):
        # Compute relative differences
        deltas = torch.diff(waypoints, dim=1, prepend=torch.zeros_like(waypoints[:, :1, :]))
        distance = torch.norm(deltas, dim=2)
        angle = torch.atan2(deltas[:, :, 1], deltas[:, :, 0])
        return distance, angle
