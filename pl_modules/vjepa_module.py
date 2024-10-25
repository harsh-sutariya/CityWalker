import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from model.vjepa import VideoJEPA  # Adjust the import path as needed

class VJEPAModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        
        # Initialize the Video-JEPA Encoder model
        self.model = VideoJEPA(cfg)
        
        # EMA schedule parameters
        self.ema_start = cfg.model.ema_range[0]  # Starting EMA decay value (e.g., 0.99)
        self.ema_end = cfg.model.ema_range[1]    # Ending EMA decay value (e.g., 1.0)
        self.total_epochs = cfg.training.max_epochs  # Total number of training epochs
        
        # Regularization coefficient
        self.reg_coeff = cfg.training.reg_coeff  # e.g., 0.1
    
    def forward(self, input_obs, target_obs):
        return self.model(input_obs, target_obs)
    
    def training_step(self, batch, batch_idx):
        input_frames = batch['input_frames']    # Shape: (B, N_in, 3, H, W)
        target_frames = batch['target_frames']  # Shape: (B, N_target, 3, H, W)
        
        # Forward pass
        online_embeddings, target_embeddings = self(input_frames, target_frames)
        
        # Compute JEPA loss (L1 loss)
        loss_jepa = F.l1_loss(online_embeddings, target_embeddings)
        
        # Compute regularization loss
        pstd_z = torch.sqrt(online_embeddings.var(dim=1) + 1e-4)  # Shape: (B, feature_dim)
        loss_reg = F.relu(1.0 - pstd_z).mean()
        
        # Total loss
        loss = loss_jepa + self.reg_coeff * loss_reg
        
        # Log the losses
        self.log('train/loss_jepa', loss_jepa, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss_reg', loss_reg, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss_total', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Update the target encoder with EMA
        self.model.update_target_encoder(self.ema_momentum)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_frames = batch['input_frames']
        target_frames = batch['target_frames']
        
        # Forward pass
        online_embeddings, target_embeddings = self(input_frames, target_frames)
        
        # Compute JEPA loss (L1 loss)
        loss_jepa = F.l1_loss(online_embeddings, target_embeddings)
        
        # Compute regularization loss
        pstd_z = torch.sqrt(online_embeddings.var(dim=1) + 1e-4)
        loss_reg = F.relu(1.0 - pstd_z).mean()
        
        # Total loss
        loss = loss_jepa + self.reg_coeff * loss_reg
        
        # Log the validation losses
        self.log('val/loss_jepa', loss_jepa, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/loss_reg', loss_reg, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/loss_total', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
        lr = float(self.cfg.optimizer.lr)
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.online_encoder.parameters(),
                lr=lr,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.online_encoder.parameters(),
                lr=lr,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.online_encoder.parameters(),
                lr=lr,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.optimizer.name}")
        
        # Configure scheduler
        scheduler_cfg = self.cfg.scheduler
        if scheduler_cfg.name.lower() == 'step_lr':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_cfg.step_size,
                gamma=scheduler_cfg.gamma
            )
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.total_epochs
            )
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'none':
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_cfg.name}")
    
    def on_train_epoch_start(self):
        # Compute EMA decay for the current epoch
        m = self.compute_ema_decay(self.current_epoch)
        self.ema_momentum = m
    
    def compute_ema_decay(self, epoch_num):
        """
        Compute the EMA momentum based on the current epoch using a linear schedule.

        Args:
            epoch_num (int): Current epoch number

        Returns:
            float: Computed EMA momentum
        """
        m_start = self.ema_start
        m_end = self.ema_end
        total_epochs = self.total_epochs
        
        # Linear interpolation of momentum
        m = m_start + (m_end - m_start) * (epoch_num / total_epochs)
        m = min(m, m_end)  # Ensure m does not exceed m_end
        return m
