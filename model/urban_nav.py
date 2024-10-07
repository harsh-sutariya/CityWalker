# Code inspired by https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from model.model_utils import PolarEmbedding, MultiLayerDecoder


class UrbanNav(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.obs_encoder_type = cfg.model.obs_encoder.type
        self.cord_embedding_type = cfg.model.cord_embedding.type
        self.decoder_type = cfg.model.decoder.type
        self.encoder_feat_dim = cfg.model.encoder_feat_dim
        self.len_traj_pred = cfg.model.decoder.len_traj_pred

        # Observation Encoder
        if self.obs_encoder_type .split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(self.obs_encoder_type , in_channels=3)
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError(f"Observation encoder type {self.obs_encoder_type} not implemented")
        
        # Coordinate Embedding
        if self.cord_embedding_type == 'polar':
            self.cord_embedding = PolarEmbedding(cfg)
            self.dim_cord_embedding = self.cord_embedding.out_dim * self.context_size
        else:
            raise NotImplementedError(f"Coordinate embedding type {self.cord_embedding_type} not implemented")

        # Compress observation and goal encodings to encoder_feat_dim
        if self.num_obs_features != self.encoder_feat_dim:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoder_feat_dim)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.dim_cord_embedding != self.encoder_feat_dim:
            self.compress_goal_enc = nn.Linear(self.dim_cord_embedding, self.encoder_feat_dim)
        else:
            self.compress_goal_enc = nn.Identity()

        # Decoder
        if cfg.model.decoder.type == "attention":
            self.decoder = MultiLayerDecoder(
                embed_dim=self.encoder_feat_dim,
                seq_len=self.context_size+1,
                output_layers=[256, 128, 64, 32],
                nhead=cfg.model.decoder.num_heads,
                num_layers=cfg.model.decoder.num_layers,
                ff_dim_factor=cfg.model.decoder.ff_dim_factor,
            )
            self.wp_predictor = nn.Linear(32, self.len_traj_pred * 2)
            self.arrive_predictor = nn.Linear(32, 1)
        else:
            raise NotImplementedError(f"Decoder type {cfg.model.decoder.type} not implemented")

    def forward(self, obs, cord):
        """
        Args:
            obs: (B, N, 3, H, W) tensor
            cord: (B, N, 2) tensor
        """
        B, N, _, H, W = obs.shape
        obs = obs.view(B * N, 3, H, W)
        # cord = cord.view(B * N, 2)
        
        if self.obs_encoder_type .split("-")[0] == "efficientnet":
            obs_enc = self.obs_encoder.extract_features(obs)
            obs_enc = self.obs_encoder._avg_pooling(obs_enc)
            if self.obs_encoder._global_params.include_top:
                obs_enc = obs_enc.flatten(start_dim=1)
                obs_enc = self.obs_encoder._dropout(obs_enc)

        if self.cord_embedding_type == 'polar':        
            cord_enc = self.cord_embedding(cord).view(B, -1)

        obs_enc = self.compress_obs_enc(obs_enc).view(B, N, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)

        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        
        # Decoder
        if self.decoder_type == "attention":
            dec_out = self.decoder(tokens)
            wp_pred = self.wp_predictor(dec_out).view(B, self.len_traj_pred, 2)
            arrive_pred = self.arrive_predictor(dec_out).view(B, 1)

        wp_pred = torch.cumsum(wp_pred, dim=1)

        return wp_pred, arrive_pred