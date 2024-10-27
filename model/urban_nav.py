import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from model.model_utils import PolarEmbedding, MultiLayerDecoder
from torchvision import models

class UrbanNav(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.obs_encoder_type = cfg.model.obs_encoder.type
        self.cord_embedding_type = cfg.model.cord_embedding.type
        self.decoder_type = cfg.model.decoder.type
        self.encoder_feat_dim = cfg.model.encoder_feat_dim
        self.len_traj_pred = cfg.model.decoder.len_traj_pred
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
        self.output_coordinate_repr = cfg.model.output_coordinate_repr  # 'polar' or 'euclidean'

        if self.obs_encoder_type.startswith("dinov2"):
            self.image_height = cfg.model.obs_encoder.image_height
            self.image_width = cfg.model.obs_encoder.image_width

        if self.do_rgb_normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Observation Encoder
        if self.obs_encoder_type .split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(self.obs_encoder_type , in_channels=3)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder._fc = nn.Identity()  # Remove classification layer
        elif self.obs_encoder_type.startswith("resnet"):
            model_constructor = getattr(models, self.obs_encoder_type)
            self.obs_encoder = model_constructor()
            self.num_obs_features = self.obs_encoder.fc.in_features
            self.obs_encoder.fc = nn.Identity()  # Remove classification layer
        elif self.obs_encoder_type.startswith("vit"):
            model_constructor = getattr(models, self.obs_encoder_type)
            self.obs_encoder = model_constructor()
            self.num_obs_features = self.obs_encoder.hidden_dim
            self.obs_encoder.heads = nn.Identity()  # Remove classification head
        elif self.obs_encoder_type.startswith("dinov2"):
            self.obs_encoder = torch.hub.load('facebookresearch/dinov2', self.obs_encoder_type)
            feature_dim = {
                "dinov2_vits14": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitg14": 1536,
            }
            self.num_obs_features = feature_dim[self.obs_encoder_type]
        else:
            raise NotImplementedError(f"Observation encoder type {self.obs_encoder_type} not implemented")

        # Coordinate Embedding
        if self.cord_embedding_type == 'polar':
            self.cord_embedding = PolarEmbedding(cfg)
            self.dim_cord_embedding = self.cord_embedding.out_dim * self.context_size
        elif self.cord_embedding_type == 'target':
            self.cord_embedding = PolarEmbedding(cfg)
            self.dim_cord_embedding = self.cord_embedding.out_dim
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
        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
        if self.do_resize:
            if self.obs_encoder_type.startswith("vit"):
                obs = F.interpolate(obs, size=(224, 224), mode='bilinear', align_corners=False)
            elif self.obs_encoder_type.startswith("dinov2"):
                obs = F.interpolate(obs, size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)
            else:
                obs = F.interpolate(obs, size=(400, 400), mode='bilinear', align_corners=False)

        # Observation Encoding
        if self.obs_encoder_type.startswith("efficientnet"):
            obs_enc = self.obs_encoder.extract_features(obs)
            obs_enc = self.obs_encoder._avg_pooling(obs_enc)
            obs_enc = obs_enc.flatten(start_dim=1)
        elif self.obs_encoder_type.startswith("resnet"):
            x = self.obs_encoder.conv1(obs)
            x = self.obs_encoder.bn1(x)
            x = self.obs_encoder.relu(x)
            x = self.obs_encoder.maxpool(x)
            x = self.obs_encoder.layer1(x)
            x = self.obs_encoder.layer2(x)
            x = self.obs_encoder.layer3(x)
            x = self.obs_encoder.layer4(x)
            x = self.obs_encoder.avgpool(x)
            obs_enc = torch.flatten(x, 1)
        elif self.obs_encoder_type.startswith("vit"):
            obs_enc = self.obs_encoder(obs)  # Returns class token embedding
        elif self.obs_encoder_type.startswith("dinov2"):
            obs_enc = self.obs_encoder(obs)
        else:
            raise NotImplementedError(f"Observation encoder type {self.obs_encoder_type} not implemented")

        obs_enc = self.compress_obs_enc(obs_enc).view(B, N, -1)

        # Coordinate Encoding
        cord_enc = self.cord_embedding(cord).view(B, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(B, 1, -1)

        tokens = torch.cat([obs_enc, cord_enc], dim=1)

        # Decoder
        if self.decoder_type == "attention":
            dec_out = self.decoder(tokens)
            wp_pred = self.wp_predictor(dec_out).view(B, self.len_traj_pred, 2)
            arrive_pred = self.arrive_predictor(dec_out).view(B, 1)

        # Waypoint Prediction Processing
        if self.output_coordinate_repr == 'euclidean':
            # Predict deltas and compute cumulative sum
            wp_pred = torch.cumsum(wp_pred, dim=1)
            return wp_pred, arrive_pred
        elif self.output_coordinate_repr == 'polar':
            # Convert polar deltas to Cartesian deltas and compute cumulative sum
            distances = wp_pred[:, :, 0]
            angles = wp_pred[:, :, 1]
            dx = distances * torch.cos(angles)
            dy = distances * torch.sin(angles)
            deltas = torch.stack([dx, dy], dim=-1)
            wp_pred = torch.cumsum(deltas, dim=1)
            return wp_pred, arrive_pred, distances, angles
        else:
            raise NotImplementedError(f"Output coordinate representation {self.output_coordinate_repr} not implemented")
