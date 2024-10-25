import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models
from efficientnet_pytorch import EfficientNet

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.obs_encoder_type = cfg.model.obs_encoder.type
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
        self.encoder_feat_dim = cfg.model.encoder_feat_dim

        if self.do_rgb_normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Build the observation encoder
        if self.obs_encoder_type.startswith("efficientnet"):
            self.obs_encoder = EfficientNet.from_name(self.obs_encoder_type, in_channels=3)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder._fc = nn.Identity()
        elif self.obs_encoder_type.startswith("resnet"):
            model_constructor = getattr(models, self.obs_encoder_type)
            self.obs_encoder = model_constructor(pretrained=False)
            self.num_obs_features = self.obs_encoder.fc.in_features
            self.obs_encoder.fc = nn.Identity()
        elif self.obs_encoder_type.startswith("vit"):
            model_constructor = getattr(models, self.obs_encoder_type)
            self.obs_encoder = model_constructor(pretrained=False)
            self.num_obs_features = self.obs_encoder.hidden_dim
            self.obs_encoder.heads = nn.Identity()
        else:
            raise NotImplementedError(f"Observation encoder type {self.obs_encoder_type} not implemented")

        # Compress observation encodings to encoder_feat_dim
        if self.num_obs_features != self.encoder_feat_dim:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.encoder_feat_dim)
        else:
            self.compress_obs_enc = nn.Identity()

    def forward(self, obs):
        """
        Args:
            obs: (B*N, 3, H, W)
        Returns:
            embeddings: (B*N, encoder_feat_dim)
        """
        # Pre-processing
        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
        if self.do_resize:
            obs = F.interpolate(obs, size=(224, 224), mode='bilinear', align_corners=False)

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
        else:
            raise NotImplementedError(f"Observation encoder type {self.obs_encoder_type} not implemented")

        # Compress embeddings
        obs_enc = self.compress_obs_enc(obs_enc)

        return obs_enc

class VideoJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Initialize the online encoder
        self.online_encoder = ImageEncoder(cfg)

        # Initialize the target encoder as a copy of the online encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        # Freeze the target encoder parameters (they will be updated via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, decay=0.998):
        # EMA update for the target encoder
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.mul_(decay).add_(param_o.data * (1 - decay))

    def forward(self, input_obs, target_obs):
        """
        Args:
            input_obs: (B, N_in, 3, H, W) tensor of past frames
            target_obs: (B, N_target, 3, H, W) tensor of future frames
        Returns:
            online_embeddings: (B, N_in, encoder_feat_dim)
            target_embeddings: (B, N_target, encoder_feat_dim)
        """
        # Process input observations through online encoder
        B, N_in, C, H, W = input_obs.shape
        input_obs_flat = input_obs.view(B * N_in, C, H, W)
        online_embeddings_flat = self.online_encoder(input_obs_flat)
        online_embeddings = online_embeddings_flat.view(B, N_in, -1)

        # Process target observations through target encoder with stop gradient
        with torch.no_grad():
            B, N_target, C, H, W = target_obs.shape
            target_obs_flat = target_obs.view(B * N_target, C, H, W)
            target_embeddings_flat = self.target_encoder(target_obs_flat)
            target_embeddings = target_embeddings_flat.view(B, N_target, -1)

        return online_embeddings, target_embeddings
