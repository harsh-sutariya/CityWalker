import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models
from efficientnet_pytorch import EfficientNet
from model.model_utils import JEPAPredictor

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

        # print(self.num_obs_features)
        # print(self.encoder_feat_dim)
        # assert()

    def forward(self, obs):
        """
        Args:
            obs: (B*N, 3, H, W)
        Returns:
            embeddings: (B*N, encoder_feat_dim)
        """

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
        self.cfg = cfg
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
        self.obs_encoder_type = cfg.model.obs_encoder.type
        if self.do_rgb_normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
        # Initialize the online encoder
        self.online_encoder = ImageEncoder(cfg)

        # Initialize the target encoder as a copy of the online encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Initialize the predictor using the decoder from UrbanNav
        # Assuming UrbanNav's decoder is compatible as a predictor
        # You might need to adjust the parameters based on the actual UrbanNav decoder implementation
        if cfg.model.decoder.type == "attention":
            self.predictor = JEPAPredictor(
                embed_dim=self.online_encoder.encoder_feat_dim,
                seq_len=cfg.model.obs_encoder.context_size,
                nhead=cfg.model.decoder.num_heads,
                num_layers=cfg.model.decoder.num_layers,
                ff_dim_factor=cfg.model.decoder.ff_dim_factor,
            )
        else:
            raise NotImplementedError(f"Decoder type {cfg.model.decoder.type} not implemented")

    @torch.no_grad()
    def update_target_encoder(self, m):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.mul_(m).add_((1. - m) * param_o.data)

    def forward(self, input_obs, target_obs):
        """
        Args:
            input_obs: (B, N_in, 3, H, W) tensor of input frames
            target_obs: (B, N_target, 3, H, W) tensor of target frames
        Returns:
            predicted_embeddings: (B, N_target, encoder_feat_dim)
            target_embeddings: (B, N_target, encoder_feat_dim)
        """
        B, N_in, C, H, W = input_obs.shape
        B, N_target, C, H, W = target_obs.shape
        input_obs = input_obs.view(B * N_in, C, H, W)
        target_obs = target_obs.view(B * N_target, C, H, W)
        if self.do_rgb_normalize:
            input_obs = (input_obs - self.mean) / self.std
            target_obs = (target_obs - self.mean) / self.std
        if self.do_resize:
            if self.obs_encoder_type.startswith("vit"):
                input_obs = F.interpolate(input_obs, size=(224, 224), mode='bilinear', align_corners=False)
                target_obs = F.interpolate(target_obs, size=(224, 224), mode='bilinear', align_corners=False)
            else:
                input_obs = F.interpolate(input_obs, size=(400, 400), mode='bilinear', align_corners=False)
                target_obs = F.interpolate(target_obs, size=(400, 400), mode='bilinear', align_corners=False)

        online_embeddings_flat = self.online_encoder(input_obs)  # (B*N_in, encoder_feat_dim)
        online_embeddings = online_embeddings_flat.view(B, N_in, -1)  # (B, N_in, encoder_feat_dim)
        # print(online_embeddings.size())

        # Predict target embeddings using the predictor (decoder)
        # The predictor takes the sequence of input embeddings and predicts the sequence of target embeddings
        predicted_embeddings = self.predictor(online_embeddings)  # (B, N_target, hidden_dim)
        # print(predicted_embeddings.size())

        # Encode target observations through target encoder
        target_embeddings_flat = self.target_encoder(target_obs)  # (B*N_target, encoder_feat_dim)
        target_embeddings = target_embeddings_flat.view(B, N_target, -1)  # (B, N_target, encoder_feat_dim)
        # print(target_embeddings.size())
        # assert()

        return predicted_embeddings, target_embeddings
