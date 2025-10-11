import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from model.model_utils import PolarEmbedding, MultiLayerDecoder, PositionalEncoding
from torchvision import models

class CityWalker(nn.Module):
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
        
        # DBR (Depth Barrier Regularization) support
        self.use_dbr = getattr(cfg.model, 'use_dbr', False)
        if self.use_dbr:
            from model.dbr import DBRModule
            self.dbr_module = DBRModule(cfg)

        # if self.obs_encoder_type.startswith("dinov2"):
        self.crop = cfg.model.obs_encoder.crop
        self.resize = cfg.model.obs_encoder.resize

        if self.do_rgb_normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Observation Encoder
        if self.obs_encoder_type.startswith("efficientnet"):
            model_constructor = getattr(models, self.obs_encoder_type)
            self.obs_encoder = model_constructor(weights="DEFAULT")
            self.num_obs_features = self.obs_encoder.classifier[1].in_features
            self.obs_encoder.classifier = nn.Identity()  # Remove classification layer
        elif self.obs_encoder_type.startswith("resnet"):
            model_constructor = getattr(models, self.obs_encoder_type)
            self.obs_encoder = model_constructor(weights="DEFAULT")
            self.num_obs_features = self.obs_encoder.fc.in_features
            self.obs_encoder.fc = nn.Identity()  # Remove classification layer
        elif self.obs_encoder_type.startswith("vit"):
            model_constructor = getattr(models, self.obs_encoder_type)
            self.obs_encoder = model_constructor(weights="IMAGENET1K_SWAG_E2E_V1")
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
            if cfg.model.obs_encoder.freeze:
                for param in self.obs_encoder.parameters():
                    param.requires_grad = False
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
        elif self.cord_embedding_type == 'input_target':
            self.cord_embedding = PolarEmbedding(cfg)
            self.dim_cord_embedding = self.cord_embedding.out_dim * (self.context_size + 1)
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
        elif cfg.model.decoder.type == "diff_policy":
            from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            self.positional_encoding = PositionalEncoding(self.encoder_feat_dim, max_seq_len=self.context_size+1)
            self.sa_layer = nn.TransformerEncoderLayer(
                d_model=self.encoder_feat_dim, 
                nhead=cfg.model.decoder.num_heads, 
                dim_feedforward=cfg.model.decoder.ff_dim_factor*self.encoder_feat_dim, 
                activation="gelu", 
                batch_first=True, 
                norm_first=True
            )
            self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=cfg.model.decoder.num_layers)
            self.wp_predictor = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=self.encoder_feat_dim,
                down_dims=[64, 128, 256],
                cond_predict_scale=False,
            )
            self.arrive_predictor = nn.Sequential(
                nn.Linear(self.encoder_feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.model.decoder.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        else:
            raise NotImplementedError(f"Decoder type {cfg.model.decoder.type} not implemented")  

    def forward(self, obs, cord, gt_action=None, depth_map=None, depth_mask=None):
        """
        Args:
            obs: (B, N, 3, H, W) tensor
            cord: (B, N, 2) tensor
            gt_action: (B, T, 2) tensor (optional, for diffusion policy)
            depth_map: (B, H, W) tensor (optional, for DBR)
            depth_mask: (B, H, W) tensor (optional, for DBR)
        """
        B, N, _, H, W = obs.shape
        obs = obs.view(B * N, 3, H, W)
        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
        if self.do_resize:
            obs = TF.center_crop(obs, self.crop)
            obs = TF.resize(obs, self.resize)

        # Observation Encoding
        if self.obs_encoder_type.startswith("efficientnet"):
            obs_enc = self.obs_encoder(obs)
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
        elif self.decoder_type == "diff_policy":
            tokens = self.positional_encoding(tokens)
            dec_out = self.sa_decoder(tokens).mean(dim=1)
            
            deltas = torch.diff(gt_action, dim=1, prepend=torch.zeros_like(gt_action[:, :1, :]))
            if self.output_coordinate_repr == 'polar':
                distances = torch.norm(deltas, dim=-1)
                angles = torch.atan2(deltas[:, :, 1], deltas[:, :, 0])
                deltas = torch.stack([distances, angles], dim=-1)
            elif self.output_coordinate_repr == 'euclidean':
                pass
            else:
                raise NotImplementedError(f"Output coordinate representation {self.output_coordinate_repr} not implemented")
            noise = torch.randn_like(deltas)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=noise.device).long()
            noisy_action = self.noise_scheduler.add_noise(deltas, noise, timesteps)
            
            # Pad noisy_action with zeros to make the second dimension 12
            # Refer to https://github.com/real-stanford/diffusion_policy/issues/32#issuecomment-1834622174
            padding_size = 12 - noisy_action.size(1)
            if padding_size > 0:
                noisy_action_pad = F.pad(noisy_action, (0, 0, 0, padding_size))
            
            noise_pred = self.wp_predictor(sample=noisy_action_pad, timestep=timesteps, global_cond=dec_out)
            noise_pred = noise_pred[:, :self.len_traj_pred]
            alpha_cumprod = self.noise_scheduler.alphas_cumprod[timesteps].view(B, 1, 1)
            wp_pred = (noisy_action - noise_pred * (1 - alpha_cumprod).sqrt()) / alpha_cumprod.sqrt()
            if self.output_coordinate_repr == 'polar':
                distances = wp_pred[:, :, 0]
                angles = wp_pred[:, :, 1]
                dx = distances * torch.cos(angles)
                dy = distances * torch.sin(angles)
                wp_pred = torch.stack([dx, dy], dim=-1)
            wp_pred = torch.cumsum(wp_pred, dim=1)
            
            arrived_pres = self.arrive_predictor(dec_out).view(B, 1)
            
            return wp_pred, noise_pred, arrived_pres, noise
