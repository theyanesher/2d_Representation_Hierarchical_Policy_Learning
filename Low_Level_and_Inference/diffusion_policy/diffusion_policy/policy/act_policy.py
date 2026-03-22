"""
Action Chunking Transformer (ACT) Policy for ArticuBot.

Ported from lerobot's ACT implementation and adapted to ArticuBot's
BaseImagePolicy interface with LinearNormalizer.

Supports two vision backbones:
  - resnet18: torchvision ResNet18 with FrozenBatchNorm2d (fine-tuned)
  - dinov3: frozen DINOv3 ViT with its own ImageNet normalization

Reference: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
           https://arxiv.org/abs/2304.13705
"""

import math
from collections import deque
from itertools import chain
from typing import Callable, Dict

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class ACTImagePolicy(BaseImagePolicy):
    """ACT policy adapted to ArticuBot's BaseImagePolicy interface.

    Observations: 3 RGB cameras (cam0, cam1, cam2) + 10D state
    Actions: 10D (pos_delta + 6D rot_delta + gripper_delta)
    """

    def __init__(
        self,
        shape_meta: dict,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        n_obs_steps: int = 1,
        # Vision
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str = "ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation: bool = False,
        crop_shape: tuple = (240, 240),
        # Transformer
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        feedforward_activation: str = "relu",
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        pre_norm: bool = False,
        # VAE
        use_vae: bool = True,
        latent_dim: int = 32,
        n_vae_encoder_layers: int = 4,
        kl_weight: float = 10.0,
        # Training
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # Parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]

        # Identify image keys and state keys
        image_keys = []
        state_dim = 0
        for key, attr in obs_shape_meta.items():
            stype = attr.get("type", "low_dim")
            if stype == "rgb":
                image_keys.append(key)
            elif stype == "low_dim":
                state_dim = attr["shape"][0]

        # Store config
        self.image_keys = sorted(image_keys)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_vae = use_vae
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.dim_model = dim_model
        self.vision_backbone = vision_backbone

        # Build ACT model
        self.model = ACTModel(
            action_dim=action_dim,
            state_dim=state_dim,
            n_cameras=len(image_keys),
            chunk_size=chunk_size,
            vision_backbone=vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
            replace_final_stride_with_dilation=replace_final_stride_with_dilation,
            crop_shape=crop_shape,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            pre_norm=pre_norm,
            use_vae=use_vae,
            latent_dim=latent_dim,
            n_vae_encoder_layers=n_vae_encoder_layers,
            dropout=dropout,
        )

        self.normalizer = LinearNormalizer()
        self._action_queue = deque([], maxlen=n_action_steps)

        print(
            "ACT params: %e"
            % sum(p.numel() for p in self.model.parameters())
        )
        backbone = self.model.backbone
        print(
            "ACT backbone params: %e"
            % sum(p.numel() for p in backbone.parameters())
        )

    # ========= inference ============
    def reset(self):
        self._action_queue = deque([], maxlen=self.n_action_steps)

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps

        # Extract images: each (B, To, C, H, W) -> take last obs step -> (B, C, H, W)
        # Use raw (unnormalized) images — both backbones apply ImageNet norm internally
        images = []
        for key in self.image_keys:
            img = obs_dict[key][:, To - 1]
            images.append(img)

        # Extract state: (B, To, D) -> last obs step -> (B, D)
        state = nobs["state"][:, To - 1]  # (B, D)

        # Forward through model (no action input during inference)
        batch = {
            "observation.images": images,
            "observation.state": state,
        }
        actions_pred, _ = self.model(batch)  # (B, chunk_size, action_dim)

        # Unnormalize actions
        action_pred = self.normalizer["action"].unnormalize(actions_pred)

        # Return n_action_steps worth of actions
        action = action_pred[:, : self.n_action_steps]

        return {"action": action, "action_pred": action_pred}

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # Normalize observations and actions
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(
            batch["action"]
        )  # (B, T, Da)

        B = nactions.shape[0]
        To = self.n_obs_steps

        # Extract images: each (B, To, C, H, W) -> last obs step
        # Use raw (unnormalized) images — both backbones apply ImageNet norm internally
        images = []
        for key in self.image_keys:
            img = batch["obs"][key][:, To - 1]
            images.append(img)

        # Extract state
        state = nobs["state"][:, To - 1]  # (B, D)

        # Prepare target actions for the chunk
        # The dataset provides (B, horizon, Da). We use chunk_size steps.
        target_actions = nactions[:, : self.chunk_size]  # (B, chunk_size, Da)

        # Forward through model with actions (for VAE)
        model_batch = {
            "observation.images": images,
            "observation.state": state,
            "action": target_actions,
        }
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(model_batch)

        # L1 loss
        l1_loss = F.l1_loss(actions_hat, target_actions)

        loss_dict = {"l1_loss": l1_loss.item()}

        if self.use_vae and mu_hat is not None:
            # KL divergence: D_KL(q(z|x,a) || p(z)) where p(z) = N(0, I)
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.kl_weight
        else:
            loss = l1_loss

        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def get_optimizer(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        lr_backbone: float = 1e-5,
        betas: tuple = (0.95, 0.999),
    ) -> torch.optim.Optimizer:
        """Create optimizer with separate LR for backbone.

        For DINOv3: backbone is frozen, so only non-backbone params are optimized.
        For ResNet: backbone gets a separate (potentially lower) LR.
        """
        non_backbone_params = [
            p
            for n, p in self.named_parameters()
            if not n.startswith("model.backbone") and p.requires_grad
        ]
        backbone_params = [
            p
            for n, p in self.named_parameters()
            if n.startswith("model.backbone") and p.requires_grad
        ]

        param_groups = [{"params": non_backbone_params, "lr": learning_rate}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_backbone})

        return torch.optim.AdamW(
            param_groups, lr=learning_rate, weight_decay=weight_decay, betas=betas
        )

    def forward(self, batch):
        return self.compute_loss(batch)


# ============================================================
# ACT Model (core transformer architecture)
# ============================================================


class ACTModel(nn.Module):
    """Action Chunking Transformer model.

    Architecture:
      - Optional VAE encoder (during training): encodes [cls, state, actions] -> latent
      - Transformer encoder: processes [latent, state, image_features]
      - Transformer decoder: decodes action queries using encoder output
      - Action head: projects decoder output to action space
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        n_cameras: int,
        chunk_size: int,
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str = "ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation: bool = False,
        crop_shape: tuple = (240, 240),
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        feedforward_activation: str = "relu",
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        pre_norm: bool = False,
        use_vae: bool = True,
        latent_dim: int = 32,
        n_vae_encoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_vae = use_vae
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.dim_model = dim_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._use_dino = vision_backbone.startswith("dino")

        # VAE encoder
        if use_vae:
            self.vae_encoder = ACTEncoder(
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                feedforward_activation=feedforward_activation,
                n_layers=n_vae_encoder_layers,
                pre_norm=pre_norm,
                dropout=dropout,
            )
            self.vae_encoder_cls_embed = nn.Embedding(1, dim_model)
            if state_dim > 0:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    state_dim, dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(action_dim, dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(
                dim_model, latent_dim * 2
            )
            # Fixed sinusoidal positional embedding for VAE encoder input
            num_input_tokens = 1 + chunk_size  # cls + actions
            if state_dim > 0:
                num_input_tokens += 1  # + state
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_tokens, dim_model).unsqueeze(
                    0
                ),
            )

        # Vision backbone
        if self._use_dino:
            # DINOv3 backbone (frozen, handles its own normalization)
            self.backbone = DINOv3Backbone(
                model_name=vision_backbone,
                crop_shape=crop_shape,
            )
            backbone_feat_dim = self.backbone.hidden_dim
            # DINOv3 outputs patch tokens (sequence), not spatial feature maps
            # Linear projection from DINO hidden dim to dim_model
            self.encoder_img_feat_input_proj = nn.Linear(
                backbone_feat_dim, dim_model
            )
            # Learnable positional embeddings for patch tokens
            # DINOv3 ViT-B/16 with 224x224 crop -> 14x14 = 196 patches per camera
            n_patches_per_cam = self.backbone.n_patches
            self.encoder_cam_patch_pos_embed = nn.Embedding(
                n_patches_per_cam * n_cameras, dim_model
            )
        else:
            # ResNet backbone (fine-tuned)
            backbone_model = getattr(torchvision.models, vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    replace_final_stride_with_dilation,
                ],
                weights=pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )
            backbone_feat_dim = backbone_model.fc.in_features
            # Conv2d projection from ResNet feature map channels to dim_model
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_feat_dim, dim_model, kernel_size=1
            )
            # 2D sinusoidal positional embeddings for spatial feature maps
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
                dim_model // 2
            )
            # Image preprocessing: center crop + ImageNet normalization
            self.img_preprocess = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(crop_shape),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        # Transformer encoder and decoder
        self.encoder = ACTEncoder(
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            n_layers=n_encoder_layers,
            pre_norm=pre_norm,
            dropout=dropout,
        )
        self.decoder = ACTDecoder(
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            n_layers=n_decoder_layers,
            pre_norm=pre_norm,
            dropout=dropout,
        )

        # Encoder input projections
        self.encoder_latent_input_proj = nn.Linear(latent_dim, dim_model)
        if state_dim > 0:
            self.encoder_robot_state_input_proj = nn.Linear(state_dim, dim_model)

        # Encoder positional embeddings for 1D tokens (latent, state)
        n_1d_tokens = 1  # latent
        if state_dim > 0:
            n_1d_tokens += 1  # state
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, dim_model)

        # Decoder positional embeddings (learnable, like DETR object queries)
        self.decoder_pos_embed = nn.Embedding(chunk_size, dim_model)

        # Action regression head
        self.action_head = nn.Linear(dim_model, action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of transformer parameters."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, batch: dict
    ) -> tuple:
        """Forward pass through ACT model.

        Args:
            batch: dict with keys:
                - "observation.images": list of (B, C, H, W) tensors
                - "observation.state": (B, state_dim) tensor
                - "action" (optional, training only): (B, chunk_size, action_dim)

        Returns:
            actions: (B, chunk_size, action_dim)
            (mu, log_sigma_x2): latent distribution params, or (None, None)
        """
        batch_size = batch["observation.images"][0].shape[0]

        # ---- VAE encoder (training only) ----
        if self.use_vae and "action" in batch:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            vae_encoder_input = [cls_embed]

            if self.state_dim > 0:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(
                    batch["observation.state"]
                ).unsqueeze(1)
                vae_encoder_input.append(robot_state_embed)

            action_embed = self.vae_encoder_action_input_proj(
                batch["action"]
            )  # (B, S, D)
            vae_encoder_input.append(action_embed)
            vae_encoder_input = torch.cat(vae_encoder_input, dim=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            # Key padding mask: False = not padded
            n_prefix = 1 + (1 if self.state_dim > 0 else 0)  # cls + optional state
            cls_state_not_pad = torch.full(
                (batch_size, n_prefix),
                False,
                device=vae_encoder_input.device,
            )
            action_not_pad = torch.full(
                (batch_size, batch["action"].shape[1]),
                False,
                device=vae_encoder_input.device,
            )
            key_padding_mask = torch.cat([cls_state_not_pad, action_not_pad], dim=1)

            # Forward VAE encoder: (S, B, D) format
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # CLS token output: (B, D)

            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.latent_dim :]

            # Reparameterization trick
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.latent_dim],
                dtype=torch.float32,
                device=batch["observation.state"].device,
            )

        # ---- Transformer encoder ----
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        )

        if self.state_dim > 0:
            encoder_in_tokens.append(
                self.encoder_robot_state_input_proj(batch["observation.state"])
            )

        # Process camera images
        if self._use_dino:
            # DINOv3: frozen backbone outputs patch tokens (B, n_patches, hidden_dim)
            all_cam_features = []
            for img in batch["observation.images"]:
                patch_tokens = self.backbone(img)  # (B, n_patches, hidden_dim)
                patch_tokens = self.encoder_img_feat_input_proj(patch_tokens)  # (B, n_patches, dim_model)
                # Convert to (n_patches, B, dim_model) for transformer
                patch_tokens = patch_tokens.permute(1, 0, 2)
                all_cam_features.append(patch_tokens)

            all_cam_features = torch.cat(all_cam_features, dim=0)  # (total_patches, B, dim_model)
            # Learnable positional embeddings for all camera patches
            cam_pos_embed = self.encoder_cam_patch_pos_embed.weight.unsqueeze(1)  # (total_patches, 1, dim_model)

            encoder_in_tokens.extend(all_cam_features)
            encoder_in_pos_embed.extend(cam_pos_embed)
        else:
            # ResNet: backbone outputs spatial feature maps (B, C, H, W)
            all_cam_features = []
            all_cam_pos_embeds = []
            for img in batch["observation.images"]:
                img = self.img_preprocess(img)
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            encoder_in_tokens.extend(torch.cat(all_cam_features, dim=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, dim=0))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        # Forward encoder
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # ---- Transformer decoder ----
        decoder_in = torch.zeros(
            (self.chunk_size, batch_size, self.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # (S, B, D) -> (B, S, D)
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


# ============================================================
# DINOv3 Vision Backbone
# ============================================================


class DINOv3Backbone(nn.Module):
    """Frozen DINOv3 backbone for ACT.

    Handles its own ImageNet normalization internally, so raw [0,1] or [0,255]
    images should be passed in (not dataset-normalized images).

    Outputs patch tokens (B, n_patches, hidden_dim) per image.
    """

    # Map config names to HuggingFace model IDs
    MODEL_REGISTRY = {
        "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov2": "facebook/dinov2-base",
        "dinov2-base": "facebook/dinov2-base",
    }

    def __init__(
        self,
        model_name: str = "dinov3",
        crop_shape: tuple = (224, 224),
    ):
        super().__init__()
        from transformers import AutoModel

        # Resolve model name
        hf_model_id = self.MODEL_REGISTRY.get(model_name, model_name)

        self.model = AutoModel.from_pretrained(hf_model_id)
        self.model.requires_grad_(False)  # Freeze backbone

        self.hidden_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

        # Number of patches for the crop size
        # DINOv3 ViT-B/16: crop 224x224 -> 14x14 = 196 patches
        crop_h, crop_w = crop_shape
        self.n_patches = (crop_h // self.patch_size) * (crop_w // self.patch_size)

        # Image preprocessing: center crop + resize to ViT input + ImageNet normalize
        self.img_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(crop_shape),
            torchvision.transforms.Resize(
                (crop_h, crop_w),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Number of prefix tokens to skip (CLS + register tokens)
        # DINOv3 typically has 1 CLS + 4 register tokens = 5 prefix tokens
        # DINOv2 has 1 CLS token = 1 prefix token
        # We auto-detect based on output size
        self._n_prefix_tokens = None

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: (B, C, H, W) raw images in [0, 1] float range
                    (not dataset-normalized, DINOv3 handles its own normalization)

        Returns:
            (B, n_patches, hidden_dim) patch token features
        """
        # Apply center crop + ImageNet normalization
        images = self.img_preprocess(images)

        outputs = self.model(pixel_values=images)
        all_tokens = outputs.last_hidden_state  # (B, 1+n_reg+n_patches, hidden_dim)

        # Auto-detect prefix token count on first forward
        if self._n_prefix_tokens is None:
            self._n_prefix_tokens = all_tokens.shape[1] - self.n_patches

        # Skip CLS and register tokens, keep only patch tokens
        patch_tokens = all_tokens[:, self._n_prefix_tokens:]  # (B, n_patches, hidden_dim)

        return patch_tokens


# ============================================================
# Transformer components
# ============================================================


class ACTEncoder(nn.Module):
    """Stack of encoder layers with optional layer normalization."""

    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        n_layers: int,
        pre_norm: bool,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ACTEncoderLayer(
                    dim_model, n_heads, dim_feedforward, feedforward_activation,
                    pre_norm, dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim_model) if pre_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        pre_norm: bool,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(feedforward_activation)
        self.pre_norm = pre_norm

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    """Stack of decoder layers followed by normalization."""

    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        n_layers: int,
        pre_norm: bool,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ACTDecoderLayer(
                    dim_model, n_heads, dim_feedforward, feedforward_activation,
                    pre_norm, dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        pre_norm: bool,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(feedforward_activation)
        self.pre_norm = pre_norm

    def _maybe_add(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        # Self-attention
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self._maybe_add(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        # Cross-attention
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self._maybe_add(x, decoder_pos_embed),
            key=self._maybe_add(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        # Feed-forward
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


# ============================================================
# Positional embeddings
# ============================================================


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / dimension)
            for hid_j in range(dimension)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(num_positions)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings for image feature maps."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            (1, C, H, W) positional embeddings
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2
            * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2)
            / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack(
            (x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_embed_y = torch.stack(
            (y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(
            0, 3, 1, 2
        )  # (1, C, H, W)

        return pos_embed


# ============================================================
# Utilities
# ============================================================


def _get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
