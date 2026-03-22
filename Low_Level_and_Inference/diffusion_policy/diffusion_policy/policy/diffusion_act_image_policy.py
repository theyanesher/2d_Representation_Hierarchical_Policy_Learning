"""
Diffusion ACT Policy — Transformer Encoder-Decoder as Diffusion Denoiser.

Replaces the UNet in diffusion policy with ACT's transformer encoder-decoder
architecture, using DiT-style adaptive LayerNorm for timestep conditioning.
No VAE — diffusion provides stochasticity.

Architecture:
  - Vision backbone (ResNet18 or DINOv3) encodes images into tokens
  - State is projected to a token
  - AdaLN Encoder processes [state_token, image_tokens] conditioned on timestep
  - AdaLN Decoder cross-attends noisy action queries to encoder output
  - Output head predicts noise (epsilon)
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
from diffusion_policy.model.vision.rope_3d import (
    RoPE3DMultiheadAttention,
    patches_to_3d_positions,
)
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Reuse from act_policy
from diffusion_policy.policy.act_policy import (
    DINOv3Backbone,
    ACTSinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding,
    _get_activation_fn,
)


# ============================================================
# Adaptive LayerNorm (DiT-style)
# ============================================================


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm: scale and shift are predicted from a conditioning vector."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, dim * 2)
        # Initialize to identity transform (scale=1, shift=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (..., dim)
            cond: (..., cond_dim) — must be broadcastable to x
        Returns:
            (..., dim)
        """
        scale_shift = self.linear(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# ============================================================
# AdaLN Transformer Layers
# ============================================================


class AdaLNEncoderLayer(nn.Module):
    """Encoder layer with adaptive LayerNorm conditioned on timestep."""

    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        cond_dim: int,
        dropout: float,
        use_3d_rope: bool = False,
    ):
        super().__init__()
        self.use_3d_rope = use_3d_rope
        if use_3d_rope:
            self.self_attn = RoPE3DMultiheadAttention(dim_model, n_heads, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = AdaLayerNorm(dim_model, cond_dim)
        self.norm2 = AdaLayerNorm(dim_model, cond_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(feedforward_activation)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        pos_embed: Tensor | None = None,
        positions_3d: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (S, B, D) sequence-first
            cond: (1, B, cond_dim) or (S, B, cond_dim) — timestep conditioning
            pos_embed: (S, B, D) or (S, 1, D) — used when use_3d_rope=False
            positions_3d: (B, S, 3) — used when use_3d_rope=True
        """
        # Pre-norm with AdaLN
        skip = x
        x = self.norm1(x, cond)
        if self.use_3d_rope:
            # RoPE handles positional info inside attention
            x = self.self_attn(x, x, x, positions_3d=positions_3d, key_padding_mask=key_padding_mask)[0]
        else:
            q = k = x if pos_embed is None else x + pos_embed
            x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)

        # FFN with AdaLN
        skip = x
        x = self.norm2(x, cond)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        return x


class AdaLNDecoderLayer(nn.Module):
    """Decoder layer with adaptive LayerNorm conditioned on timestep."""

    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        cond_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = AdaLayerNorm(dim_model, cond_dim)
        self.norm2 = AdaLayerNorm(dim_model, cond_dim)
        self.norm3 = AdaLayerNorm(dim_model, cond_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(feedforward_activation)

    def _maybe_add(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        cond: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (S_dec, B, D) decoder input (noisy action tokens)
            encoder_out: (S_enc, B, D) encoder output
            cond: (1, B, cond_dim) timestep conditioning
            decoder_pos_embed: (S_dec, 1, D) or (S_dec, B, D)
            encoder_pos_embed: (S_enc, 1, D) or (S_enc, B, D)
        """
        # Self-attention
        skip = x
        x = self.norm1(x, cond)
        q = k = self._maybe_add(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        # Cross-attention
        skip = x
        x = self.norm2(x, cond)
        x = self.multihead_attn(
            query=self._maybe_add(x, decoder_pos_embed),
            key=self._maybe_add(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        # Feed-forward
        skip = x
        x = self.norm3(x, cond)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        return x


class AdaLNEncoder(nn.Module):
    """Stack of AdaLN encoder layers."""

    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        cond_dim: int,
        n_layers: int,
        dropout: float,
        use_3d_rope: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AdaLNEncoderLayer(
                    dim_model, n_heads, dim_feedforward,
                    feedforward_activation, cond_dim, dropout,
                    use_3d_rope=use_3d_rope,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = AdaLayerNorm(dim_model, cond_dim)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        pos_embed: Tensor | None = None,
        positions_3d: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, cond, pos_embed=pos_embed, positions_3d=positions_3d, key_padding_mask=key_padding_mask)
        x = self.final_norm(x, cond)
        return x


class AdaLNDecoder(nn.Module):
    """Stack of AdaLN decoder layers."""

    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        cond_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AdaLNDecoderLayer(
                    dim_model, n_heads, dim_feedforward,
                    feedforward_activation, cond_dim, dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = AdaLayerNorm(dim_model, cond_dim)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        cond: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, cond,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        x = self.final_norm(x, cond)
        return x


# ============================================================
# Diffusion ACT Model (denoising network)
# ============================================================


class DiffusionACTModel(nn.Module):
    """Transformer encoder-decoder denoiser for diffusion policy.

    Replaces the 1D ConditionalUnet with ACT's transformer architecture,
    using DiT-style adaptive LayerNorm for timestep conditioning.

    Forward: (noisy_actions, timesteps, images, state) → predicted_noise
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
        dropout: float = 0.1,
        use_3d_rope: bool = False,
        token_dropout: float = 0.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.dim_model = dim_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_3d_rope = use_3d_rope
        self.token_dropout = token_dropout
        self._use_dino = vision_backbone.startswith("dino")
        self.crop_shape = crop_shape

        if use_3d_rope:
            head_dim = dim_model // n_heads
            assert head_dim % 6 == 0, (
                f"3D RoPE requires (dim_model // n_heads) % 6 == 0, "
                f"got dim_model={dim_model}, n_heads={n_heads}, head_dim={head_dim}"
            )

        # ---- Timestep embedding: sinusoidal → MLP ----
        self.timestep_embed_dim = dim_model
        # Sinusoidal embedding for diffusion timesteps
        self.register_buffer(
            "_timestep_sinusoidal",
            self._build_sinusoidal_embedding(1000, dim_model),
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.SiLU(),
            nn.Linear(dim_model * 4, dim_model),
        )

        # ---- Vision backbone (identical to ACTModel) ----
        if self._use_dino:
            self.backbone = DINOv3Backbone(
                model_name=vision_backbone,
                crop_shape=crop_shape,
            )
            backbone_feat_dim = self.backbone.hidden_dim
            self.encoder_img_feat_input_proj = nn.Linear(
                backbone_feat_dim, dim_model
            )
            n_patches_per_cam = self.backbone.n_patches
            self._dino_patch_grid = (
                crop_shape[0] // self.backbone.patch_size,
                crop_shape[1] // self.backbone.patch_size,
            )
            if not use_3d_rope:
                self.encoder_cam_patch_pos_embed = ACTSinusoidalPositionEmbedding2d(
                    dim_model // 2
                )
        else:
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
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_feat_dim, dim_model, kernel_size=1
            )
            if not use_3d_rope:
                self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
                    dim_model // 2
                )
            self.img_preprocess = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(crop_shape),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        # ---- Encoder input projections ----
        if state_dim > 0:
            self.encoder_robot_state_input_proj = nn.Linear(state_dim, dim_model)

        # Positional embeddings for 1D tokens (state only — no latent in diffusion)
        n_1d_tokens = 1 if state_dim > 0 else 0
        if n_1d_tokens > 0 and not use_3d_rope:
            self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, dim_model)

        # ---- AdaLN Encoder ----
        self.encoder = AdaLNEncoder(
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            cond_dim=dim_model,
            n_layers=n_encoder_layers,
            dropout=dropout,
            use_3d_rope=use_3d_rope,
        )

        # ---- Noisy action input projection ----
        self.noisy_action_input_proj = nn.Linear(action_dim, dim_model)

        # Decoder positional embeddings (learnable, like DETR object queries)
        self.decoder_pos_embed = nn.Embedding(chunk_size, dim_model)

        # ---- AdaLN Decoder ----
        self.decoder = AdaLNDecoder(
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            cond_dim=dim_model,
            n_layers=n_decoder_layers,
            dropout=dropout,
        )

        # ---- Output head ----
        self.action_head = nn.Linear(dim_model, action_dim)

        self._reset_parameters()

    @staticmethod
    def _build_sinusoidal_embedding(max_steps: int, dim: int) -> Tensor:
        """Build sinusoidal embedding table for diffusion timesteps."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        pos = torch.arange(max_steps, dtype=torch.float32)
        emb = pos[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (max_steps, dim)

    def _reset_parameters(self):
        """Xavier-uniform initialization of transformer parameters."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_timestep_embedding(self, timesteps: Tensor) -> Tensor:
        """Get timestep conditioning vector.

        Args:
            timesteps: (B,) integer timesteps
        Returns:
            (B, dim_model) conditioning vector
        """
        # Look up sinusoidal embeddings
        t_emb = self._timestep_sinusoidal[timesteps.long()]  # (B, dim_model)
        # Pass through MLP
        return self.timestep_mlp(t_emb)  # (B, dim_model)

    def forward(
        self,
        noisy_actions: Tensor,
        timesteps: Tensor,
        images: list,
        state: Tensor,
        depths: list | None = None,
        intrinsics: list | None = None,
        extrinsics: list | None = None,
        raw_state: Tensor | None = None,
    ) -> Tensor:
        """Forward pass: predict noise given noisy actions and observations.

        Args:
            noisy_actions: (B, chunk_size, action_dim) noisy action sequence
            timesteps: (B,) integer diffusion timesteps
            images: list of (B, C, H, W) per camera
            state: (B, state_dim) normalized robot state
            depths: list of (B, 1, H, W) per camera (required when use_3d_rope=True)
            intrinsics: list of (B, 3, 3) per camera (required when use_3d_rope=True)
            extrinsics: list of (B, 4, 4) w2c per camera (required when use_3d_rope=True)
            raw_state: (B, state_dim) unnormalized state (required when use_3d_rope=True;
                       first 3 elements are EEF world position)

        Returns:
            predicted_noise: (B, chunk_size, action_dim)
        """
        batch_size = noisy_actions.shape[0]

        # ---- Timestep conditioning ----
        cond = self._get_timestep_embedding(timesteps)  # (B, dim_model)
        # Reshape for transformer: (1, B, dim_model) to broadcast across sequence
        cond_seq = cond.unsqueeze(0)  # (1, B, dim_model)

        # ---- Build encoder input tokens ----
        encoder_in_tokens = []
        encoder_in_pos_embed = []

        if self.state_dim > 0:
            state_token = self.encoder_robot_state_input_proj(state)  # (B, dim_model)
            encoder_in_tokens.append(state_token)
            if not self.use_3d_rope:
                encoder_in_pos_embed.append(
                    self.encoder_1d_feature_pos_embed.weight[0].unsqueeze(0)  # (1, dim_model)
                )

        # Process camera images
        encoder_pos = None  # will be set for non-rope path
        positions_3d = None  # will be set for rope path

        if self._use_dino:
            all_cam_features = []
            all_cam_pos_embeds = []
            gh, gw = self._dino_patch_grid
            for img in images:
                patch_tokens = self.backbone(img)  # (B, n_patches, hidden_dim)
                patch_tokens = self.encoder_img_feat_input_proj(patch_tokens)  # (B, n_patches, dim_model)
                if not self.use_3d_rope:
                    # Reshape to (B, D, H, W) for sinusoidal pos embed, then flatten back
                    feat_map = patch_tokens.permute(0, 2, 1).reshape(
                        patch_tokens.shape[0], -1, gh, gw
                    )  # (B, dim_model, gh, gw)
                    cam_pos_embed = self.encoder_cam_patch_pos_embed(feat_map).to(
                        dtype=patch_tokens.dtype
                    )  # (1, dim_model, gh, gw)
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    all_cam_pos_embeds.append(cam_pos_embed)
                patch_tokens = patch_tokens.permute(1, 0, 2)  # (n_patches, B, dim_model)
                all_cam_features.append(patch_tokens)

            all_cam_features = torch.cat(all_cam_features, dim=0)  # (total_patches, B, dim_model)

            # Convert 1D tokens to (S, B, D) format
            encoder_in_tokens_stacked = []
            for tok in encoder_in_tokens:
                encoder_in_tokens_stacked.append(tok.unsqueeze(0))  # (1, B, D)

            if encoder_in_tokens_stacked:
                tokens_1d = torch.cat(encoder_in_tokens_stacked, dim=0)  # (n_1d, B, D)
                encoder_in = torch.cat([tokens_1d, all_cam_features], dim=0)
            else:
                encoder_in = all_cam_features

            if self.use_3d_rope:
                positions_3d = patches_to_3d_positions(
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    eef_pos=raw_state[:, :3],
                    patch_size=self.backbone.patch_size,
                    crop_shape=self.crop_shape,
                    use_dino=True,
                )
            else:
                cam_pos_embed = torch.cat(all_cam_pos_embeds, dim=0)  # (total_patches, 1, dim_model)
                if encoder_in_tokens_stacked:
                    pos_1d = self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)  # (n_1d, 1, D)
                    encoder_pos = torch.cat([pos_1d, cam_pos_embed], dim=0)
                else:
                    encoder_pos = cam_pos_embed
        else:
            all_cam_features = []
            all_cam_pos_embeds = []
            resnet_feat_shape = None
            for img in images:
                img = self.img_preprocess(img)
                cam_features = self.backbone(img)["feature_map"]
                if resnet_feat_shape is None:
                    resnet_feat_shape = (cam_features.shape[2], cam_features.shape[3])
                if not self.use_3d_rope:
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                        dtype=cam_features.dtype
                    )
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    all_cam_pos_embeds.append(cam_pos_embed)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                all_cam_features.append(cam_features)

            all_cam_features = torch.cat(all_cam_features, dim=0)  # (total_hw, B, D)

            # Build full encoder input
            encoder_in_tokens_stacked = []
            for tok in encoder_in_tokens:
                encoder_in_tokens_stacked.append(tok.unsqueeze(0))  # (1, B, D)

            if encoder_in_tokens_stacked:
                tokens_1d = torch.cat(encoder_in_tokens_stacked, dim=0)  # (n_1d, B, D)
                encoder_in = torch.cat([tokens_1d, all_cam_features], dim=0)
            else:
                encoder_in = all_cam_features

            if self.use_3d_rope:
                positions_3d = patches_to_3d_positions(
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    eef_pos=raw_state[:, :3],
                    feat_shape=resnet_feat_shape,
                    crop_shape=self.crop_shape,
                    use_dino=False,
                )
            else:
                all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, dim=0)  # (total_hw, B, D)
                if encoder_in_tokens_stacked:
                    pos_1d = self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)  # (n_1d, 1, D)
                    all_cam_pos_embeds = all_cam_pos_embeds.expand(-1, batch_size, -1)
                    pos_1d = pos_1d.expand(-1, batch_size, -1)
                    encoder_pos = torch.cat([pos_1d, all_cam_pos_embeds], dim=0)
                else:
                    encoder_pos = all_cam_pos_embeds

        # ---- Encoder forward ----
        encoder_out = self.encoder(
            encoder_in, cond_seq,
            pos_embed=encoder_pos,
            positions_3d=positions_3d,
        )

        # ---- Token dropout on encoder output (regularization) ----
        # During training, randomly remove a fraction of encoder tokens so the
        # decoder cannot over-rely on any single token (MAE-style).
        if self.training and self.token_dropout > 0.0:
            S, B, D = encoder_out.shape
            # Sample a per-token keep mask (shared across batch for simplicity)
            keep_mask = torch.rand(S, device=encoder_out.device) > self.token_dropout  # (S,)
            encoder_out = encoder_out[keep_mask]  # (S', B, D)
            if encoder_pos is not None:
                encoder_pos = encoder_pos[keep_mask]

        # ---- Decoder forward ----
        # Project noisy actions to transformer dim
        noisy_action_tokens = self.noisy_action_input_proj(noisy_actions)  # (B, chunk_size, dim_model)
        noisy_action_tokens = noisy_action_tokens.permute(1, 0, 2)  # (chunk_size, B, dim_model)

        # With 3D RoPE, encoder output already has positional info baked in —
        # decoder cross-attention does not use encoder_pos_embed.
        decoder_out = self.decoder(
            noisy_action_tokens,
            encoder_out,
            cond_seq,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            encoder_pos_embed=encoder_pos if not self.use_3d_rope else None,
        )

        # (chunk_size, B, dim_model) -> (B, chunk_size, dim_model)
        decoder_out = decoder_out.transpose(0, 1)

        # Predict noise
        noise_pred = self.action_head(decoder_out)  # (B, chunk_size, action_dim)
        return noise_pred


# ============================================================
# Diffusion ACT Image Policy (BaseImagePolicy wrapper)
# ============================================================


class DiffusionACTImagePolicy(BaseImagePolicy):
    """Diffusion policy using ACT transformer as denoiser.

    Mirrors the DiffusionUnetHybridImagePolicy interface:
    set_normalizer(), predict_action(), compute_loss(), get_optimizer().

    Uses DDPMScheduler from diffusers for the diffusion process.
    """

    def __init__(
        self,
        shape_meta: dict,
        # Diffusion
        noise_scheduler,
        num_inference_steps: int = 100,
        # Architecture
        chunk_size: int = 16,
        n_action_steps: int = 8,
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
        dropout: float = 0.1,
        token_dropout: float = 0.0,
        # 3D RoPE
        use_3d_rope: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]

        # Identify image, depth, intrinsic, extrinsic, and state keys
        image_keys = []
        depth_keys = []
        intrinsic_keys = []
        extrinsic_keys = []
        state_dim = 0
        for key, attr in obs_shape_meta.items():
            stype = attr.get("type", "low_dim")
            if stype == "rgb":
                image_keys.append(key)
            elif stype == "depth":
                depth_keys.append(key)
            elif stype == "intrinsic":
                intrinsic_keys.append(key)
            elif stype == "extrinsic":
                extrinsic_keys.append(key)
            elif stype == "low_dim":
                state_dim = attr["shape"][0]

        # Store config
        self.image_keys = sorted(image_keys)
        self.depth_keys = sorted(depth_keys)
        self.intrinsic_keys = sorted(intrinsic_keys)
        self.extrinsic_keys = sorted(extrinsic_keys)
        self.use_3d_rope = use_3d_rope
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.vision_backbone = vision_backbone

        if use_3d_rope:
            assert len(self.depth_keys) == len(self.image_keys), (
                f"3D RoPE requires one depth per camera: "
                f"{len(self.depth_keys)} depths vs {len(self.image_keys)} images"
            )
            assert len(self.intrinsic_keys) == len(self.image_keys), (
                f"3D RoPE requires one intrinsic per camera: "
                f"{len(self.intrinsic_keys)} intrinsics vs {len(self.image_keys)} images"
            )
            assert len(self.extrinsic_keys) == len(self.image_keys), (
                f"3D RoPE requires one extrinsic per camera: "
                f"{len(self.extrinsic_keys)} extrinsics vs {len(self.image_keys)} images"
            )

        # Noise scheduler (DDPMScheduler from diffusers)
        self.noise_scheduler = noise_scheduler

        # Build denoising model
        self.model = DiffusionACTModel(
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
            dropout=dropout,
            use_3d_rope=use_3d_rope,
            token_dropout=token_dropout,
        )

        self.normalizer = LinearNormalizer()

        print(
            "DiffusionACT params: %e"
            % sum(p.numel() for p in self.model.parameters())
        )
        backbone = self.model.backbone
        print(
            "DiffusionACT backbone params: %e"
            % sum(p.numel() for p in backbone.parameters())
        )

    # ========= helpers ============
    def _extract_camera_data(self, obs_dict, t_idx):
        """Extract depth, intrinsic, extrinsic lists from obs_dict at time t_idx.

        Returns (depths, intrinsics, extrinsics) or (None, None, None).
        """
        if not self.use_3d_rope:
            return None, None, None
        depths = [obs_dict[k][:, t_idx] for k in self.depth_keys]
        intrinsics = [obs_dict[k][:, t_idx] for k in self.intrinsic_keys]
        extrinsics = [obs_dict[k][:, t_idx] for k in self.extrinsic_keys]
        return depths, intrinsics, extrinsics

    # ========= inference ============
    def conditional_sample(
        self,
        images: list,
        state: Tensor,
        depths: list | None = None,
        intrinsics: list | None = None,
        extrinsics: list | None = None,
        raw_state: Tensor | None = None,
        generator=None,
    ) -> Tensor:
        """Run DDPM/DDIM reverse diffusion process.

        Args:
            images: list of (B, C, H, W) per camera
            state: (B, state_dim) normalized robot state
            depths: list of (B, 1, H, W) per camera (for 3D RoPE)
            intrinsics: list of (B, 3, 3) per camera (for 3D RoPE)
            extrinsics: list of (B, 4, 4) w2c per camera (for 3D RoPE)
            raw_state: (B, state_dim) unnormalized state (for 3D RoPE EEF pos)
            generator: optional torch.Generator for reproducibility

        Returns:
            (B, chunk_size, action_dim) denoised action sequence
        """
        B = state.shape[0]
        device = state.device
        dtype = state.dtype

        # Start from pure noise
        noisy_actions = torch.randn(
            (B, self.chunk_size, self.action_dim),
            device=device, dtype=dtype, generator=generator,
        )

        # Set up scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            timesteps = t.expand(B).to(device)
            # Predict noise
            noise_pred = self.model(
                noisy_actions, timesteps, images, state,
                depths=depths, intrinsics=intrinsics, extrinsics=extrinsics,
                raw_state=raw_state,
            )
            # Scheduler step
            noisy_actions = self.noise_scheduler.step(
                noise_pred, t, noisy_actions, generator=generator,
            ).prev_sample

        return noisy_actions

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        To = self.n_obs_steps

        # Extract images: raw (unnormalized) — backbone handles ImageNet norm
        images = []
        for key in self.image_keys:
            img = obs_dict[key][:, To - 1]
            images.append(img)

        # Extract state: normalized for encoder, raw for 3D RoPE EEF position
        state = nobs["state"][:, To - 1]  # (B, D)
        raw_state = obs_dict["state"][:, To - 1] if self.use_3d_rope else None

        # Extract camera data for 3D RoPE
        depths, intrinsics, extrinsics = self._extract_camera_data(obs_dict, To - 1)

        # Run reverse diffusion
        action_pred = self.conditional_sample(
            images, state,
            depths=depths, intrinsics=intrinsics, extrinsics=extrinsics,
            raw_state=raw_state,
        )  # (B, chunk_size, action_dim)

        # Unnormalize actions
        action_pred = self.normalizer["action"].unnormalize(action_pred)

        # Return n_action_steps worth of actions
        action = action_pred[:, : self.n_action_steps]

        return {"action": action, "action_pred": action_pred}

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute diffusion training loss (MSE on noise prediction).

        Returns:
            loss: scalar tensor (single value, compatible with diffusion workspace)
        """
        # Normalize observations and actions
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])  # (B, T, Da)

        B = nactions.shape[0]
        To = self.n_obs_steps

        # Extract images: raw (unnormalized)
        images = []
        for key in self.image_keys:
            img = batch["obs"][key][:, To - 1]
            images.append(img)

        # Extract state: normalized for encoder, raw for 3D RoPE EEF position
        state = nobs["state"][:, To - 1]  # (B, D)
        raw_state = batch["obs"]["state"][:, To - 1] if self.use_3d_rope else None

        # Extract camera data for 3D RoPE
        depths, intrinsics, extrinsics = self._extract_camera_data(batch["obs"], To - 1)

        # Target actions for the chunk
        target_actions = nactions[:, : self.chunk_size]  # (B, chunk_size, Da)

        # Sample noise
        noise = torch.randn_like(target_actions)

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=target_actions.device,
        ).long()

        # Add noise to actions (forward diffusion)
        noisy_actions = self.noise_scheduler.add_noise(
            target_actions, noise, timesteps
        )

        # Predict noise
        noise_pred = self.model(
            noisy_actions, timesteps, images, state,
            depths=depths, intrinsics=intrinsics, extrinsics=extrinsics,
            raw_state=raw_state,
        )

        # MSE loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def get_optimizer(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        lr_backbone: float = 1e-5,
        betas: tuple = (0.95, 0.999),
    ) -> torch.optim.Optimizer:
        """Create optimizer with separate LR for backbone."""
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
