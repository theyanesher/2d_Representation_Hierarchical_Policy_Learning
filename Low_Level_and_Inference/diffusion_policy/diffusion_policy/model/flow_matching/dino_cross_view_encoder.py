"""
Dino Cross-View DINOv2 Encoder
=============================
A self-contained multi-view visual encoder based on Depth Anything 3's
input-adaptive cross-view self-attention mechanism.

Architecture (from DA3 paper, Figure 2):
    1. Image Patch Embed: each view → (N_patches, embed_dim) tokens
    2. Single DINOv2 Transformer with alternating attention:
       - Layers 0 .. alt_start-1:  within-view self-attention (local)
       - Layers alt_start .. end:  even = local, odd = cross-view (global)
    3. Camera tokens (learnable ref/src) injected at alt_start layer
    4. Optional: CameraEnc encodes extrinsics+intrinsics → camera tokens

Usage in the policy:
    visual_encoder_type: dino_crossview
    visual_encoder_cfg:
        backbone: vitb         # vits | vitb | vitl
        pretrained: true
        alt_start: 4
        cat_token: true        # concat local+global features
        crop_shape: [224, 224]

Adapted from: https://github.com/DepthAnything/Depth-Anything-3
Original DINOv2: https://github.com/facebookresearch/dinov2
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import torchvision.transforms.v2.functional as tvf
from diffusion_policy.model.vision.image_augmentations import (
    ImageAugmentor,
    RandomResizedCropAug,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Geometry / Camera Utilities (from DA3)
# ============================================================================

def affine_inverse(A: Tensor) -> Tensor:
    """Invert a rigid-body transform [..., 4, 4]."""
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def _sqrt_positive_part(x: Tensor) -> Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: Tensor) -> Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat(matrix: Tensor) -> Tensor:
    """Rotation matrix → quaternion (xyzw, scalar-last)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack([
            1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,
        ], dim=-1)
    )
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]
    return standardize_quaternion(out)


def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw):
    """Convert camera extrinsics (c2w, Bx Sx3x4) + intrinsics → 9D encoding."""
    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]
    quat = mat_to_quat(R)
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    return torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


# ============================================================================
# ViT Layer Components (from DA3's DINOv2)
# ============================================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


try:
    from xformers.ops import SwiGLU
    _XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    _XFORMERS_AVAILABLE = False


class SwiGLUFFNFused(SwiGLU):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features, hidden_features=hidden_features,
                         out_features=out_features, bias=bias)


class PositionGetter:
    """Generates and caches 2D spatial positions for patches."""
    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int,
                 device: torch.device) -> Tensor:
        if (height, width) not in self.position_cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y, x)
            self.position_cache[height, width] = positions
        cached = self.position_cache[height, width]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding."""
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[Tensor, Tensor]] = {}

    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: Tensor) -> Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )
        v_feat, h_feat = tokens.chunk(2, dim=-1)
        v_feat = self._apply_1d_rope(v_feat, positions[..., 0], cos_comp, sin_comp)
        h_feat = self._apply_1d_rope(h_feat, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((v_feat, h_feat), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0, norm_layer=nn.LayerNorm,
                 qk_norm=False, rope=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            attn_mask=(
                attn_mask[:, None].repeat(1, self.num_heads, 1, 1)
                if attn_mask is not None else None
            ),
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio=0.0, pos=None):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    if pos is not None:
        residual = residual_func(x_subset, pos=pos[brange])
    else:
        residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, proj_bias=True,
                 ffn_bias=True, drop=0.0, attn_drop=0.0, init_values=None,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_class=Attention, ffn_layer=Mlp, qk_norm=False, rope=None,
                 ln_eps=1e-6):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=ln_eps)
        self.attn = attn_class(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm, rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=ln_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        def attn_residual_func(x, pos=None, attn_mask=None):
            return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = _drop_add_residual_stochastic_depth(
                x, residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio, pos=pos,
            )
            x = _drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, attn_mask=attn_mask))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
            x = x + ffn_residual_func(x)
        return x


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) → (B,N,D)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten_embedding=True):
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0 and W % patch_W == 0
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


# ============================================================================
# DinoVisionTransformer with Cross-View Self-Attention
# ============================================================================

class DinoVisionTransformer(nn.Module):
    """
    DINOv2 ViT with DA3's input-adaptive cross-view self-attention.

    Key mechanism (``process_attention``):
        - "local" attention:  (B*S, N, C) — within-view
        - "global" attention: (B, S*N, C) — cross-view

    Layers 0 .. alt_start-1: all local.
    Layers alt_start .. end: even = local, odd = global (cross-view).
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        cat_token=True,
        num_camera_tokens=2,
    ):
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = embed_dim
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, num_camera_tokens, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens else None
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer_cls = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer_cls = SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_layer_cls = lambda *a, **kw: nn.Identity()
        else:
            raise NotImplementedError(f"Unknown ffn_layer: {ffn_layer}")

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None

        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                ffn_layer=ffn_layer_cls, init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic", antialias=self.interpolate_antialias, **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        cls_token = self.cls_token.expand(B, S, -1).reshape(B * S, -1, self.embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((
                x[:, :1],
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ), dim=1)
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None):
        """Reshape tokens for local (within-view) or global (cross-view) attention."""
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        x = block(x, pos=pos, attn_mask=attn_mask)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x

    def imagenet_norm(self, x):
        B, S, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return x.reshape(B, S, C, H, W)

    def forward_features(
        self,
        x: Tensor,
        out_layers: Union[int, Sequence] = 1,
        cam_token: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Forward pass through the transformer.

        Args:
            x: (B, S, C, H, W) multi-view images. in [0,1]
            out_layers: layer indices to extract features from (or int n for last n).
            cam_token: (B, S, D) optional camera tokens.
            attn_mask: optional attention mask for global attention.

        Returns:
            List of (B, S, N_patches, token_dim) feature tensors for each out_layer.
            token_dim = embed_dim * 2 if cat_token else embed_dim.
        """
        x = self.imagenet_norm(x)
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output = []
        total_block_len = len(self.blocks)
        blocks_to_take = (
            range(total_block_len - out_layers, total_block_len)
            if isinstance(out_layers, int) else out_layers
        )
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
        local_x = x

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            # Inject camera tokens at alt_start
            if self.alt_start != -1 and i == self.alt_start:
                if cam_token is not None:
                    x[:, :, 0] = cam_token
                else:
                    cam_tok = self.camera_token[:, :S].expand(B, -1, -1)
                    x[:, :, 0] = cam_tok

            # Alternating attention: global on odd layers after alt_start
            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(x, blk, "global", pos=g_pos, attn_mask=attn_mask)
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append(out_x)

        # Apply norm and strip CLS + register tokens
        processed = []
        for out in output:
            if out.shape[-1] == self.embed_dim:
                out = self.norm(out)
            elif out.shape[-1] == self.embed_dim * 2:
                out = torch.cat([
                    out[..., :self.embed_dim],
                    self.norm(out[..., self.embed_dim:]),
                ], dim=-1)
            # Remove CLS token and register tokens
            skip = 1 + self.num_register_tokens
            out = out[..., skip:, :]
            processed.append(out)

        return processed


# ============================================================================
# Camera Encoder
# ============================================================================

class CameraEnc(nn.Module):
    """
    Encodes camera extrinsics + intrinsics into tokens that replace the
    CLS token at alt_start in the DinoVisionTransformer.

    Input:  extrinsics (B, S, 4, 4) w2c, intrinsics (B, S, 3, 3)
    Output: (B, S, embed_dim)
    """

    def __init__(self, dim_out=768, dim_in=9, trunk_depth=4, num_heads=12,
                 mlp_ratio=4, init_values=0.01):
        super().__init__()
        self.pose_branch = Mlp(
            in_features=dim_in, hidden_features=dim_out // 2,
            out_features=dim_out, drop=0,
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk = nn.Sequential(*[
            Block(
                dim=dim_out, num_heads=num_heads, mlp_ratio=mlp_ratio,
                init_values=init_values, qkv_bias=True,
            )
            for _ in range(trunk_depth)
        ])
        self.trunk_norm = nn.LayerNorm(dim_out)

    def forward(self, extrinsics, intrinsics, image_size_hw):
        """
        Args:
            extrinsics: (B, S, 4, 4) world-to-camera transforms.
            intrinsics: (B, S, 3, 3) camera intrinsic matrices.
            image_size_hw: (H, W) tuple.

        Returns:
            (B, S, dim_out) camera tokens.
        """
        c2ws = affine_inverse(extrinsics)
        pose_encoding = extri_intri_to_pose_encoding(c2ws, intrinsics, image_size_hw)
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens


# ============================================================================
# Weight Loading
# ============================================================================

_DINOV2_HUB_MODELS = {
    "vits": "dinov2_vits14",
    "vitb": "dinov2_vitb14",
    "vitl": "dinov2_vitl14",
    "vitg": "dinov2_vitg14",
}

_VIT_CONFIGS = {
    "vits": dict(embed_dim=384,  depth=12, num_heads=6,  ffn_layer="mlp"),
    "vitb": dict(embed_dim=768,  depth=12, num_heads=12, ffn_layer="mlp"),
    "vitl": dict(embed_dim=1024, depth=24, num_heads=16, ffn_layer="mlp"),
    "vitg": dict(embed_dim=1536, depth=40, num_heads=24, ffn_layer="swiglufused"),
}


def load_pretrained_dinov2(model: DinoVisionTransformer, backbone: str = "vitb"):
    """
    Load pretrained DINOv2 weights from torch.hub into a DinoVisionTransformer.

    The hub model (facebookresearch/dinov2) uses the same DinoVisionTransformer
    architecture, so state dict keys match directly.  New params like
    ``camera_token`` will be left at their random init (reported as missing).

    Args:
        model: Target model.
        backbone: "vits", "vitb", "vitl", "vitg".
    """
    hub_name = _DINOV2_HUB_MODELS[backbone]
    pretrained = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=True)
    sd = pretrained.state_dict()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded DINOv2 from torch.hub ({hub_name}). "
                f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        logger.info(f"  Missing keys (expected for new params): {missing}")


# ============================================================================
# DinoCrossViewTokenEncoder — VisualTokenEncoder interface
# ============================================================================

class DinoCrossViewTokenEncoder(nn.Module):
    """
    Multi-view DINOv2 encoder with DA3's cross-view self-attention.

    Self-contained: owns CropRandomizer, linear projector, and positional
    embeddings (camera + temporal).  The policy calls ``encoder.encode(nobs)``
    and gets back (B, n_obs_steps * n_cams * N_tokens, embed_dim).

    Interface:
        forward(x)   : (B, S, C, H, W) → (B, S, N_patches, token_dim) [backbone only]
        encode(nobs)  : full pipeline → (B, total_tokens, embed_dim)

    Properties:
        num_tokens  — patches per view
        token_dim   — feature dim per token (vit_embed_dim*2 if cat_token else vit_embed_dim)
        is_multi_view = True

    Injected by policy (via build_visual_encoder)
        cam_keys, n_obs_steps, embed_dim, crop_shape, in_channels, image_size

    YAML (visual_encoder_cfg)
        backbone, pretrained, alt_start, qknorm_start, rope_start, rope_freq,
        cat_token, out_layer, num_camera_tokens, include_camera_enc, camera_enc_cfg
    """

    def __init__(
        self,
        # injected by policy
        cam_keys: list = None,
        n_obs_steps: int = 2,
        embed_dim: int = 512,
        crop_shape: tuple = (224, 224),
        in_channels: int = 3,
        image_size: int = 256,
        # encoder-specific (YAML)
        backbone: str = "vitb",
        pretrained: bool = True,
        alt_start: int = 4,
        qknorm_start: int = 4,
        rope_start: int = 4,
        rope_freq: int = 100,
        cat_token: bool = True,
        out_layer: int = -1,
        num_camera_tokens: int = 3,
        include_camera_enc: bool = True,
        camera_enc_cfg: Optional[dict] = None,
        # crop augmentation
        crop_scale: Tuple[float, float] = (0.75, 1.0),
        crop_ratio: Tuple[float, float] = (0.9, 1.1),
        # photometric augmentation
        augmentation_cfg: dict = {},
        # camera noise augmentation (for robustness to calibration error)
        camera_noise_cfg: Optional[dict] = None,
    ):
        super().__init__()
        assert backbone in _VIT_CONFIGS, f"backbone must be one of {list(_VIT_CONFIGS)}"

        self.cam_keys = cam_keys or []
        self.n_obs_steps = n_obs_steps
        self.embed_dim = embed_dim  # output projection dim (= policy input_embedding_dim)

        cfg = _VIT_CONFIGS[backbone]
        depth = cfg["depth"]
        vit_embed_dim = cfg["embed_dim"]

        # Resolve out_layer
        self._out_layer = out_layer if out_layer >= 0 else (depth + out_layer)
        assert 0 <= self._out_layer < depth

        self.vit = DinoVisionTransformer(
            img_size=518,  # DINOv2 default; pos embed gets interpolated for other sizes
            patch_size=14,
            embed_dim=vit_embed_dim,
            depth=depth,
            num_heads=cfg["num_heads"],
            mlp_ratio=4,
            ffn_layer=cfg["ffn_layer"],
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            rope_freq=rope_freq,
            cat_token=cat_token,
            num_camera_tokens=num_camera_tokens,
        )

        if pretrained:
            load_pretrained_dinov2(self.vit, backbone=backbone)

        # Freeze the purely local (within-view) blocks before alt_start.
        # These are pretrained DINOv2 layers that only do single-image
        # processing — no cross-view interaction yet — so they're the
        # safest to freeze.  Blocks from alt_start onward are unfrozen
        # because the alternating local/global attention pattern is new
        # behaviour that needs to be learned.
        if alt_start > 0:
            # patch_embed and pos_embed feed into these early blocks
            for p in self.vit.patch_embed.parameters():
                p.requires_grad = False
            self.vit.pos_embed.requires_grad = False
            self.vit.cls_token.requires_grad = False
            for i in range(alt_start):
                for p in self.vit.blocks[i].parameters():
                    p.requires_grad = False

        self._token_dim = vit_embed_dim * 2 if cat_token else vit_embed_dim

        # Compute num_tokens for the crop size
        crop_h, crop_w = crop_shape
        self._num_tokens = (crop_h // 14) * (crop_w // 14)

        # Optional camera encoder
        self.camera_enc = None
        if include_camera_enc:
            cam_cfg = camera_enc_cfg or {}
            cam_cfg.setdefault("dim_out", vit_embed_dim)
            cam_cfg.setdefault("num_heads", cfg["num_heads"])
            self.camera_enc = CameraEnc(**cam_cfg)

        # ---- Self-contained encoder infrastructure ----
        self.crop_aug = RandomResizedCropAug(
            output_size=(crop_h, crop_w),
            scale=crop_scale,
            ratio=crop_ratio,
        )
        self.image_aug = ImageAugmentor(**augmentation_cfg)

        self.projector = nn.Linear(self._token_dim, embed_dim)

        n_cams = len(self.cam_keys)
        self.vis_camera_embed = nn.Embedding(max(n_cams, 1), embed_dim)
        nn.init.normal_(self.vis_camera_embed.weight, std=0.02)
        self.vis_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.vis_temporal_embed.weight, std=0.02)

        # Derive camera param keys from cam_keys by naming convention.
        self.extrinsic_keys = [k.replace("_image", "_extrinsic") for k in self.cam_keys]
        self.intrinsic_keys = [k.replace("_image", "_intrinsic") for k in self.cam_keys]

        # Camera noise (applied during training only). Disabled when cfg is None.
        _cn = camera_noise_cfg or {}
        self.cam_noise_trans_std = _cn.get("translation_std", 0.0)
        self.cam_noise_rot_deg = _cn.get("rotation_deg_std", 0.0)
        self.cam_noise_focal_rel = _cn.get("focal_rel_std", 0.0)
        self.cam_noise_pp_px = _cn.get("principal_point_px_std", 0.0)

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def token_dim(self) -> int:
        return self._token_dim

    @property
    def is_multi_view(self) -> bool:
        return True

    def _perturb_cameras(
        self,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Add noise to camera extrinsics/intrinsics for calibration robustness.

        Noise is per (B, camera), constant across timesteps (already collapsed
        into B dimension by the caller).  No-op at eval time.

        Args:
            extrinsics: (B, S, 4, 4) w2c transforms.
            intrinsics: (B, S, 3, 3) intrinsic matrices.

        Returns:
            Perturbed (extrinsics, intrinsics) with same shapes.
        """
        if not self.training:
            return extrinsics, intrinsics

        # extrinsics: (B, To, S, 4, 4),  intrinsics: (B, To, S, 3, 3)
        # Noise is per (B, camera), broadcast across To.
        B, To, S = extrinsics.shape[:3]
        device = extrinsics.device
        dtype = extrinsics.dtype

        # --- Extrinsic noise: small SE(3) perturbation ---
        # Sample per (B, S), unsqueeze To so it broadcasts across timesteps.
        t_noise = torch.randn(B, 1, S, 3, device=device, dtype=dtype) * self.cam_noise_trans_std

        angle_std = math.radians(self.cam_noise_rot_deg)
        axis = F.normalize(torch.randn(B, 1, S, 3, device=device, dtype=dtype), dim=-1)
        angle = torch.randn(B, 1, S, 1, device=device, dtype=dtype) * angle_std
        # Rodrigues: R = I + sin(a)*K + (1-cos(a))*K^2
        K = torch.zeros(B, 1, S, 3, 3, device=device, dtype=dtype)
        K[..., 0, 1] = -axis[..., 2]
        K[..., 0, 2] =  axis[..., 1]
        K[..., 1, 0] =  axis[..., 2]
        K[..., 1, 2] = -axis[..., 0]
        K[..., 2, 0] = -axis[..., 1]
        K[..., 2, 1] =  axis[..., 0]
        sin_a = torch.sin(angle).unsqueeze(-1)   # (B, 1, S, 1, 1)
        cos_a = torch.cos(angle).unsqueeze(-1)
        eye = torch.eye(3, device=device, dtype=dtype)
        dR = eye + sin_a * K + (1 - cos_a) * (K @ K)

        # Build 4x4 perturbation and apply: T_perturbed = dT @ T_original
        dT = torch.eye(4, device=device, dtype=dtype).expand(B, 1, S, 4, 4).clone()
        dT[..., :3, :3] = dR
        dT[..., :3, 3] = t_noise
        extrinsics = dT @ extrinsics

        # --- Intrinsic noise (per B, S; broadcast across To) ---
        intrinsics = intrinsics.clone()
        focal_noise = 1.0 + torch.randn(B, 1, S, 1, device=device, dtype=dtype) * self.cam_noise_focal_rel
        intrinsics[..., 0, 0] *= focal_noise.squeeze(-1)  # fx
        intrinsics[..., 1, 1] *= focal_noise.squeeze(-1)  # fy
        pp_noise = torch.randn(B, 1, S, device=device, dtype=dtype) * self.cam_noise_pp_px
        intrinsics[..., 0, 2] += pp_noise  # cx
        pp_noise = torch.randn(B, 1, S, device=device, dtype=dtype) * self.cam_noise_pp_px
        intrinsics[..., 1, 2] += pp_noise  # cy

        return extrinsics, intrinsics

    def forward(
        self,
        x: Tensor,
        extrinsics: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, S, C, H, W) multi-view images (raw images in [0,1]).
            extrinsics: (B, S, 4, 4) optional w2c camera extrinsics.
            intrinsics: (B, S, 3, 3) optional camera intrinsics.

        Returns:
            (B, S, N_patches, token_dim) visual tokens.
        """
        cam_token = None
        if self.camera_enc is not None and extrinsics is not None and intrinsics is not None:
            H, W = x.shape[-2], x.shape[-1]
            cam_token = self.camera_enc(extrinsics, intrinsics, (H, W))

        feats = self.vit.forward_features(
            x, out_layers=[self._out_layer], cam_token=cam_token,
        )
        return feats[-1]  # (B, S, N_patches, token_dim)

    def encode(self, nobs: dict) -> Tensor:
        """
        Full self-contained pipeline:
        crop+resize → un-norm → photometric aug → backbone → project → embed.

        Parameters
        ----------
        nobs : normalised observation dict; each value is (B, To_full, ...)

        Returns
        -------
        (B, n_obs_steps * n_cams * N_tokens, embed_dim)
        """
        # RandomResizedCrop each camera independently; params sampled per (B,)
        # and applied identically across all To timesteps.
        cropped = []
        crop_params_list = []
        for k in self.cam_keys:
            imgs = nobs[k][:, :self.n_obs_steps]           # (B, To, C, H, W)
            B, To = imgs.shape[:2]
            imgs, crop_params = self.crop_aug(imgs)        # (B, To, C, Hc, Wc)
            crop_params_list.append(crop_params)           # each value is (B,)
            cropped.append(imgs)

        n_cams = len(self.cam_keys)
        device = cropped[0].device

        # Stack cameras: (B, To, n_cams, C, Hc, Wc)
        cam_imgs = torch.stack(cropped, dim=2)
        # Un-normalize [-1,1] → [0,1]; ViT applies ImageNet norm internally.
        cam_imgs = (cam_imgs + 1.0) / 2.0
        B, To, n_cams, C, Hc, Wc = cam_imgs.shape

        # Photometric augmentations — params sampled per (batch, camera), shared
        # across timesteps to preserve temporal signal.
        cam_imgs = self.image_aug(cam_imgs)

        # Get camera params if available in nobs.
        extrinsics = None
        intrinsics = None
        if self.extrinsic_keys and self.extrinsic_keys[0] in nobs:
            extrinsics = torch.stack(
                [nobs[k][:, :To] for k in self.extrinsic_keys], dim=2
            )  # (B, To, n_cams, 4, 4)
            intrinsics = torch.stack(
                [nobs[k][:, :To] for k in self.intrinsic_keys], dim=2
            ).clone()  # (B, To, n_cams, 3, 3)

            # Adjust intrinsics for crop offset + resize scale.
            # Crop params are (B,) per camera; stack → (B, n_cams), then
            # unsqueeze To so they broadcast across all timesteps.
            tops = torch.stack(
                [p["top"] for p in crop_params_list], dim=1
            ).to(intrinsics.dtype)[:, None, :]                                    # (B, 1, n_cams)
            lefts = torch.stack(
                [p["left"] for p in crop_params_list], dim=1
            ).to(intrinsics.dtype)[:, None, :]
            scale_hs = torch.stack(
                [p["scale_h"] for p in crop_params_list], dim=1
            ).to(intrinsics.dtype)[:, None, :]
            scale_ws = torch.stack(
                [p["scale_w"] for p in crop_params_list], dim=1
            ).to(intrinsics.dtype)[:, None, :]

            intrinsics[..., 0, 0] *= scale_ws                                    # fx
            intrinsics[..., 1, 1] *= scale_hs                                    # fy
            intrinsics[..., 0, 2] = (intrinsics[..., 0, 2] - lefts) * scale_ws   # cx
            intrinsics[..., 1, 2] = (intrinsics[..., 1, 2] - tops) * scale_hs    # cy

            extrinsics, intrinsics = self._perturb_cameras(extrinsics, intrinsics)

        # Process per timestep: (B*To, n_cams, C, Hc, Wc)
        imgs_mv = cam_imgs.reshape(B * To, n_cams, C, Hc, Wc)
        ext = extrinsics.reshape(B * To, n_cams, 4, 4) if extrinsics is not None else None
        intr = intrinsics.reshape(B * To, n_cams, 3, 3) if intrinsics is not None else None

        tokens = self.forward(imgs_mv, extrinsics=ext, intrinsics=intr)
        # tokens: (B*To, n_cams, N_tok, token_dim)
        N_tok = tokens.shape[2]

        tokens = self.projector(tokens)  # (B*To, n_cams, N_tok, embed_dim)
        tokens = tokens.reshape(B, To, n_cams, N_tok, self.embed_dim)

        # Add camera and temporal embeddings.
        cam_ids = torch.arange(n_cams, device=device)
        time_ids = torch.arange(To, device=device)
        tokens = tokens + self.vis_camera_embed(cam_ids)[None, None, :, None, :]
        tokens = tokens + self.vis_temporal_embed(time_ids)[None, :, None, None, :]

        return tokens.reshape(B, To * n_cams * N_tok, self.embed_dim)