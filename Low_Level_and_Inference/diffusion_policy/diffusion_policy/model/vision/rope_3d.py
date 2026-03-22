"""3D Rotary Position Embeddings for multi-camera encoder.

Back-projects each image patch to world coordinates (x, y, z) using depth +
camera intrinsics/extrinsics, then applies rotary position embeddings so that
attention dot products naturally encode 3D proximity.

Components:
  - patches_to_3d_positions(): depth → world coords per patch
  - RotaryPositionEmbedding3D: computes and applies RoPE from (x, y, z)
  - RoPE3DMultiheadAttention: drop-in replacement for nn.MultiheadAttention
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from diffusion_policy.common.camera_utils import get_pointmap


# ============================================================
# Core RoPE helpers (adapted from prope.py for continuous 3D)
# ============================================================


def _rope_precompute_freqs(dim: int, freq_base: float = 10000.0) -> Tensor:
    """Geometric frequency series for RoPE.

    Args:
        dim: must be even — number of dimensions for this axis
        freq_base: base for the geometric series
    Returns:
        (dim // 2,) frequency tensor
    """
    assert dim % 2 == 0
    num_freqs = dim // 2
    freqs = 1.0 / (
        freq_base ** (torch.arange(num_freqs, dtype=torch.float32) / num_freqs)
    )
    return freqs  # (num_freqs,)


def _rope_apply(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Apply rotary embedding using split convention (first half, second half).

    Args:
        x: (..., dim)
        cos, sin: (..., dim // 2) — broadcastable to x's leading dims
    Returns:
        (..., dim) rotated tensor
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1)


# ============================================================
# RotaryPositionEmbedding3D
# ============================================================


class RotaryPositionEmbedding3D(nn.Module):
    """Computes and applies 3D RoPE from continuous (x, y, z) positions.

    Splits head_dim into 3 equal even blocks (one per axis).
    Requires head_dim % 6 == 0.
    """

    def __init__(self, head_dim: int, freq_base: float = 10000.0):
        super().__init__()
        assert head_dim % 6 == 0, f"head_dim must be divisible by 6, got {head_dim}"
        self.head_dim = head_dim
        self.axis_dim = head_dim // 3  # dims per axis (must be even)

        # Precompute frequency series (shared across axes)
        freqs = _rope_precompute_freqs(self.axis_dim, freq_base)
        self.register_buffer("freqs", freqs, persistent=False)  # (axis_dim // 2,)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        positions_3d: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply 3D RoPE to Q and K.

        Args:
            q: (B, n_heads, S, head_dim)
            k: (B, n_heads, S, head_dim)
            positions_3d: (B, S, 3) world coordinates per token.
                          Tokens with all-zero positions get identity rotation.
        Returns:
            q_rot, k_rot: same shapes as inputs
        """
        B, n_heads, S, D = q.shape
        assert D == self.head_dim
        assert positions_3d.shape == (B, S, 3)

        ad = self.axis_dim
        half = ad // 2

        # Compute angles for each axis: (B, S, half)
        # positions_3d[..., i] is (B, S), freqs is (half,)
        cos_list, sin_list = [], []
        for axis in range(3):
            pos = positions_3d[..., axis]  # (B, S)
            angles = pos.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)  # (B, S, half)
            cos_list.append(torch.cos(angles))
            sin_list.append(torch.sin(angles))

        # Concatenate across axes: (B, S, 3 * half) = (B, S, head_dim // 2)
        cos_all = torch.cat(cos_list, dim=-1)  # (B, S, head_dim // 2)
        sin_all = torch.cat(sin_list, dim=-1)

        # Expand for heads: (B, 1, S, head_dim // 2)
        cos_all = cos_all.unsqueeze(1)
        sin_all = sin_all.unsqueeze(1)

        q_rot = _rope_apply(q, cos_all, sin_all)
        k_rot = _rope_apply(k, cos_all, sin_all)

        return q_rot, k_rot


# ============================================================
# RoPE3DMultiheadAttention
# ============================================================


class RoPE3DMultiheadAttention(nn.Module):
    """Multi-head attention with 3D RoPE on Q and K.

    Drop-in replacement for nn.MultiheadAttention in encoder layers.
    Uses separate linear projections and F.scaled_dot_product_attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        freq_base: float = 10000.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

        self.rope_3d = RotaryPositionEmbedding3D(self.head_dim, freq_base=freq_base)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        positions_3d: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, None]:
        """
        Args:
            q, k, v: (S, B, D) — sequence-first (matching nn.MultiheadAttention)
            positions_3d: (B, S, 3) world coordinates, or None for identity
            key_padding_mask: (B, S) bool mask (True = ignore)
        Returns:
            (output, None) — None for compat with nn.MultiheadAttention API
        """
        S, B, D = q.shape
        H = self.num_heads
        hd = self.head_dim

        # Project
        q_proj = self.q_proj(q)  # (S, B, D)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # Reshape to (B, H, S, hd)
        q_proj = q_proj.permute(1, 0, 2).reshape(B, S, H, hd).permute(0, 2, 1, 3)
        k_proj = k_proj.permute(1, 0, 2).reshape(B, S, H, hd).permute(0, 2, 1, 3)
        v_proj = v_proj.permute(1, 0, 2).reshape(B, S, H, hd).permute(0, 2, 1, 3)

        # Apply 3D RoPE
        if positions_3d is not None:
            q_proj, k_proj = self.rope_3d(q_proj, k_proj, positions_3d)

        # Attention mask from key_padding_mask: (B, S) -> (B, 1, 1, S)
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: True means ignore
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn_mask = attn_mask.to(dtype=torch.bool)
            # sdpa expects: True = attend, so invert, OR use float -inf
            attn_mask = torch.where(attn_mask, float("-inf"), 0.0)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q_proj, k_proj, v_proj,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, S, hd)

        # Reshape back to (S, B, D)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D).permute(1, 0, 2)

        out = self.out_proj(out)
        return out, None


# ============================================================
# patches_to_3d_positions()
# ============================================================


def patches_to_3d_positions(
    depths: List[Tensor],
    intrinsics: List[Tensor],
    extrinsics: List[Tensor],
    eef_pos: Tensor,
    crop_shape: Tuple[int, int],
    patch_size: Optional[int] = None,
    feat_shape: Optional[Tuple[int, int]] = None,
    use_dino: bool = True,
) -> Tensor:
    """Compute 3D world coordinates for each token: [state, patches...].

    For each patch/feature, finds the center pixel in the original image,
    looks up its 3D world coordinate from the depth-based pointmap.
    The state token uses the end-effector position (first 3 dims of state).

    The backbone applies CenterCrop(crop_shape) before extracting features,
    so patch positions must be offset to index into the original depth map.

    Args:
        depths: list of (B, 1, H, W) per camera (original resolution)
        intrinsics: list of (B, 3, 3) per camera (for original resolution)
        extrinsics: list of (B, 4, 4) w2c per camera (dataset convention; inverted internally)
        eef_pos: (B, 3) end-effector world position for the state token
        crop_shape: (crop_H, crop_W) — CenterCrop size applied by backbone
        patch_size: ViT patch size (for DINOv3, mutually exclusive with feat_shape)
        feat_shape: (feat_H, feat_W) actual backbone feature map spatial dims
                    (for ResNet, mutually exclusive with patch_size)
        use_dino: True for DINOv3 (uses patch_size), False for ResNet (uses feat_shape)

    Returns:
        (B, 1 + total_patches, 3) world coordinates.
        First token is the EEF position, rest are patch positions.
    """
    n_cams = len(depths)
    B = depths[0].shape[0]
    device = depths[0].device
    dtype = depths[0].dtype
    crop_h, crop_w = crop_shape

    all_positions = []

    for cam_idx in range(n_cams):
        depth = depths[cam_idx]       # (B, 1, H, W)
        K = intrinsics[cam_idx]       # (B, 3, 3)
        w2c = extrinsics[cam_idx]     # (B, 4, 4) — dataset stores w2c
        _, _, dH, dW = depth.shape

        # get_pointmap expects c2w, but dataset stores w2c → invert
        c2w = torch.inverse(w2c)

        # Get full 3D pointmap from the original depth map: (B, 3, H, W)
        pointmap = get_pointmap(K, c2w, depth, channel_first=True)
        # CenterCrop offset: patch positions are in crop coords, shift to
        # original image coords to index the pointmap correctly.
        offset_y = (dH - crop_h) // 2
        offset_x = (dW - crop_w) // 2

        if use_dino:
            assert patch_size is not None
            # Patch grid within the crop
            n_patches_h = crop_h // patch_size
            n_patches_w = crop_w // patch_size
            # Center pixel of each patch, in crop coordinates
            cy_crop = torch.arange(n_patches_h, device=device) * patch_size + patch_size // 2
            cx_crop = torch.arange(n_patches_w, device=device) * patch_size + patch_size // 2
        else:
            assert feat_shape is not None
            feat_h, feat_w = feat_shape
            # Evenly space feature centers across the crop region.
            # stride = crop_dim / feat_dim gives the effective stride.
            stride_h = crop_h / feat_h
            stride_w = crop_w / feat_w
            cy_crop = torch.arange(feat_h, device=device).float() * stride_h + stride_h / 2
            cx_crop = torch.arange(feat_w, device=device).float() * stride_w + stride_w / 2

        # Shift to original image coordinates and clamp
        cy = (cy_crop + offset_y).long().clamp(0, dH - 1)
        cx = (cx_crop + offset_x).long().clamp(0, dW - 1)

        grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        cam_positions = pointmap[:, :, grid_y, grid_x]  # (B, 3, n_feats)
        cam_positions = cam_positions.permute(0, 2, 1)   # (B, n_feats, 3)
        all_positions.append(cam_positions)

    # Concatenate all cameras
    image_positions = torch.cat(all_positions, dim=1)  # (B, total_patches, 3)

    # State token uses EEF position: (B, 3) -> (B, 1, 3)
    state_position = eef_pos.unsqueeze(1)  # (B, 1, 3)
    positions = torch.cat([state_position, image_positions], dim=1)

    return positions  # (B, 1 + total_patches, 3)
