"""
RoPE4D grounding primitives
===========================
4D rotary position embedding over continuous (x, y, z, t) coordinates, a
self-attention block that uses it, and the geometry helpers that turn a depth
buffer into per-patch world-frame anchors.

Used by the grounded visual encoder (see ``grounded_encoder.py``), which sits
between DINOv2 and the DiT and produces tokens that carry an explicit 3D anchor.

Extrinsic convention
--------------------
``obs/cam{i}_extrinsic`` in the MimicGen low-level h5 files is **camera-to-world**
(verified against ``obs/point_cloud``: 5.4 mm reconstruction error as c2w vs
545 mm as world-to-camera). Note this is the OPPOSITE of the convention used in
the MINO ArticuBot RoPE4D policy, which assumes world-to-camera. Do not port
that unprojection verbatim.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from torch import Tensor


# --------------------------------------------------------------------------- #
# Geometry                                                                      #
# --------------------------------------------------------------------------- #

def unproject_depth_to_world(
    depth_m: Tensor,
    intrinsic: Tensor,
    extrinsic_c2w: Tensor,
) -> Tensor:
    """Unproject a metric depth buffer into world-frame XYZ.

    Args:
        depth_m:        (N, H, W) depth in METRES (raw, not normalised).
        intrinsic:      (N, 3, 3) pinhole intrinsics for the same resolution.
        extrinsic_c2w:  (N, 4, 4) camera-to-world transform.

    Returns:
        (N, 3, H, W) world-frame XYZ, laid out like an image so the same crop
        that is applied to the RGB can be applied to it.
    """
    N, H, W = depth_m.shape
    device, dtype = depth_m.device, depth_m.dtype

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")            # (H, W)

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    z = depth_m
    cam = torch.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], dim=-1)  # (N,H,W,3)

    # camera-to-world: world = R @ cam + t
    R = extrinsic_c2w[:, :3, :3]                            # (N, 3, 3)
    t = extrinsic_c2w[:, :3, 3]                             # (N, 3)
    world = torch.einsum("nij,nhwj->nhwi", R, cam) + t[:, None, None, :]
    return world.permute(0, 3, 1, 2).contiguous()           # (N, 3, H, W)


def extract_patch_centers(pointmap: Tensor, patch_size: int = 14) -> Tensor:
    """(N, 3, H, W) -> (N, n_patches, 3): the XYZ at each patch's centre pixel.

    Mirrors how a ViT tiles its input, so patch token i of the backbone and row i
    of the returned tensor describe the same image region.
    """
    N, C, H, W = pointmap.shape
    ph, pw = H // patch_size, W // patch_size
    c = patch_size // 2
    rows = torch.arange(ph, device=pointmap.device) * patch_size + c
    cols = torch.arange(pw, device=pointmap.device) * patch_size + c
    return pointmap[:, :, rows][:, :, :, cols].permute(0, 2, 3, 1).reshape(N, ph * pw, C)


# --------------------------------------------------------------------------- #
# 4D rotary position embedding                                                  #
# --------------------------------------------------------------------------- #

class RotaryPositionEmbedding4D(nn.Module):
    """Rotary embedding over continuous (x, y, z, t).

    Splits ``head_dim`` into four equal parts and applies 1-D RoPE to each
    coordinate. Positions are continuous, not integer indices.

    Rotations are applied to Q and K only, so they influence attention weights
    but leave no trace in the block's output tokens.
    """

    def __init__(self, head_dim: int, base_frequency: float = 100.0):
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        self.quarter = head_dim // 4
        inv_freq = 1.0 / (
            base_frequency ** (torch.arange(0, self.quarter, 2).float() / self.quarter)
        )
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """
        Args:
            tokens:    (B, heads, N, head_dim)
            positions: (B, N, 4) continuous (x, y, z, t)
        """
        parts = []
        q = self.quarter
        for i in range(4):
            coord = positions[..., i: i + 1]                 # (B, N, 1)
            angles = coord * self.inv_freq                   # (B, N, quarter//2)
            freqs = torch.cat([angles, angles], dim=-1)[:, None, :, :]  # (B,1,N,quarter)
            x = tokens[..., i * q:(i + 1) * q]
            parts.append(x * freqs.cos() + self._rotate_half(x) * freqs.sin())
        return torch.cat(parts, dim=-1)


class RoPE4DSelfAttention(nn.Module):
    """Multi-head self-attention with 4D RoPE on Q and K."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        base_frequency: float = 100.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=bias)

        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

        self.rope = RotaryPositionEmbedding4D(head_dim, base_frequency=base_frequency)
        self.dropout = dropout

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """x: (B, N, dim), pos: (B, N, 4)."""
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.to_q(x).reshape(B, N, H, D).transpose(1, 2)
        k = self.to_k(x).reshape(B, N, H, D).transpose(1, 2)
        v = self.to_v(x).reshape(B, N, H, D).transpose(1, 2)

        q = self.rope(self.q_norm(q), pos)
        k = self.rope(self.k_norm(k), pos)
        v = v.to(q.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.to_out(out.transpose(1, 2).reshape(B, N, -1))


class RoPE4DBlock(nn.Module):
    """Pre-norm transformer block: RoPE4D self-attention + feed-forward.

    Deliberately has no timestep conditioning — the trunk output is independent
    of the flow-matching time, so it can be computed once and reused across every
    Euler step at inference.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        head_dim: int = 64,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        base_frequency: float = 100.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPE4DSelfAttention(
            dim, num_heads, head_dim, dropout=dropout, base_frequency=base_frequency,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.ff(self.norm2(x))
        return x
