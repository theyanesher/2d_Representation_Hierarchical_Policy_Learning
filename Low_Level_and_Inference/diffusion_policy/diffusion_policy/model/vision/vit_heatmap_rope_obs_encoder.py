"""
ViTHeatmapRoPEObsEncoder

Wraps ViTHeatmapRoPEEmbedding (from visual_encoders.py) to match the
robomimic obs_encoder interface expected by DiffusionUnetHybridImagePolicy.

The policy flattens nobs from (B, To, ...) → (B*To, ...) before calling
obs_encoder, then reshapes the output (B*To, feat) → (B, To*feat).
This wrapper:
  1. Reverses the flatten: (B*To, ...) → (B, To, ...)
  2. Calls ViTHeatmapRoPEEmbedding.encode() → (B, To*n_rgb*N_tok, D)
  3. Mean-pools over spatial tokens (N_tok) → (B, To, n_rgb, D)
  4. Returns (B*To, n_rgb * embed_dim)

output_shape() → (n_rgb * embed_dim,)
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from diffusion_policy.model.flow_matching.visual_encoders import ViTHeatmapRoPEEmbedding


class ViTHeatmapRoPEObsEncoder(nn.Module):
    """
    Obs encoder wrapper around ViTHeatmapRoPEEmbedding compatible with
    DiffusionUnetHybridImagePolicy.

    Parameters
    ----------
    cam_keys          : all observation keys (both _image and _heatmap)
    n_obs_steps       : number of observation timesteps (To)
    embed_dim         : token embedding dimension
    crop_shape        : (H, W) crop passed to CropRandomizer
    in_channels       : RGB channels (default 3)
    image_size        : original image H/W before crop
    heatmap_channels  : number of heatmap channels (default 4 for ghost heatmap)
    patch_size        : patch size for Conv2d embed (default 14)
    num_rope_layers   : number of HeatmapRoPEAttentionLayer stacked (default 4)
    num_heads         : attention heads (default 8)
    ffn_dim           : FFN hidden dim (default 2048)
    dropout           : dropout (default 0.0)
    """

    def __init__(
        self,
        cam_keys: List[str],
        n_obs_steps: int,
        embed_dim: int,
        crop_shape: Tuple[int, int],
        in_channels: int = 3,
        image_size: int = 256,
        heatmap_channels: int = 4,
        patch_size: int = 14,
        num_rope_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self._n_rgb      = len([k for k in cam_keys if k.endswith("_image")])
        self._embed_dim  = embed_dim

        self.vit = ViTHeatmapRoPEEmbedding(
            cam_keys=cam_keys,
            n_obs_steps=n_obs_steps,
            embed_dim=embed_dim,
            crop_shape=crop_shape,
            in_channels=in_channels,
            image_size=image_size,
            heatmap_channels=heatmap_channels,
            patch_size=patch_size,
            num_rope_layers=num_rope_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def output_shape(self):
        """Returns (n_rgb * embed_dim,) — the flat feature size per timestep."""
        return (self._n_rgb * self._embed_dim,)

    def forward(self, this_nobs: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        this_nobs : dict{key: Tensor(B*To, C, H, W)}
            Flattened by the policy before calling obs_encoder.

        Returns
        -------
        Tensor (B*To, n_rgb * embed_dim)
        """
        To   = self.n_obs_steps
        B_To = next(iter(this_nobs.values())).shape[0]
        B    = B_To // To

        # ---- Step 1: Unflatten (B*To, ...) → (B, To, ...) ----
        nobs_unflat = {
            k: v.reshape(B, To, *v.shape[1:])
            for k, v in this_nobs.items()
        }
        # import pdb; pdb.set_trace();
        # ---- Step 2: Encode → (B, To * n_rgb * N_tok, D) ----
        tokens = self.vit.encode(nobs_unflat)

        n_rgb = self._n_rgb
        D     = self._embed_dim
        N_tok = tokens.shape[1] // (To * n_rgb)

        # ---- Step 3: Mean-pool over N_tok → (B, To, n_rgb, D) ----
        tokens = tokens.reshape(B, To, n_rgb, N_tok, D).mean(dim=3)

        # ---- Step 4: Return (B*To, n_rgb * D) ----
        return tokens.reshape(B * To, n_rgb * D)
