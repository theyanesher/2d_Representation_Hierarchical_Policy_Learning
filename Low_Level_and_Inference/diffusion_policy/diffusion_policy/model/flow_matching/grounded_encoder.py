"""
DINOv2 + RoPE4D grounded visual encoder
=======================================
Sits between the frozen DINOv2 backbone and the DiT. Gives every patch token an
explicit world-frame 3D anchor (unprojected from depth), adds the gripper
keypoints as extra grounded tokens, and fuses them with a few layers of 4D-RoPE
self-attention.

Its output serves two consumers:
  * the DiT, as ``encoder_hidden_states`` (drop-in replacement for the plain
    DINOv2 tokens), and
  * the goal-GMM auxiliary head, which needs the per-token anchors.

Both read the same tensor, so the auxiliary loss shapes the representation the
policy actually cross-attends to.

The trunk carries no flow-timestep conditioning, so its output can be computed
once and reused across every Euler step at inference.

Token order is (obs_step, camera, patch) throughout, matching
``DINOv2TokenEncoder.encode`` so downstream reshapes behave identically.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_policy.model.flow_matching.rope4d_grounding import (
    RoPE4DBlock,
    extract_patch_centers,
    unproject_depth_to_world,
)
from diffusion_policy.model.vision.crop_randomizer import (
    CropRandomizer,
    crop_image_from_indices,
)


class DINOv2RoPE4DGroundedEncoder(nn.Module):
    """Frozen DINOv2 -> projector -> 3D grounding -> RoPE4D self-attention trunk.

    Injected by the policy
    ----------------------
    cam_keys, n_obs_steps, embed_dim, crop_shape, in_channels, image_size

    YAML (visual_encoder_cfg)
    -------------------------
    model_name, frozen, patch_size, n_trunk_layers, num_heads, head_dim,
    xyz_scale, time_scale, base_frequency, dropout
    """

    def __init__(
        self,
        # injected by policy
        cam_keys: List[str],
        n_obs_steps: int,
        embed_dim: int,
        crop_shape: Tuple[int, int],
        in_channels: int = 3,
        image_size: int = 256,
        # obs keys for geometry (defaults derived from cam_keys when None)
        depth_keys: Optional[List[str]] = None,
        intrinsic_keys: Optional[List[str]] = None,
        extrinsic_keys: Optional[List[str]] = None,
        gripper_key: str = "present_gripper_pts",
        n_total_steps: int = 18,
        n_keypoints: int = 4,
        # YAML cfg
        model_name: str = "facebook/dinov2-base",
        frozen: bool = True,
        patch_size: int = 14,
        n_trunk_layers: int = 2,
        num_heads: int = 16,
        head_dim: int = 64,
        xyz_scale: float = 5.0,
        time_scale: float = 1.0,
        base_frequency: float = 100.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        from transformers import AutoModel

        self.cam_keys = cam_keys
        self.n_obs_steps = n_obs_steps
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.xyz_scale = xyz_scale
        self.time_scale = time_scale
        self.n_total_steps = n_total_steps
        self.n_keypoints = n_keypoints
        self.gripper_key = gripper_key

        stem = [k.replace("_image", "") for k in cam_keys]
        self.depth_keys = depth_keys or [f"{s}_depth" for s in stem]
        self.intrinsic_keys = intrinsic_keys or [f"{s}_intrinsic" for s in stem]
        self.extrinsic_keys = extrinsic_keys or [f"{s}_extrinsic" for s in stem]

        crop_h, crop_w = crop_shape
        self.crop_h, self.crop_w = crop_h, crop_w
        self.crop_randomizer = CropRandomizer(
            input_shape=(in_channels, image_size, image_size),
            crop_height=crop_h,
            crop_width=crop_w,
        )

        self.dino = AutoModel.from_pretrained(model_name)
        self.frozen = frozen
        if frozen:
            for p in self.dino.parameters():
                p.requires_grad = False

        self._token_dim = self.dino.config.hidden_size
        self._num_tokens: Optional[int] = None
        self.projector = nn.Linear(self._token_dim, embed_dim)

        n_cams = len(cam_keys)
        self.vis_camera_embed = nn.Embedding(n_cams, embed_dim)
        nn.init.normal_(self.vis_camera_embed.weight, std=0.02)
        self.vis_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.vis_temporal_embed.weight, std=0.02)

        # Gripper keypoints: one token per keypoint per obs step, each at its own
        # 3D position. Mirrors ArticuBot's prepare_scene_pcd, which prepends the
        # 4 gripper points to the scene cloud as anchors.
        self.gripper_encoder = nn.Sequential(
            nn.Linear(3, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim),
        )
        self.keypoint_embed = nn.Embedding(n_keypoints, embed_dim)
        nn.init.normal_(self.keypoint_embed.weight, std=0.02)
        self.grip_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.grip_temporal_embed.weight, std=0.02)

        assert embed_dim == num_heads * head_dim, (
            f"embed_dim ({embed_dim}) must equal num_heads*head_dim "
            f"({num_heads}*{head_dim}={num_heads*head_dim})"
        )
        self.trunk = nn.ModuleList([
            RoPE4DBlock(
                embed_dim, num_heads=num_heads, head_dim=head_dim,
                dropout=dropout, base_frequency=base_frequency,
            )
            for _ in range(n_trunk_layers)
        ])

    # ------------------------------------------------------------------ #
    @property
    def num_tokens(self) -> int:
        assert self._num_tokens is not None, "call encode_with_positions() first"
        return self._num_tokens

    @property
    def token_dim(self) -> int:
        return self._token_dim

    def forward(self, x: Tensor) -> Tensor:
        """Backbone only. (N, 3, H, W) -> (N, N_patches, token_dim)."""
        if self.frozen:
            with torch.no_grad():
                out = self.dino(pixel_values=x)
        else:
            out = self.dino(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:]      # drop CLS
        self._num_tokens = tokens.shape[1]
        return tokens

    def encode(self, nobs: dict) -> Tensor:
        raise NotImplementedError(
            "DINOv2RoPE4DGroundedEncoder needs unnormalised depth/state, so it "
            "must be driven via encode_with_positions(nobs, raw_obs)."
        )

    # ------------------------------------------------------------------ #
    def encode_with_positions(self, nobs: dict, raw_obs: dict):
        """
        Args:
            nobs:    normalised obs dict (used for RGB only).
            raw_obs: unnormalised obs dict (depth in metres, camera matrices,
                     gripper keypoints in world metres).

        Returns:
            vis_tokens : (B, To*n_cams*N_tok, D)   grounded patch tokens
            vis_xyz    : (B, To*n_cams*N_tok, 3)   world anchors (raw metres)
            vis_valid  : (B, To*n_cams*N_tok)      bool, depth-valid patches
            grip_tokens: (B, To*n_keypoints, D)
            grip_xyz   : (B, To*n_keypoints, 3)
        """
        To = self.n_obs_steps
        n_cams = len(self.cam_keys)

        # -- RGB crop, capturing the offsets so depth gets the identical crop --
        from diffusion_policy.model.flow_matching.visual_encoders import _crop_cam_keys
        cropped, B, _, offsets = _crop_cam_keys(
            self.cam_keys, self.crop_randomizer, nobs, To, return_offsets=True,
        )

        tok_per_cam, xyz_per_cam, valid_per_cam = [], [], []
        for ci in range(n_cams):
            imgs = cropped[ci]                                  # (B, To, C, Hc, Wc)
            C, Hc, Wc = imgs.shape[2:]
            toks = self.projector(self.forward(imgs.reshape(B * To, C, Hc, Wc)))
            N_tok = toks.shape[1]

            depth = raw_obs[self.depth_keys[ci]][:, :To]         # (B,To,1,H,W) metres
            depth = depth.reshape(B * To, *depth.shape[2:])
            if depth.dim() == 4:
                depth = depth[:, 0]                              # (B*To, H, W)
            K = raw_obs[self.intrinsic_keys[ci]][:, :To].reshape(B * To, 3, 3)
            E = raw_obs[self.extrinsic_keys[ci]][:, :To].reshape(B * To, 4, 4)

            pm = unproject_depth_to_world(depth.float(), K.float(), E.float())
            # Identical crop to the RGB, so patch i of the ViT and row i of the
            # anchors describe the same pixels.
            pm = crop_image_from_indices(pm, offsets[ci], Hc, Wc)
            xyz = extract_patch_centers(pm, self.patch_size)      # (B*To, N_tok, 3)

            d_crop = crop_image_from_indices(
                depth[:, None].float(), offsets[ci], Hc, Wc,
            )
            valid = extract_patch_centers(d_crop, self.patch_size)[..., 0] > 1e-6

            assert xyz.shape[1] == N_tok, (
                f"patch-centre count {xyz.shape[1]} != ViT token count {N_tok}; "
                f"check patch_size ({self.patch_size}) against the backbone"
            )
            tok_per_cam.append(toks)
            xyz_per_cam.append(xyz)
            valid_per_cam.append(valid)

        D = self.embed_dim
        # (B*To, n_cams, N, ...) -> (B, To, n_cams, N, ...) keeps (To, cam, patch)
        toks = torch.stack(tok_per_cam, dim=1).reshape(B, To, n_cams, -1, D)
        xyz = torch.stack(xyz_per_cam, dim=1).reshape(B, To, n_cams, -1, 3)
        valid = torch.stack(valid_per_cam, dim=1).reshape(B, To, n_cams, -1)
        N_tok = toks.shape[3]

        device = toks.device
        cam_ids = torch.arange(n_cams, device=device)
        step_ids = torch.arange(To, device=device)
        toks = toks + self.vis_camera_embed(cam_ids)[None, None, :, None, :]
        toks = toks + self.vis_temporal_embed(step_ids)[None, :, None, None, :]

        n_vis = To * n_cams * N_tok
        vis_tokens = toks.reshape(B, n_vis, D)
        vis_xyz = xyz.reshape(B, n_vis, 3)
        vis_valid = valid.reshape(B, n_vis)

        # -- Gripper keypoint tokens (raw world metres) --
        gp = raw_obs[self.gripper_key][:, :To].float()           # (B, To, K, 3)
        K_pts = gp.shape[2]
        gtok = self.gripper_encoder(gp.reshape(-1, 3)).reshape(B, To, K_pts, D)
        gtok = gtok + self.keypoint_embed(torch.arange(K_pts, device=device))
        gtok = gtok + self.grip_temporal_embed(step_ids)[None, :, None, :]
        grip_tokens = gtok.reshape(B, To * K_pts, D)
        grip_xyz = gp.reshape(B, To * K_pts, 3)

        # -- 4D positions: raw metres * xyz_scale, obs-step fraction * time_scale --
        t_vals = (step_ids.to(vis_xyz.dtype) / self.n_total_steps) * self.time_scale
        t_vis = t_vals.repeat_interleave(n_cams * N_tok)[None, :, None].expand(B, -1, 1)
        t_grip = t_vals.repeat_interleave(K_pts)[None, :, None].expand(B, -1, 1)

        pos = torch.cat([
            torch.cat([vis_xyz * self.xyz_scale, t_vis], dim=-1),
            torch.cat([grip_xyz * self.xyz_scale, t_grip], dim=-1),
        ], dim=1)

        # -- RoPE4D trunk over [patches ; gripper keypoints] --
        x = torch.cat([vis_tokens, grip_tokens], dim=1)
        for block in self.trunk:
            x = block(x, pos)

        return x[:, :n_vis], vis_xyz, vis_valid, x[:, n_vis:], grip_xyz
