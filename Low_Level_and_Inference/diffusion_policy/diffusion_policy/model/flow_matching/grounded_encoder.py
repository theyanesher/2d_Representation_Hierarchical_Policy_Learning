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
from diffusion_policy.model.flow_matching.visual_encoders import _crop_cam_keys
from diffusion_policy.model.vision.crop_randomizer import (
    CropRandomizer,
    crop_image_from_indices,
)

# NOTE on the import direction: this module imports from visual_encoders at
# module level, and visual_encoders imports THIS module lazily inside
# build_visual_encoder(). Keep it that way — making both directions eager
# creates a circular import.


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
        # --- optional goal-candidate tokens in the trunk ---------------------
        # When goal_key is None the trunk carries only patches + gripper (the
        # Approach 2 / plain-RoPE configuration). When set, the top-K GMM
        # candidates join as grounded tokens so patches can ground themselves
        # against where the robot is being sent. They are context only —
        # discarded at the output like the gripper tokens — and this does NOT
        # replace the DiT's own goal pathway (WCA still runs unchanged).
        goal_key: Optional[str] = None,
        goal_weights_key: Optional[str] = None,
        goal_top_k: int = 6,
        # Goals are FUTURE targets, so they sit at the far end of the time axis
        # rather than alongside the observations (which live at obs_step/n_total).
        goal_time_frac: float = 1.0,
        # Prior knobs, namespaced away from the DiT's wca_alpha / wca_beta since
        # both sets are live at once:
        #   logits(goal key j) = alpha_trunk * (q.k)/sqrt(d)
        #                        + beta_trunk  * log(pi_j / pi_max)
        #                        + gamma_trunk * log(pi_max)
        # beta_trunk  ranks candidates against each other.
        # gamma_trunk ties the goal stream's overall loudness to how peaked the
        #             high-level mixture is; 0.0 decouples them, 1.0 reproduces a
        #             raw log(pi) bias. Raw log(pi) is NOT the natural default
        #             here: unlike WCA, this softmax also spans the 1000+ patch
        #             keys, so a uniformly negative bias would suppress goals
        #             against patches rather than merely rank them.
        alpha_trunk: float = 1.0,
        beta_trunk: float = 1.0,
        gamma_trunk: float = 0.0,
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

        # Goal candidates: ONE token per candidate. Its RoPE position is
        # keypoint 3, which is bit-identically obs/state[:3] in these datasets —
        # the EE frame origin. Using the 4-point centroid instead would sit a
        # systematic ~29.5 mm off that convention, comparable to the whole
        # nearest-anchor scale (25-35 mm). The content still carries all 4
        # keypoints, so orientation and aperture are not lost.
        self.goal_key = goal_key
        self.goal_weights_key = goal_weights_key
        self.goal_top_k = int(goal_top_k)
        self.goal_time_frac = float(goal_time_frac)
        self.alpha_trunk = float(alpha_trunk)
        self.beta_trunk = float(beta_trunk)
        self.gamma_trunk = float(gamma_trunk)
        self.goal_encoder = None
        if goal_key is not None:
            assert goal_weights_key is not None, (
                "goal_key requires goal_weights_key (the GMM mixture weights)"
            )
            self.goal_encoder = nn.Sequential(
                nn.Linear(n_keypoints * 3, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.goal_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
            nn.init.normal_(self.goal_temporal_embed.weight, std=0.02)

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
        """Standard VisualTokenEncoder entry point: (B, To*n_cams*N_tok, D).

        Reads geometry straight out of ``nobs``, which requires the task to set
        ``identity_normalize_depth: true`` so depth arrives in METRES. The
        intrinsics/extrinsics (matrix keys) and present_gripper_pts
        (goal_gripper key) are identity-normalised already.

        Used by policies that only want better visual tokens — the anchors and
        validity mask are computed and discarded. Callers that need them (the
        goal-GMM auxiliary head) use ``encode_with_positions`` instead.
        """
        return self._encode(nobs, geom=nobs)[0]

    def encode_with_positions(self, nobs: dict, raw_obs: dict):
        """As ``encode``, but sources geometry from ``raw_obs`` and returns the
        per-token anchors and validity mask alongside the tokens.

        For policies that read raw obs directly instead of relying on
        ``identity_normalize_depth``.

        Returns:
            vis_tokens : (B, To*n_cams*N_tok, D)   grounded patch tokens
            vis_xyz    : (B, To*n_cams*N_tok, 3)   world anchors (raw metres)
            vis_valid  : (B, To*n_cams*N_tok)      bool, depth-valid patches
            grip_tokens: (B, To*n_keypoints, D)
            grip_xyz   : (B, To*n_keypoints, 3)
        """
        return self._encode(nobs, geom=raw_obs)

    # ------------------------------------------------------------------ #
    def _encode(self, nobs: dict, geom: dict):
        """Shared body.

        Args:
            nobs: normalised obs dict — used for RGB only.
            geom: dict to read geometry from (depth in METRES, camera matrices,
                  gripper keypoints in world metres). Either ``nobs`` itself
                  when the task identity-normalises depth, or a raw obs dict.
        """
        To = self.n_obs_steps
        n_cams = len(self.cam_keys)

        # -- RGB crop, capturing the offsets so depth gets the identical crop --
        cropped, B, _, offsets = _crop_cam_keys(
            self.cam_keys, self.crop_randomizer, nobs, To, return_offsets=True,
        )

        tok_per_cam, xyz_per_cam, valid_per_cam = [], [], []
        for ci in range(n_cams):
            imgs = cropped[ci]                                  # (B, To, C, Hc, Wc)
            C, Hc, Wc = imgs.shape[2:]
            toks = self.projector(self.forward(imgs.reshape(B * To, C, Hc, Wc)))
            N_tok = toks.shape[1]

            depth = geom[self.depth_keys[ci]][:, :To]         # (B,To,1,H,W) metres
            depth = depth.reshape(B * To, *depth.shape[2:])
            if depth.dim() == 4:
                depth = depth[:, 0]                              # (B*To, H, W)
            K = geom[self.intrinsic_keys[ci]][:, :To].reshape(B * To, 3, 3)
            E = geom[self.extrinsic_keys[ci]][:, :To].reshape(B * To, 4, 4)

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
        gp = geom[self.gripper_key][:, :To].float()           # (B, To, K, 3)
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

        seq = [vis_tokens, grip_tokens]
        pos = [
            torch.cat([vis_xyz * self.xyz_scale, t_vis], dim=-1),
            torch.cat([grip_xyz * self.xyz_scale, t_grip], dim=-1),
        ]

        # -- Goal-candidate tokens (optional) --
        n_goal, goal_bias = 0, None
        if self.goal_encoder is not None:
            gpts = geom[self.goal_key][:, :To].float()            # (B, To, N, K, 3)
            gw = geom[self.goal_weights_key][:, :To].float()      # (B, To, N)
            Kc = min(self.goal_top_k, gw.shape[-1])
            # Same topk on the same weights as the policy's gmm_top_k, so the
            # two goal pathways always see the identical candidate set.
            idx = torch.topk(gw, Kc, dim=-1).indices              # (B, To, Kc)
            gw = torch.gather(gw, 2, idx)
            gpts = torch.gather(
                gpts, 2,
                idx[..., None, None].expand(*idx.shape, gpts.shape[-2], gpts.shape[-1]),
            )                                                    # (B, To, Kc, K, 3)

            gt = self.goal_encoder(gpts.reshape(B * To * Kc, -1)).reshape(B, To, Kc, D)
            gt = gt + self.goal_temporal_embed(step_ids)[None, :, None, :]
            n_goal = To * Kc
            seq.append(gt.reshape(B, n_goal, D))

            # Keypoint 3 is the EE origin (== obs/state[:3]), not the centroid.
            goal_xyz = gpts[..., 3, :].reshape(B, n_goal, 3)
            t_goal = goal_xyz.new_full(
                (B, n_goal, 1), self.goal_time_frac * self.time_scale,
            )
            pos.append(torch.cat([goal_xyz * self.xyz_scale, t_goal], dim=-1))

            # pi_max is taken per (batch, obs step): each step carries its own
            # mixture, so the gauge must be fixed within a step, not across.
            log_pi = torch.log(gw.clamp(min=1e-8))                # (B, To, Kc)
            log_pi_max = log_pi.max(dim=-1, keepdim=True).values  # (B, To, 1)
            goal_bias = (
                self.beta_trunk * (log_pi - log_pi_max)
                + self.gamma_trunk * log_pi_max
            ).reshape(B, n_goal)

        x = torch.cat(seq, dim=1)
        pos = torch.cat(pos, dim=1)
        n_grip = grip_tokens.shape[1]
        N_total = x.shape[1]

        # Goal keys only: scale their content logits (a per-key scale hits the
        # matching COLUMN of the score matrix) and add the log-prior bias.
        key_scale = key_bias = None
        if n_goal:
            key_bias = x.new_zeros(B, N_total)
            key_bias[:, N_total - n_goal:] = goal_bias
            if self.alpha_trunk != 1.0:
                key_scale = x.new_ones(N_total)
                key_scale[N_total - n_goal:] = self.alpha_trunk

        # -- RoPE4D trunk over [patches ; gripper keypoints ; goal candidates] --
        for block in self.trunk:
            x = block(x, pos, key_scale=key_scale, key_bias=key_bias)

        # Gripper and goal tokens are context only — discarded here. Just the
        # patch tokens go on to the DiT.
        return x[:, :n_vis], vis_xyz, vis_valid, x[:, n_vis:n_vis + n_grip], grip_xyz
