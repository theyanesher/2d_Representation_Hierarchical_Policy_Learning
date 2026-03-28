"""
Modular visual token encoders for FlowMatchingDiTImagePolicy.

Each encoder is fully self-contained: it owns its own CropRandomizer, linear
projector, and positional embeddings (spatial / camera / temporal).  The policy
calls ``encoder.encode(nobs)`` and gets back

    (B, n_obs_steps * n_cams * N_tokens, embed_dim)

ready for cross-attention in the DiT.

Interface (VisualTokenEncoder)
    forward(x)      : (B, C, H, W) → (B, N_tokens, token_dim)  [backbone only]
    encode(nobs)    : full pipeline → (B, total_tokens, embed_dim)

Implementations
    ResNetTokenEncoder       — ResNet feature map + spatial/camera/temporal embed
    ResNetPRoPETokenEncoder  — ResNet + PRoPE self-attention (camera-aware rotary PE)
    DINOv2TokenEncoder       — HuggingFace DINOv2 + optional spatial/camera/temporal embed
    DINOv3TokenEncoder       — DINOv3Backbone + optional spatial/camera/temporal embed

Constructor params injected by policy (all encoders)
    cam_keys    : List[str]       — nobs keys that contain image observations
    n_obs_steps : int
    embed_dim   : int             — output token dimension (= policy input_embedding_dim)
    crop_shape  : Tuple[int, int] — (H, W) passed to CropRandomizer
    in_channels : int             — C per image (may be > 3 for early-fusion modes)
    image_size  : int             — original image H/W before crop

For resnet_prope, camera param keys are derived automatically from cam_keys:
    cam{N}_image → cam{N}_extrinsic (4×4 w2c),  cam{N}_intrinsic (3×3 K)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from diffusion_policy.model.flow_matching.helpers import make_2d_sinusoidal_pos_embed
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.vision.prope import PropeDotProductAttention


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class VisualTokenEncoder(nn.Module, ABC):
    """
    All subclasses must expose ``num_tokens`` and ``token_dim`` as properties,
    implement the backbone-only ``forward(x)``, and implement the full
    end-to-end ``encode(nobs)``.
    """

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        """Number of tokens produced per image by the backbone."""

    @property
    @abstractmethod
    def token_dim(self) -> int:
        """Feature dimension of each backbone token (before projection)."""

    @abstractmethod
    def encode(self, nobs: dict) -> torch.Tensor:
        """
        Full pipeline: crop → backbone → project → positional encode.

        Parameters
        ----------
        nobs : normalised observation dict; each value is (B, To, ...)

        Returns
        -------
        (B, n_obs_steps * n_cams * N_tokens, embed_dim)
        """


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _replace_bn_with_gn(module: nn.Module, num_groups: int = 32) -> None:
    """Recursively replace BatchNorm2d with GroupNorm in-place."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            nf = child.num_features
            setattr(module, name, nn.GroupNorm(
                num_groups=min(num_groups, nf),
                num_channels=nf,
            ))
        else:
            _replace_bn_with_gn(child, num_groups)


def _build_spatial_embed(
    vis_spatial_embed_type: str,
    n_tokens: int,
    embed_dim: int,
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Build a spatial positional embedding tensor.

    Returns (tensor_or_None, is_buffer).
    ``is_buffer=True``  → register as buffer (sinusoidal, fixed).
    ``is_buffer=False`` → register as Parameter (learned).
    Returns (None, False) when vis_spatial_embed_type == 'none'.
    """
    if vis_spatial_embed_type == "none":
        return None, False
    if vis_spatial_embed_type == "sinusoidal":
        h = w = int(round(n_tokens ** 0.5))
        assert h * w == n_tokens, (
            f"sinusoidal spatial embed requires a square feature grid; "
            f"got n_tokens={n_tokens}"
        )
        return make_2d_sinusoidal_pos_embed(h, w, embed_dim), True
    # "learned"
    param = torch.zeros(n_tokens, embed_dim)
    nn.init.normal_(param, std=0.02)
    return param, False


def _crop_cam_keys(cam_keys, crop_randomizer, img_src, n_obs_steps, return_offsets=False):
    """
    Crop each camera's images independently and return a list of
    (B, To, C, Hc, Wc) tensors in the same order as cam_keys.

    If return_offsets=True, also returns a list of (B*To, 2) offset tensors
    [offset_h, offset_w] per camera (used to correct intrinsics for PRoPE).
    """
    cropped = []
    offsets_list = []
    B = To = None
    for k in cam_keys:
        imgs = img_src[k][:, :n_obs_steps]        # (B, To, C, H, W)
        B, To = imgs.shape[:2]
        flat = imgs.reshape(B * To, *imgs.shape[2:])
        if return_offsets:
            flat, offsets = crop_randomizer.forward_with_offsets(flat)  # offsets: (B*To, 2)
            offsets_list.append(offsets)
        else:
            flat = crop_randomizer(flat)               # (B*To, C, Hc, Wc)
        cropped.append(flat.reshape(B, To, *flat.shape[1:]))
    if return_offsets:
        return cropped, B, To, offsets_list
    return cropped, B, To


# ---------------------------------------------------------------------------
# Private ResNet backbone (pure CNN, no positional encoding)
# ---------------------------------------------------------------------------

class _ResNetBackbone(nn.Module):
    """
    ResNet trunk (avgpool and fc removed).  Used internally by
    ResNetTokenEncoder and ResNetPRoPETokenEncoder.
    """

    _TOKEN_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

    def __init__(
        self,
        backbone: str = "resnet18",
        in_channels: int = 3,
        image_size: int = 224,   # post-crop spatial size
        pretrained: bool = False,
        use_group_norm: bool = True,
    ):
        super().__init__()
        assert backbone in self._TOKEN_DIMS, (
            f"backbone must be one of {list(self._TOKEN_DIMS)}, got '{backbone}'"
        )

        weights = "DEFAULT" if (pretrained and in_channels == 3) else None
        print(f"[_ResNetBackbone] backbone={backbone}, pretrained={pretrained} → weights={weights}")
        resnet = getattr(tv_models, backbone)(weights=weights)

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        if use_group_norm:
            _replace_bn_with_gn(self.backbone)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            feat = self.backbone(dummy)

        self._num_tokens = feat.shape[2] * feat.shape[3]
        self._token_dim  = feat.shape[1]
        self._feat_h     = feat.shape[2]
        self._feat_w     = feat.shape[3]

        # When using ImageNet-pretrained weights the backbone expects images
        # in ImageNet mean/std over [0,1].  The policy LinearNormalizer maps
        # [0,1] → [-1,1], so we undo that and apply ImageNet normalisation.
        self._apply_imagenet_norm = pretrained and in_channels == 3
        if self._apply_imagenet_norm:
            self.register_buffer(
                "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def token_dim(self) -> int:
        return self._token_dim

    @property
    def feat_shape(self) -> Tuple[int, int]:
        return (self._feat_h, self._feat_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, h*w, token_dim)"""
        if self._apply_imagenet_norm:
            x = x * 0.5 + 0.5
            x = (x - self.img_mean) / self.img_std
        feat = self.backbone(x)
        B, C, h, w = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, h * w, C)


# ---------------------------------------------------------------------------
# ResNet encoder
# ---------------------------------------------------------------------------

class ResNetTokenEncoder(VisualTokenEncoder):
    """
    ResNet backbone with spatial, camera, and temporal positional embeddings.
    Owns its own CropRandomizer and projection layer.

    Backbone       token_dim   tokens (224×224 crop)
    ----------     ---------   ----------------------
    resnet18 / 34    512        7×7 = 49
    resnet50        2048        7×7 = 49

    Injected by policy
    ------------------
    cam_keys, n_obs_steps, embed_dim, crop_shape, in_channels, image_size

    YAML (visual_encoder_cfg)
    -------------------------
    backbone               : resnet18 | resnet34 | resnet50
    pretrained             : bool
    use_group_norm         : bool
    vis_spatial_embed_type : learned | sinusoidal | none  (default: learned)
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
        # YAML cfg
        backbone: str = "resnet18",
        pretrained: bool = True,
        use_group_norm: bool = True,
        vis_spatial_embed_type: str = "sinusoidal",
        wrist_cam_key: Optional[str] = None,
    ):
        super().__init__()
        self.cam_keys    = cam_keys
        self.n_obs_steps = n_obs_steps
        self.embed_dim   = embed_dim
        self.wrist_cam_key = wrist_cam_key

        print(f"[ResNetTokenEncoder] cam_keys={cam_keys}, backbone={backbone}, pretrained={pretrained}, wrist_cam_key={wrist_cam_key}")

        crop_h, crop_w = crop_shape
        self.crop_randomizer = CropRandomizer(
            input_shape=(in_channels, image_size, image_size),
            crop_height=crop_h,
            crop_width=crop_w,
        )

        self._resnet = _ResNetBackbone(
            backbone=backbone,
            in_channels=in_channels,
            image_size=crop_h,
            pretrained=pretrained,
            use_group_norm=use_group_norm,
        )

        self.projector = nn.Linear(self._resnet.token_dim, embed_dim)

        if wrist_cam_key is not None:
            print(f"[ResNetTokenEncoder] Creating separate wrist encoder for key='{wrist_cam_key}'")
            self._wrist_resnet = _ResNetBackbone(
                backbone=backbone,
                in_channels=in_channels,
                image_size=crop_h,
                pretrained=pretrained,
                use_group_norm=use_group_norm,
            )
            self.wrist_projector = nn.Linear(self._resnet.token_dim, embed_dim)

        # Spatial positional embed
        n_tok = self._resnet.num_tokens
        spatial, is_buf = _build_spatial_embed(vis_spatial_embed_type, n_tok, embed_dim)
        if spatial is not None:
            if is_buf:
                self.register_buffer("vis_spatial_embed", spatial)
            else:
                self.vis_spatial_embed = nn.Parameter(spatial)
        else:
            self.vis_spatial_embed = None

        # Camera embed (one per camera; +1 if separate wrist encoder)
        n_cams = len(cam_keys) + (1 if wrist_cam_key is not None else 0)
        self.vis_camera_embed = nn.Embedding(n_cams, embed_dim)
        nn.init.normal_(self.vis_camera_embed.weight, std=0.02)

        # Temporal embed (one per obs step)
        self.vis_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.vis_temporal_embed.weight, std=0.02)

    @property
    def num_tokens(self) -> int:
        return self._resnet.num_tokens

    @property
    def token_dim(self) -> int:
        return self._resnet.token_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone-only. x: (B, C, H, W) → (B, N_tok, token_dim)"""
        return self._resnet(x)

    def encode(self, nobs: dict) -> torch.Tensor:
        """Full pipeline. Returns (B, To * n_total_cams * N_tok, embed_dim)."""
        # Crop each main camera independently, then stack.
        cropped, B, To = _crop_cam_keys(
            self.cam_keys, self.crop_randomizer, nobs, self.n_obs_steps
        )
        n_main = len(self.cam_keys)
        device = cropped[0].device

        # Stack → (B, To, n_main, C, Hc, Wc) → flatten for backbone
        cam_imgs = torch.stack(cropped, dim=2)
        _, _, _, C, Hc, Wc = cam_imgs.shape
        flat = cam_imgs.reshape(B * To * n_main, C, Hc, Wc)

        toks = self._resnet(flat)               # (B*To*n_main, N_tok, token_dim)
        N_tok = toks.shape[1]
        toks = self.projector(toks)             # (B*To*n_main, N_tok, D)

        if self.vis_spatial_embed is not None:
            toks = toks + self.vis_spatial_embed

        toks = toks.reshape(B, To, n_main, N_tok, self.embed_dim)

        if self.wrist_cam_key is not None:
            cropped_wrist, _, _ = _crop_cam_keys(
                [self.wrist_cam_key], self.crop_randomizer, nobs, self.n_obs_steps
            )
            wrist_img = torch.stack(cropped_wrist, dim=2)  # (B, To, 1, C, H, W)
            flat_w = wrist_img.reshape(B * To, C, Hc, Wc)
            toks_w = self._wrist_resnet(flat_w)            # (B*To, N_tok, token_dim)
            toks_w = self.wrist_projector(toks_w)          # (B*To, N_tok, D)
            if self.vis_spatial_embed is not None:
                toks_w = toks_w + self.vis_spatial_embed
            toks_w = toks_w.reshape(B, To, 1, N_tok, self.embed_dim)
            toks = torch.cat([toks, toks_w], dim=2)        # (B, To, n_main+1, N_tok, D)

        n_total_cams = toks.shape[2]
        cam_ids  = torch.arange(n_total_cams, device=device)
        time_ids = torch.arange(To, device=device)
        toks = toks + self.vis_camera_embed(cam_ids)[None, None, :, None, :]
        toks = toks + self.vis_temporal_embed(time_ids)[None, :, None, None, :]

        return toks.reshape(B, To * n_total_cams * N_tok, self.embed_dim)


# ---------------------------------------------------------------------------
# PRoPE transformer block
# ---------------------------------------------------------------------------

class PRoPETransformerBlock(nn.Module):
    """
    Pre-norm transformer block with PRoPE self-attention for multi-camera tokens.

    Expects tokens ordered as [cam0_patches..., cam1_patches..., ..., camN_patches...]
    so that seqlen == n_cams * patches_x * patches_y.

    head_dim = embed_dim // num_heads must be divisible by 4 (PRoPE constraint).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_cams: int,
        patches_x: int,
        patches_y: int,
        image_size: int,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, (
            f"PRoPE requires head_dim % 4 == 0; got head_dim={head_dim} "
            f"(embed_dim={embed_dim}, num_heads={num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim  = head_dim

        self.norm1    = nn.LayerNorm(embed_dim)
        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.prope = PropeDotProductAttention(
            head_dim=head_dim,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_size,
            image_height=image_size,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        ff_dim = 4 * embed_dim
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        print(f"[PRoPETransformerBlock] embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}")

    def forward(
        self,
        x: torch.Tensor,         # (B, n_cams*patches_x*patches_y, embed_dim)
        viewmats: torch.Tensor,  # (B, n_cams, 4, 4)  w2c
        Ks: torch.Tensor,        # (B, n_cams, 3, 3)  intrinsics
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, hd = self.num_heads, self.head_dim

        residual = x
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).reshape(B, S, H, hd).permute(0, 2, 1, 3)
        k = self.k_proj(x_norm).reshape(B, S, H, hd).permute(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(B, S, H, hd).permute(0, 2, 1, 3)
        attn_out = self.prope(q, k, v, viewmats, Ks)   # (B, H, S, hd)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
        x = residual + self.out_proj(attn_out)

        return x + self.ff(self.norm2(x))


# ---------------------------------------------------------------------------
# ResNet + PRoPE encoder
# ---------------------------------------------------------------------------

class ResNetPRoPETokenEncoder(VisualTokenEncoder):
    """
    ResNet spatial tokeniser followed by stacked PRoPE self-attention blocks.
    Fully self-contained: owns CropRandomizer, projector, PRoPE blocks, and
    temporal embedding.  Spatial and camera position are handled geometrically
    by PRoPE (no explicit spatial/camera embed needed).

    Camera extrinsic/intrinsic keys are derived from cam_keys by convention:
        cam{N}_image → cam{N}_extrinsic  (4×4 w2c)
        cam{N}_image → cam{N}_intrinsic  (3×3 K)

    Injected by policy
    ------------------
    cam_keys, n_obs_steps, embed_dim, crop_shape, in_channels, image_size

    YAML (visual_encoder_cfg)
    -------------------------
    backbone         : resnet18 | resnet34 | resnet50
    pretrained       : bool
    use_group_norm   : bool
    num_prope_layers : int
    num_heads        : int
    wrist_cam_key    : str | None  — if set, adds a second ResNet for this camera;
                       its tokens join the main cameras in the PRoPE self-attention
                       block (must have matching extrinsic/intrinsic keys in nobs)
    """

    needs_camera_params: bool = True  # kept for backwards compatibility

    def __init__(
        self,
        # injected by policy
        cam_keys: List[str],
        n_obs_steps: int,
        embed_dim: int,
        crop_shape: Tuple[int, int],
        in_channels: int = 3,
        image_size: int = 256,
        # YAML cfg
        backbone: str = "resnet18",
        pretrained: bool = False,
        use_group_norm: bool = True,
        num_prope_layers: int = 2,
        num_heads: int = 8,
        wrist_cam_key: Optional[str] = None,
    ):
        super().__init__()
        self.cam_keys    = cam_keys
        self.n_obs_steps = n_obs_steps
        self.embed_dim   = embed_dim
        self.wrist_cam_key = wrist_cam_key

        print(f"[ResNetPRoPETokenEncoder] cam_keys={cam_keys}, backbone={backbone}, pretrained={pretrained}, "
              f"wrist_cam_key={wrist_cam_key}, num_prope_layers={num_prope_layers}, num_heads={num_heads}")

        # Derive camera param keys from cam_keys by naming convention.
        self.extrinsic_keys = [k.replace("_image", "_extrinsic") for k in cam_keys]
        self.intrinsic_keys = [k.replace("_image", "_intrinsic") for k in cam_keys]

        if wrist_cam_key is not None:
            self.wrist_extrinsic_key = wrist_cam_key.replace("_image", "_extrinsic")
            self.wrist_intrinsic_key = wrist_cam_key.replace("_image", "_intrinsic")
            n_total_cams = len(cam_keys) + 1
        else:
            n_total_cams = len(cam_keys)

        crop_h, crop_w = crop_shape
        self.crop_randomizer = CropRandomizer(
            input_shape=(in_channels, image_size, image_size),
            crop_height=crop_h,
            crop_width=crop_w,
        )

        self._resnet = _ResNetBackbone(
            backbone=backbone,
            in_channels=in_channels,
            image_size=crop_h,
            pretrained=pretrained,
            use_group_norm=use_group_norm,
        )
        self.projector = nn.Linear(self._resnet.token_dim, embed_dim)

        if wrist_cam_key is not None:
            print(f"[ResNetPRoPETokenEncoder] Creating separate wrist encoder for key='{wrist_cam_key}'")
            self._wrist_resnet = _ResNetBackbone(
                backbone=backbone,
                in_channels=in_channels,
                image_size=crop_h,
                pretrained=pretrained,
                use_group_norm=use_group_norm,
            )
            self.wrist_projector = nn.Linear(self._resnet.token_dim, embed_dim)

        feat_h, feat_w = self._resnet.feat_shape
        self.prope_blocks = nn.ModuleList([
            PRoPETransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                n_cams=n_total_cams,
                patches_x=feat_w,
                patches_y=feat_h,
                image_size=crop_h,
            )
            for _ in range(num_prope_layers)
        ])

        self.vis_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.vis_temporal_embed.weight, std=0.02)

    @property
    def num_tokens(self) -> int:
        return self._resnet.num_tokens

    @property
    def token_dim(self) -> int:
        return self._resnet.token_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone-only. x: (B, C, H, W) → (B, N_tok, token_dim)"""
        return self._resnet(x)

    def encode(self, nobs: dict) -> torch.Tensor:
        """Full pipeline. Returns (B, To * n_total_cams * N_tok, embed_dim)."""
        # Crop and encode main cameras; capture offsets to correct intrinsics.
        cropped_main, B, To, offsets_main = _crop_cam_keys(
            self.cam_keys, self.crop_randomizer, nobs, self.n_obs_steps,
            return_offsets=True,
        )
        n_main = len(self.cam_keys)
        device = cropped_main[0].device

        cam_imgs = torch.stack(cropped_main, dim=2)
        _, _, _, C, Hc, Wc = cam_imgs.shape
        flat = cam_imgs.reshape(B * To * n_main, C, Hc, Wc)
        toks = self._resnet(flat)               # (B*To*n_main, N_tok, token_dim)
        N_tok = toks.shape[1]
        toks = self.projector(toks)             # (B*To*n_main, N_tok, D)
        toks = toks.reshape(B * To, n_main * N_tok, self.embed_dim)

        ext_keys = list(self.extrinsic_keys)
        int_keys = list(self.intrinsic_keys)
        all_offsets = list(offsets_main)        # (B*To, 2) per camera

        if self.wrist_cam_key is not None:
            # Crop and encode wrist camera with its own backbone.
            cropped_wrist, _, _, offsets_wrist = _crop_cam_keys(
                [self.wrist_cam_key], self.crop_randomizer, nobs, self.n_obs_steps,
                return_offsets=True,
            )
            wrist_img = torch.stack(cropped_wrist, dim=2)  # (B, To, 1, C, H, W)
            flat_w = wrist_img.reshape(B * To, C, Hc, Wc)
            toks_w = self._wrist_resnet(flat_w)            # (B*To, N_tok, token_dim)
            toks_w = self.wrist_projector(toks_w)          # (B*To, N_tok, D)
            toks_w = toks_w.reshape(B * To, N_tok, self.embed_dim)

            # All tokens participate in the same PRoPE self-attention block.
            toks = torch.cat([toks, toks_w], dim=1)        # (B*To, n_total*N_tok, D)
            ext_keys.append(self.wrist_extrinsic_key)
            int_keys.append(self.wrist_intrinsic_key)
            all_offsets.extend(offsets_wrist)

        n_total_cams = len(ext_keys)

        # Per-timestep camera params from nobs (identity-normalised, so nobs == raw).
        # nobs[k]: (B, To_full, 4, 4) — slice to the To obs steps used.
        viewmats = (
            torch.stack([nobs[k][:, :To] for k in ext_keys], dim=2)
            .reshape(B * To, n_total_cams, 4, 4)
            .contiguous()
        )
        Ks = (
            torch.stack([nobs[k][:, :To] for k in int_keys], dim=2)
            .reshape(B * To, n_total_cams, 3, 3)
            .clone()        # clone so we can modify cx/cy in-place
            .contiguous()
        )

        # Correct cx/cy for the random crop offset by shifting principal point by (offset_h, offset_w).
        for i, offsets in enumerate(all_offsets):
            # offsets: (B*To, 2) with [:, 0]=offset_h, [:, 1]=offset_w
            Ks[:, i, 0, 2] -= offsets[:, 1].to(Ks.dtype)  # cx -= offset_w
            Ks[:, i, 1, 2] -= offsets[:, 0].to(Ks.dtype)  # cy -= offset_h

        for block in self.prope_blocks:
            toks = block(toks, viewmats, Ks)

        toks = toks.reshape(B, To, n_total_cams * N_tok, self.embed_dim)
        time_ids = torch.arange(To, device=device)
        toks = toks + self.vis_temporal_embed(time_ids)[None, :, None, :]
        return toks.reshape(B, To * n_total_cams * N_tok, self.embed_dim)


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

class DINOv2TokenEncoder(VisualTokenEncoder):
    """
    DINOv2 from HuggingFace.  Fully self-contained with crop/projector/embeds.

    vis_spatial_embed_type defaults to 'none' because the ViT backbone already
    bakes 2-D patch position into its tokens.

    Injected by policy
    ------------------
    cam_keys, n_obs_steps, embed_dim, crop_shape, in_channels, image_size

    YAML (visual_encoder_cfg)
    -------------------------
    model_name             : e.g. facebook/dinov2-base
    include_cls            : bool  (default False)
    frozen                 : bool  (default True)
    vis_spatial_embed_type : learned | sinusoidal | none  (default: none)
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
        # YAML cfg
        model_name: str = "facebook/dinov2-base",
        include_cls: bool = False,
        frozen: bool = True,
        vis_spatial_embed_type: str = "none",
    ):
        super().__init__()
        from transformers import AutoModel

        self.cam_keys    = cam_keys
        self.n_obs_steps = n_obs_steps
        self.embed_dim   = embed_dim

        crop_h, crop_w = crop_shape
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

        self.include_cls = include_cls
        self._token_dim  = self.dino.config.hidden_size
        self._num_tokens: Optional[int] = None  # set after first forward

        self.projector = nn.Linear(self._token_dim, embed_dim)

        # Spatial embed is created lazily (num_tokens depends on input size).
        self._vis_spatial_embed_type = vis_spatial_embed_type
        self._embeds_ready = False
        self.vis_spatial_embed = None

        n_cams = len(cam_keys)
        self.vis_camera_embed = nn.Embedding(n_cams, embed_dim)
        nn.init.normal_(self.vis_camera_embed.weight, std=0.02)

        self.vis_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.vis_temporal_embed.weight, std=0.02)

    def _lazy_init_spatial_embed(self, n_tokens: int, device):
        if self._embeds_ready:
            return
        spatial, is_buf = _build_spatial_embed(
            self._vis_spatial_embed_type, n_tokens, self.embed_dim
        )
        if spatial is not None:
            if is_buf:
                self.register_buffer("vis_spatial_embed", spatial.to(device))
            else:
                self.vis_spatial_embed = nn.Parameter(spatial.to(device))
        self._embeds_ready = True

    @property
    def num_tokens(self) -> int:
        assert self._num_tokens is not None, (
            "Call encode() at least once before accessing num_tokens."
        )
        return self._num_tokens

    @property
    def token_dim(self) -> int:
        return self._token_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone-only. x: (B, 3, H, W) → (B, N_patches, token_dim)"""
        if self.frozen:
            with torch.no_grad():
                out = self.dino(pixel_values=x)
        else:
            out = self.dino(pixel_values=x)
        tokens = out.last_hidden_state
        if not self.include_cls:
            tokens = tokens[:, 1:]
        self._num_tokens = tokens.shape[1]
        return tokens

    def encode(self, nobs: dict) -> torch.Tensor:
        """Full pipeline. Returns (B, To * n_cams * N_tok, embed_dim)."""
        cropped, B, To = _crop_cam_keys(
            self.cam_keys, self.crop_randomizer, nobs, self.n_obs_steps
        )
        n_cams = len(self.cam_keys)
        device = cropped[0].device

        cam_imgs = torch.stack(cropped, dim=2)
        _, _, _, C, Hc, Wc = cam_imgs.shape
        flat = cam_imgs.reshape(B * To * n_cams, C, Hc, Wc)

        toks = self.forward(flat)               # (B*To*n_cams, N_tok, token_dim)
        N_tok = toks.shape[1]
        self._lazy_init_spatial_embed(N_tok, device)

        toks = self.projector(toks)             # (B*To*n_cams, N_tok, D)
        if self.vis_spatial_embed is not None:
            toks = toks + self.vis_spatial_embed

        toks = toks.reshape(B, To, n_cams, N_tok, self.embed_dim)
        cam_ids  = torch.arange(n_cams, device=device)
        time_ids = torch.arange(To, device=device)
        toks = toks + self.vis_camera_embed(cam_ids)[None, None, :, None, :]
        toks = toks + self.vis_temporal_embed(time_ids)[None, :, None, None, :]

        return toks.reshape(B, To * n_cams * N_tok, self.embed_dim)


# ---------------------------------------------------------------------------
# DINOv2 (RGB) + ResNet (Heatmap) dual-branch encoder
# ---------------------------------------------------------------------------

class DINOv2ResnetTokenEncoder(VisualTokenEncoder):
    """
    Dual-branch encoder: RGB images through DINOv2, heatmaps through ResNet.

    Splits cam_keys by naming convention:
        keys ending in '_image'   → DINOv2TokenEncoder
        keys ending in '_heatmap' → ResNetTokenEncoder

    encode() calls both sub-encoders and concatenates along the token dim:
        (B, To*(n_rgb_cams*N_dino + n_heatmap_cams*N_resnet), embed_dim)

    Injected by policy
    ------------------
    cam_keys, n_obs_steps, embed_dim, crop_shape, in_channels, image_size

    YAML (visual_encoder_cfg)
    -------------------------
    model_name                    : e.g. facebook/dinov2-base
    frozen                        : bool  (default True)
    dino_vis_spatial_embed_type   : learned | sinusoidal | none  (default: none)
    backbone                      : resnet18 | resnet34 | resnet50  (default: resnet18)
    pretrained                    : bool  (default False)
    use_group_norm                : bool  (default True)
    resnet_vis_spatial_embed_type : learned | sinusoidal | none  (default: sinusoidal)
    heatmap_channels              : int   (default 1)
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
        # DINOv2 cfg
        model_name: str = "facebook/dinov2-base",
        frozen: bool = True,
        dino_vis_spatial_embed_type: str = "none",
        # ResNet cfg
        backbone: str = "resnet18",
        pretrained: bool = False,
        use_group_norm: bool = True,
        resnet_vis_spatial_embed_type: str = "sinusoidal",
        heatmap_channels: int = 1,
        # Self-attention mixing (concat DINO+heatmap tokens, then self-attn)
        add_self_attention_mixing: bool = False,
        num_attn_layers: int = 2,
        attn_num_heads: int = 8,
        attn_ffn_dim: int = 2048,
        attn_dropout: float = 0.0,
        # Cross-attention mixing (Q=DINO tokens, KV=heatmap tokens → output: DINO tokens only)
        add_cross_attention_mixing: bool = False,
        num_cross_attn_layers: int = 2,
        cross_attn_num_heads: int = 8,
        cross_attn_ffn_dim: int = 2048,
        cross_attn_dropout: float = 0.0,
    ):
        super().__init__()

        rgb_keys     = [k for k in cam_keys if k.endswith("_image")]
        heatmap_keys = [k for k in cam_keys if "_heatmap" in k]
        assert rgb_keys,     "DINOv2ResnetTokenEncoder: no '_image' keys in cam_keys"
        assert heatmap_keys, "DINOv2ResnetTokenEncoder: no '_heatmap' keys in cam_keys"

        self._mixing_mode = (
            "self" if add_self_attention_mixing else
            "cross" if add_cross_attention_mixing else
            "none"
        )

        print(f"[DINOv2ResnetTokenEncoder] rgb_keys={rgb_keys}, heatmap_keys={heatmap_keys}, "
              f"model_name={model_name}, frozen={frozen}, backbone={backbone}, "
              f"mixing_mode={self._mixing_mode}")

        self.dino_encoder = DINOv2TokenEncoder(
            cam_keys=rgb_keys,
            n_obs_steps=n_obs_steps,
            embed_dim=embed_dim,
            crop_shape=crop_shape,
            in_channels=in_channels,
            image_size=image_size,
            model_name=model_name,
            frozen=frozen,
            vis_spatial_embed_type=dino_vis_spatial_embed_type,
        )

        self.resnet_encoder = ResNetTokenEncoder(
            cam_keys=heatmap_keys,
            n_obs_steps=n_obs_steps,
            embed_dim=embed_dim,
            crop_shape=crop_shape,
            in_channels=heatmap_channels,
            image_size=image_size,
            backbone=backbone,
            pretrained=pretrained,
            use_group_norm=use_group_norm,
            vis_spatial_embed_type=resnet_vis_spatial_embed_type,
        )

        if add_self_attention_mixing:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=attn_num_heads,
                dim_feedforward=attn_ffn_dim,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.attn_mixer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_attn_layers,
            )
        elif add_cross_attention_mixing:
            # Stack of cross-attention blocks: Q=DINO tokens, KV=heatmap tokens
            # Each block: cross-MHA + residual + norm + FFN + residual + norm
            self.attn_mixer = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=embed_dim,
                    nhead=cross_attn_num_heads,
                    dim_feedforward=cross_attn_ffn_dim,
                    dropout=cross_attn_dropout,
                    batch_first=True,
                )
                for _ in range(num_cross_attn_layers)
            ])
        else:
            self.attn_mixer = None

    @property
    def num_tokens(self) -> int:
        # Cross-attention outputs only DINO tokens (heatmap tokens are consumed as KV)
        if self._mixing_mode == "cross":
            return self.dino_encoder.num_tokens
        return self.dino_encoder.num_tokens + self.resnet_encoder.num_tokens

    @property
    def token_dim(self) -> int:
        return self.dino_encoder.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DINOv2ResnetTokenEncoder has two branches; use encode() directly."
        )

    def encode(self, nobs: dict) -> torch.Tensor:
        dino_toks   = self.dino_encoder.encode(nobs)    # (B, N_dino, D)
        resnet_toks = self.resnet_encoder.encode(nobs)  # (B, N_resnet, D)

        if self._mixing_mode == "self":
            # Concat all tokens, mix with self-attention
            tokens = torch.cat([dino_toks, resnet_toks], dim=1)  # (B, N_dino+N_resnet, D)
            tokens = self.attn_mixer(tokens)
        elif self._mixing_mode == "cross":
            # Q=DINO, KV=heatmap; output is DINO tokens conditioned on heatmap
            tokens = dino_toks
            for layer in self.attn_mixer:
                tokens = layer(tgt=tokens, memory=resnet_toks)   # (B, N_dino, D)
        else:
            tokens = torch.cat([dino_toks, resnet_toks], dim=1)  # (B, N_dino+N_resnet, D)

        return tokens


# ---------------------------------------------------------------------------
# DINOv3
# ---------------------------------------------------------------------------

class DINOv3TokenEncoder(VisualTokenEncoder):
    """
    DINOv3Backbone wrapper.  Fully self-contained with crop/projector/embeds.

    DINOv3Backbone applies ImageNet normalisation internally, so this encoder
    un-normalises dataset-normalised images ([-1,1] → [0,1]) before forwarding.
    vis_spatial_embed_type defaults to 'none'.

    Injected by policy
    ------------------
    cam_keys, n_obs_steps, embed_dim, crop_shape, in_channels, image_size

    YAML (visual_encoder_cfg)
    -------------------------
    model_name             : dinov3 | dinov3-vitb16 | dinov2 | dinov2-base
    vis_spatial_embed_type : learned | sinusoidal | none  (default: none)
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
        # YAML cfg
        model_name: str = "dinov3",
        vis_spatial_embed_type: str = "none",
    ):
        super().__init__()
        from diffusion_policy.policy.act_policy import DINOv3Backbone

        self.cam_keys    = cam_keys
        self.n_obs_steps = n_obs_steps
        self.embed_dim   = embed_dim

        crop_h, crop_w = crop_shape
        self.crop_randomizer = CropRandomizer(
            input_shape=(in_channels, image_size, image_size),
            crop_height=crop_h,
            crop_width=crop_w,
        )

        self._backbone = DINOv3Backbone(
            model_name=model_name, crop_shape=crop_shape
        )
        self.projector = nn.Linear(self._backbone.hidden_dim, embed_dim)

        n_tok = self._backbone.n_patches
        spatial, is_buf = _build_spatial_embed(vis_spatial_embed_type, n_tok, embed_dim)
        if spatial is not None:
            if is_buf:
                self.register_buffer("vis_spatial_embed", spatial)
            else:
                self.vis_spatial_embed = nn.Parameter(spatial)
        else:
            self.vis_spatial_embed = None

        n_cams = len(cam_keys)
        self.vis_camera_embed = nn.Embedding(n_cams, embed_dim)
        nn.init.normal_(self.vis_camera_embed.weight, std=0.02)

        self.vis_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.vis_temporal_embed.weight, std=0.02)

    @property
    def num_tokens(self) -> int:
        return self._backbone.n_patches

    @property
    def token_dim(self) -> int:
        return self._backbone.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone-only. x: raw [0,1] (B, 3, H, W) → (B, N_patches, hidden_dim)"""
        return self._backbone(x)

    def encode(self, nobs: dict) -> torch.Tensor:
        """Full pipeline. Returns (B, To*n_cams*N_tok, D)."""
        # DINOv3Backbone applies ImageNet norm internally; undo dataset norm ([-1,1]→[0,1]).
        unnorm_nobs = {k: (v + 1.0) / 2.0 if k in self.cam_keys else v for k, v in nobs.items()}
        cropped, B, To = _crop_cam_keys(
            self.cam_keys, self.crop_randomizer, unnorm_nobs, self.n_obs_steps
        )
        n_cams = len(self.cam_keys)
        device = cropped[0].device

        cam_imgs = torch.stack(cropped, dim=2)
        _, _, _, C, Hc, Wc = cam_imgs.shape
        flat = cam_imgs.reshape(B * To * n_cams, C, Hc, Wc)

        toks = self._backbone(flat)             # (B*To*n_cams, N_tok, hidden_dim)
        N_tok = toks.shape[1]

        toks = self.projector(toks)             # (B*To*n_cams, N_tok, D)
        if self.vis_spatial_embed is not None:
            toks = toks + self.vis_spatial_embed

        toks = toks.reshape(B, To, n_cams, N_tok, self.embed_dim)
        cam_ids  = torch.arange(n_cams, device=device)
        time_ids = torch.arange(To, device=device)
        toks = toks + self.vis_camera_embed(cam_ids)[None, None, :, None, :]
        toks = toks + self.vis_temporal_embed(time_ids)[None, :, None, None, :]

        return toks.reshape(B, To * n_cams * N_tok, self.embed_dim)


# ---------------------------------------------------------------------------
# DINOv2 + HoMMI-style heatmap patch embedding
# ---------------------------------------------------------------------------

class DINOHoMMIStyleEmbedding(VisualTokenEncoder):
    """
    HoMMI-style per-patch fusion of DINOv2 RGB features + heatmap patch embeddings.

    For each patch i at spatial location (h, w) in camera c at timestep t:
        token_i = fusion_proj( concat( dino_feat_i, heatmap_patch_feat_i ) )

    The heatmap is embedded with a single Conv2d(heatmap_channels, embed_dim,
    kernel_size=patch_size, stride=patch_size), matching the DINOv2 spatial grid.

    Camera matching: cam{N}_image is paired with cam{N}_heatmap* by camera number.
    RGB cameras without a matching heatmap key pass through as plain DINO tokens.

    NOTE: For exact spatial alignment, crop_shape should equal image_size (no crop).

    Injected by policy
    ------------------
    cam_keys, n_obs_steps, embed_dim, crop_shape, in_channels, image_size

    YAML (visual_encoder_cfg)
    -------------------------
    model_name                  : e.g. facebook/dinov2-base
    frozen                      : bool  (default True)
    dino_vis_spatial_embed_type : learned | sinusoidal | none  (default: none)
    heatmap_channels            : int   (default 4 for ghost heatmap)
    patch_size                  : int   (default 14 to match DINOv2-base)
    use_single_conv             : bool  (default False)
        False — simple Conv2d(kernel=patch_size, stride=patch_size), i.e. linear
                patch projection identical to ViT patch embedding.
        True  — ResNet backbone (full image receptive field) followed by
                AdaptiveAvgPool2d to match DINO's (N_h, N_w) spatial grid,
                then a linear projector.  Richer features, more parameters.
    resnet_backbone             : resnet18 | resnet34 | resnet50  (default: resnet18,
                                  only used when use_single_conv=True)
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
        # DINOv2 cfg (forwarded to DINOv2TokenEncoder as-is)
        model_name: str = "facebook/dinov2-base",
        frozen: bool = True,
        dino_vis_spatial_embed_type: str = "none",
        # Heatmap embedder cfg
        heatmap_channels: int = 4,
        patch_size: int = 14,
        use_single_conv: bool = False,
        resnet_backbone: str = "resnet18",
    ):
        super().__init__()
        # import pdb; pdb.set_trace()
        rgb_keys     = [k for k in cam_keys if k.endswith("_image")]
        heatmap_keys = [k for k in cam_keys if "_heatmap" in k]
        assert rgb_keys, "DINOHoMMIStyleEmbedding: no '_image' keys in cam_keys"

        self.n_obs_steps = n_obs_steps
        self.embed_dim   = embed_dim

        # Map cam_num -> heatmap key  (e.g. 0 -> "cam0_heatmap_ghost")
        self._heatmap_map: dict = {}
        for h_key in heatmap_keys:
            cam_num = int("".join(filter(str.isdigit, h_key.split("_")[0])))
            self._heatmap_map[cam_num] = h_key

        # cam number for each rgb key in order (e.g. [0, 1, 2])
        self._rgb_cam_nums = [
            int("".join(filter(str.isdigit, k.split("_")[0])))
            for k in rgb_keys
        ]

        N_h = crop_shape[0] // patch_size
        N_w = crop_shape[1] // patch_size
        self._N_h = N_h
        self._N_w = N_w
        self._use_single_conv = use_single_conv

        print(f"[DINOHoMMIStyleEmbedding] rgb_keys={rgb_keys}, "
              f"heatmap_keys={heatmap_keys}, heatmap_map={self._heatmap_map}, "
              f"patch_size={patch_size}, use_single_conv={use_single_conv}, frozen={frozen}")

        # ---- DINOv2 branch: reuse existing DINOv2TokenEncoder ----
        self.dino_encoder = DINOv2TokenEncoder(
            cam_keys=rgb_keys,
            n_obs_steps=n_obs_steps,
            embed_dim=embed_dim,
            crop_shape=crop_shape,
            in_channels=in_channels,
            image_size=image_size,
            model_name=model_name,
            frozen=frozen,
            vis_spatial_embed_type=dino_vis_spatial_embed_type,
        )

        # ---- Heatmap embedder ----
        self.heatmap_crop_randomizer = CropRandomizer(
            input_shape=(heatmap_channels, image_size, image_size),
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        if use_single_conv:
            # ResNet backbone processes the full image with overlapping receptive fields,
            # then AdaptiveAvgPool2d resizes spatial dims to (N_h, N_w) to match DINO.
            self._heatmap_backbone = _ResNetBackbone(
                backbone=resnet_backbone,
                in_channels=heatmap_channels,
                image_size=crop_shape[0],
                pretrained=False,
                use_group_norm=True,
            )
            self._heatmap_spatial_pool = nn.AdaptiveAvgPool2d((N_h, N_w))
            self._heatmap_proj = nn.Linear(self._heatmap_backbone.token_dim, embed_dim)
        else:
            # Simple linear patch projection: Conv2d(kernel=stride=patch_size)
            # Equivalent to ViT patch embedding — each patch processed independently.
            self._heatmap_backbone     = None
            self._heatmap_spatial_pool = None
            self._heatmap_proj         = None
            self.heatmap_patch_embed = nn.Conv2d(
                heatmap_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size,
            )

        # ---- HoMMI fusion: concat(dino_patch, heatmap_patch) -> embed_dim ----
        self.fusion_proj = nn.Linear(2 * embed_dim, embed_dim)

    @property
    def num_tokens(self) -> int:
        # Output shape is (B, To * n_rgb * N_tok, D) — same as dino_encoder alone
        return self.dino_encoder.num_tokens

    @property
    def token_dim(self) -> int:
        return self.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DINOHoMMIStyleEmbedding: use encode() directly.")

    def encode(self, nobs: dict) -> torch.Tensor:
        # import pdb; pdb.set_trace();
        # ---- Step 1: get DINO tokens for all RGB cameras ----
        dino_flat = self.dino_encoder.encode(nobs)   # (B, To*n_rgb*N_tok, D)

        n_rgb = len(self._rgb_cam_nums)
        B     = dino_flat.shape[0]
        To    = self.n_obs_steps
        # N_tok per camera per timestep
        N_tok = dino_flat.shape[1] // (To * n_rgb)
        # import pdb; pdb.set_trace();
        # Reshape to (B, To, n_rgb, N_tok, D) for per-camera access
        dino_toks = dino_flat.reshape(B, To, n_rgb, N_tok, self.embed_dim)

        # ---- Step 2: embed heatmap + fuse per matched camera ----
        cam_tokens = []
        for cam_idx, cam_num in enumerate(self._rgb_cam_nums):
            d_cam = dino_toks[:, :, cam_idx]    # (B, To, N_tok, D)
            # import pdb; pdb.set_trace();
            if cam_num in self._heatmap_map:
                h_key  = self._heatmap_map[cam_num]
                h_data = nobs[h_key][:, :To]    # (B, To, C_h, H, W)
                h_flat = h_data.reshape(B * To, *h_data.shape[2:])
                h_flat = self.heatmap_crop_randomizer(h_flat)     # (B*To, C_h, Hc, Wc)
                # import pdb; pdb.set_trace();
                if self._use_single_conv:
                    # ResNet: full-image receptive field → AdaptivePool → (N_h, N_w)
                    rh, rw = self._heatmap_backbone.feat_shape
                    h_feat = self._heatmap_backbone(h_flat)        # (B*To, rh*rw, D_r)
                    h_feat = h_feat.reshape(B * To, rh, rw, -1).permute(0, 3, 1, 2)
                    h_feat = self._heatmap_spatial_pool(h_feat)    # (B*To, D_r, N_h, N_w)
                    h_toks = h_feat.flatten(2).transpose(1, 2)     # (B*To, N_tok, D_r)
                    h_toks = self._heatmap_proj(h_toks)            # (B*To, N_tok, D)
                else:
                    h_toks = self.heatmap_patch_embed(h_flat)      # (B*To, D, Ph, Pw)
                    h_toks = h_toks.flatten(2).transpose(1, 2)     # (B*To, N_tok, D)

                h_toks = h_toks.reshape(B, To, N_tok, self.embed_dim)

                # HoMMI-style per-patch fusion
                fused = self.fusion_proj(
                    torch.cat([d_cam, h_toks], dim=-1)             # (B, To, N_tok, 2D)
                )                                                   # (B, To, N_tok, D)
            else:
                fused = d_cam                                      # no heatmap for this cam

            cam_tokens.append(fused)
        # import pdb; pdb.set_trace();
        # ---- Step 3: reassemble and flatten ----
        all_toks = torch.stack(cam_tokens, dim=2)                 # (B, To, n_rgb, N_tok, D)
        return all_toks.reshape(B, To * n_rgb * N_tok, self.embed_dim)


# ---------------------------------------------------------------------------
# ViTHeatmapPosEmbedding
# ---------------------------------------------------------------------------

class ViTHeatmapPosEmbedding(VisualTokenEncoder):
    """
    ViT from scratch (no pretrained weights) where heatmap values act as
    soft positional embeddings for the RGB patch tokens.

    For each camera at each timestep:
        rgb_patches   : Conv2d(patch_size, stride=patch_size) → (N_tok, D)
        heatmap_pos   : Conv2d(patch_size, stride=patch_size) → (N_tok, D)
        tokens        : rgb_patches + heatmap_pos            [element-wise add]

    For cameras without a heatmap (e.g. wrist cam):
        tokens        : rgb_patches                          [zero heatmap signal]

    Learnable camera and temporal embeddings are also added so the transformer
    can distinguish which camera / timestep each token belongs to.

    A stack of TransformerEncoderLayers then processes all tokens jointly,
    enabling cross-camera and cross-timestep attention.

    Output shape: (B, To * n_rgb * N_tok, embed_dim)
    """

    def __init__(
        self,
        cam_keys: List[str],
        n_obs_steps: int,
        embed_dim: int,
        crop_shape: Tuple[int, int],
        in_channels: int = 3,
        image_size: int = 256,
        patch_size: int = 14,
        heatmap_channels: int = 4,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        rgb_keys     = [k for k in cam_keys if k.endswith("_image")]
        heatmap_keys = [k for k in cam_keys if "_heatmap" in k]
        assert rgb_keys, "ViTHeatmapPosEmbedding: no '_image' keys in cam_keys"

        self.n_obs_steps = n_obs_steps
        self.embed_dim   = embed_dim
        self._rgb_keys   = rgb_keys

        # Map cam_num -> heatmap key  (e.g. 0 -> "cam0_heatmap_ghost")
        self._heatmap_map: dict = {}
        for h_key in heatmap_keys:
            cam_num = int("".join(filter(str.isdigit, h_key.split("_")[0])))
            self._heatmap_map[cam_num] = h_key

        # cam number for each rgb key in order
        self._rgb_cam_nums = [
            int("".join(filter(str.isdigit, k.split("_")[0])))
            for k in rgb_keys
        ]

        N_h = crop_shape[0] // patch_size
        N_w = crop_shape[1] // patch_size
        self._N_h   = N_h
        self._N_w   = N_w
        self._N_tok = N_h * N_w

        n_rgb = len(rgb_keys)

        print(f"[ViTHeatmapPosEmbedding] rgb_keys={rgb_keys}, "
              f"heatmap_keys={heatmap_keys}, heatmap_map={self._heatmap_map}, "
              f"patch_size={patch_size}, N_tok={self._N_tok}, "
              f"transformer_layers={num_transformer_layers}")

        # ---- RGB patch embedding ----
        self.rgb_crop_randomizer = CropRandomizer(
            input_shape=(in_channels, image_size, image_size),
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )
        self.rgb_patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # ---- Heatmap positional embedding ----
        self.heatmap_crop_randomizer = CropRandomizer(
            input_shape=(heatmap_channels, image_size, image_size),
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )
        # Same stride=patch_size conv so heatmap tokens align 1-to-1 with RGB patches
        self.heatmap_pos_embed = nn.Conv2d(
            heatmap_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # ---- Camera and temporal identity embeddings ----
        # vis_camera_embed : one learnable vector per camera, added to all tokens
        #   from that camera so the transformer knows which camera the patch came from.
        # vis_temporal_embed : one learnable vector per obs timestep, added to all
        #   tokens at that step so the transformer knows temporal position.
        self.vis_camera_embed   = nn.Embedding(n_rgb, embed_dim)
        self.vis_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.vis_camera_embed.weight,   std=0.02)
        nn.init.normal_(self.vis_temporal_embed.weight, std=0.02)

        # ---- Transformer stack ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers,
        )

    @property
    def num_tokens(self) -> int:
        return self._N_tok

    @property
    def token_dim(self) -> int:
        return self.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ViTHeatmapPosEmbedding: use encode() directly.")

    def encode(self, nobs: dict) -> torch.Tensor:
        B   = next(iter(nobs.values())).shape[0]
        To  = self.n_obs_steps
        N_tok = self._N_tok
        D   = self.embed_dim
        dev = next(self.parameters()).device

        t_idx = torch.arange(To, device=dev)   # (To,)

        cam_tokens = []
        # import pdb; pdb.set_trace();
        for cam_idx, (rgb_key, cam_num) in enumerate(
            zip(self._rgb_keys, self._rgb_cam_nums)
        ):
            # ---- RGB patch tokens ----
            rgb      = nobs[rgb_key][:, :To]                    # (B, To, C, H, W)
            rgb_flat = rgb.reshape(B * To, *rgb.shape[2:])
            rgb_flat = self.rgb_crop_randomizer(rgb_flat)       # (B*To, C, Hc, Wc)
            rgb_toks = self.rgb_patch_embed(rgb_flat)           # (B*To, D, N_h, N_w)
            rgb_toks = rgb_toks.flatten(2).transpose(1, 2)      # (B*To, N_tok, D)
            rgb_toks = rgb_toks.reshape(B, To, N_tok, D)

            # ---- Heatmap positional embedding (or zero for cameras without heatmap) ----
            if cam_num in self._heatmap_map:
                h_key   = self._heatmap_map[cam_num]
                h_data  = nobs[h_key][:, :To]                   # (B, To, C_h, H, W)
                h_flat  = h_data.reshape(B * To, *h_data.shape[2:])
                h_flat  = self.heatmap_crop_randomizer(h_flat)  # (B*To, C_h, Hc, Wc)
                h_toks  = self.heatmap_pos_embed(h_flat)        # (B*To, D, N_h, N_w)
                h_toks  = h_toks.flatten(2).transpose(1, 2)     # (B*To, N_tok, D)
                h_toks  = h_toks.reshape(B, To, N_tok, D)
            else:
                h_toks = torch.zeros_like(rgb_toks)

            # ---- token = rgb_patch + heatmap_pos + cam_embed + temporal_embed ----
            tokens = rgb_toks + h_toks                          # (B, To, N_tok, D)

            cam_embed = self.vis_camera_embed(
                torch.tensor(cam_idx, device=dev)
            )                                                   # (D,)
            tokens = tokens + cam_embed[None, None, None, :]    # broadcast

            t_embed = self.vis_temporal_embed(t_idx)            # (To, D)
            tokens  = tokens + t_embed[None, :, None, :]        # broadcast

            cam_tokens.append(tokens)

        # ---- Assemble and flatten ----
        all_toks = torch.stack(cam_tokens, dim=2)               # (B, To, n_rgb, N_tok, D)
        all_toks = all_toks.reshape(B, To * len(self._rgb_keys) * N_tok, D)
        # import pdb; pdb.set_trace();
        # ---- Transformer ----
        return self.transformer(all_toks)                       # (B, To*n_rgb*N_tok, D)


# ---------------------------------------------------------------------------
# ViT from scratch with heatmap RoPE positional encoding
# ---------------------------------------------------------------------------

class ViTHeatmapRoPEEmbedding(VisualTokenEncoder):
    """
    ViT from scratch (no pretrained weights) where heatmap distance values are
    used as RoPE coordinates in each attention layer instead of being added
    additively to the tokens.

    Difference from ViTHeatmapPosEmbedding
    ---------------------------------------
    - No additive heatmap pos embed → no content-position entanglement.
    - Heatmap distances are avgpooled per patch → (N_tok, 4) coords.
    - coords are fed into HeatmapRoPEAttentionLayer to rotate Q and K.
    - Wrist cam (no heatmap): coords = 0 → R(0) = identity → standard attention.

    Architecture
    ------------
    RGB Conv2d patch embed → tokens (B*To, N_tok, D)
    + camera embed + temporal embed
    → N × HeatmapRoPEAttentionLayer(tokens, coords)
    → (B, To * n_rgb * N_tok, D)

    YAML (visual_encoder_cfg)
    -------------------------
    heatmap_channels      : int   (default 4)
    patch_size            : int   (default 14)
    num_rope_layers       : int   (default 4)
    num_heads             : int   (default 8)
    ffn_dim               : int   (default 2048)
    dropout               : float (default 0.0)
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
        # heatmap cfg
        heatmap_channels: int = 4,
        patch_size: int = 14,
        # transformer cfg
        num_rope_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        rgb_keys     = [k for k in cam_keys if k.endswith("_image")]
        heatmap_keys = [k for k in cam_keys if "_heatmap" in k]
        assert rgb_keys, "ViTHeatmapRoPEEmbedding: no '_image' keys in cam_keys"

        self.n_obs_steps = n_obs_steps
        self.embed_dim   = embed_dim
        self._rgb_keys   = rgb_keys
        self._patch_size = patch_size

        # Map cam_num → heatmap key
        self._heatmap_map: dict = {}
        for h_key in heatmap_keys:
            cam_num = int("".join(filter(str.isdigit, h_key.split("_")[0])))
            self._heatmap_map[cam_num] = h_key

        self._rgb_cam_nums = [
            int("".join(filter(str.isdigit, k.split("_")[0])))
            for k in rgb_keys
        ]

        N_h = crop_shape[0] // patch_size
        N_w = crop_shape[1] // patch_size
        self._N_tok = N_h * N_w

        n_rgb = len(rgb_keys)

        print(f"[ViTHeatmapRoPEEmbedding] rgb_keys={rgb_keys}, "
              f"heatmap_keys={heatmap_keys}, heatmap_map={self._heatmap_map}, "
              f"N_tok={self._N_tok}, num_rope_layers={num_rope_layers}")

        # ---- RGB patch embedding ----
        self.rgb_crop_randomizer = CropRandomizer(
            input_shape=(in_channels, image_size, image_size),
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )
        self.rgb_patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # ---- Heatmap crop randomizer (same crop as RGB) ----
        self.heatmap_crop_randomizer = CropRandomizer(
            input_shape=(heatmap_channels, image_size, image_size),
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        # ---- Camera and temporal identity embeddings ----
        self.vis_camera_embed   = nn.Embedding(n_rgb, embed_dim)
        self.vis_temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.vis_camera_embed.weight,   std=0.02)
        nn.init.normal_(self.vis_temporal_embed.weight, std=0.02)

        # ---- RoPE transformer layers ----
        assert embed_dim % 8 == 0, (
            f"embed_dim ({embed_dim}) must be divisible by 8 for 4-keypoint RoPE."
        )
        self.rope_layers = nn.ModuleList([
            HeatmapRoPEAttentionLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_rope_layers)
        ])

    @property
    def num_tokens(self) -> int:
        return self._N_tok

    @property
    def token_dim(self) -> int:
        return self.embed_dim

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ViTHeatmapRoPEEmbedding: use encode() directly.")

    def encode(self, nobs: dict) -> torch.Tensor:
        """Full pipeline. Returns (B, To * n_rgb * N_tok, embed_dim)."""
        B   = next(iter(nobs.values())).shape[0]
        To  = self.n_obs_steps
        N_tok = self._N_tok
        D   = self.embed_dim
        dev = next(self.parameters()).device

        cam_tokens_list  = []
        cam_coords_list  = []

        for cam_idx, (rgb_key, cam_num) in enumerate(
            zip(self._rgb_keys, self._rgb_cam_nums)
        ):
            # import pdb; pdb.set_trace()
            # ---- RGB patch tokens ----
            rgb      = nobs[rgb_key][:, :To]                     # (B, To, C, H, W)
            rgb_flat = rgb.reshape(B * To, *rgb.shape[2:])
            rgb_flat = self.rgb_crop_randomizer(rgb_flat)        # (B*To, C, Hc, Wc)
            rgb_toks = self.rgb_patch_embed(rgb_flat)            # (B*To, D, N_h, N_w)
            rgb_toks = rgb_toks.flatten(2).transpose(1, 2)       # (B*To, N_tok, D)

            # ---- Camera + temporal embeddings ----
            cam_emb  = self.vis_camera_embed(
                torch.tensor(cam_idx, device=dev)
            )                                                    # (D,)
            t_ids    = torch.arange(To, device=dev)
            t_emb    = self.vis_temporal_embed(t_ids)            # (To, D)

            rgb_toks = rgb_toks.reshape(B, To, N_tok, D)
            rgb_toks = rgb_toks + cam_emb[None, None, None, :]
            rgb_toks = rgb_toks + t_emb[None, :, None, :]
            rgb_toks = rgb_toks.reshape(B * To, N_tok, D)

            cam_tokens_list.append(rgb_toks)                     # (B*To, N_tok, D)

            # ---- Heatmap coords via avgpool ----
            if cam_num in self._heatmap_map:
                h_key   = self._heatmap_map[cam_num]
                h_data  = nobs[h_key][:, :To]                   # (B, To, 4, H, W)
                h_flat  = h_data.reshape(B * To, *h_data.shape[2:])
                h_flat  = self.heatmap_crop_randomizer(h_flat)  # (B*To, 4, Hc, Wc)
                coords  = F.avg_pool2d(
                    h_flat,
                    kernel_size=self._patch_size,
                    stride=self._patch_size,
                )                                                # (B*To, 4, N_h, N_w)
                coords  = coords.flatten(2).permute(0, 2, 1)    # (B*To, N_tok, 4)
            else:
                # No heatmap → zero coords → R(0) = identity → standard attention
                coords = torch.zeros(
                    B * To, N_tok, 4, device=dev, dtype=rgb_toks.dtype
                )

            cam_coords_list.append(coords)                       # (B*To, N_tok, 4)
        # import pdb; pdb.set_trace()
        # ---- Assemble all cameras into one sequence ----
        # tokens: (B*To, n_rgb * N_tok, D)
        # coords: (B*To, n_rgb * N_tok, 4)
        tokens = torch.cat(cam_tokens_list, dim=1)               # (B*To, n_rgb*N_tok, D)
        coords = torch.cat(cam_coords_list, dim=1)               # (B*To, n_rgb*N_tok, 4)
        coords = coords * N_tok
        # ---- RoPE transformer layers (cross-camera, within each timestep) ----
        import pdb; pdb.set_trace()
        for layer in self.rope_layers:
            tokens = layer(tokens, coords)
        # import pdb; pdb.set_trace()
        # ---- Reshape to (B, To * n_rgb * N_tok, D) ----
        n_rgb   = len(self._rgb_keys)
        tokens  = tokens.reshape(B, To * n_rgb * N_tok, D)
        return tokens


# ---------------------------------------------------------------------------
# Heatmap RoPE — custom attention layer + full encoder
# ---------------------------------------------------------------------------

class HeatmapRoPEAttentionLayer(nn.Module):
    """
    Pre-norm transformer layer where Q and K are rotated by RoPE angles derived
    from the 4-channel heatmap distance coordinates.

    Each of the 4 heatmap channels controls an independent slice of the head
    dimension:
        dims [0          : head_dim//4) ← keypoint 0 rotations
        dims [head_dim//4: head_dim//2) ← keypoint 1 rotations
        dims [head_dim//2: 3*head_dim//4) ← keypoint 2 rotations
        dims [3*head_dim//4: head_dim)  ← keypoint 3 rotations

    V is NOT rotated — positional encoding never contaminates the value space.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads
        assert head_dim % 8 == 0, (
            f"head_dim ({head_dim}) must be divisible by 8 "
            f"(4 keypoints × 2 for rotation pairs). "
            f"Got embed_dim={embed_dim}, num_heads={num_heads}."
        )

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = head_dim
        self.scale      = head_dim ** -0.5

        self.W_Q     = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K     = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V     = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Frequency schedule: one set of frequencies per keypoint slice.
        # pairs_per_kp = (head_dim // 4) // 2 = head_dim // 8
        pairs_per_kp = head_dim // 8
        freqs = 1.0 / (
            10000 ** (torch.arange(0, pairs_per_kp).float() / pairs_per_kp)
        )
        self.register_buffer("rope_freqs", freqs)   # (pairs_per_kp,)

    def _apply_rope(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Rotate x using heatmap coords.

        x:      (B, n_heads, N_tok, head_dim)
        coords: (B, N_tok, 4)  values expected in [0, 1]

        Returns rotated tensor of same shape.
        """
        head_dim = x.shape[-1]
        dims_per_kp = head_dim // 4     # number of dims per keypoint

        freqs = self.rope_freqs                          # (pairs_per_kp,)
        x_out = x.clone()

        for kp in range(4):
            start = kp * dims_per_kp
            end   = start + dims_per_kp
            x_kp  = x[:, :, :, start:end]               # (B, H, N, dims_per_kp)

            # distance for this keypoint: (B, N_tok) → (B, 1, N_tok, 1)
            d = coords[:, :, kp].unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)

            # angles: (B, 1, N_tok, pairs_per_kp)
            angles = d * freqs                           # broadcast
            cos_a  = torch.cos(angles)
            sin_a  = torch.sin(angles)

            # rotate each pair [x0, x1] → [x0·cos − x1·sin, x0·sin + x1·cos]
            x0 = x_kp[:, :, :, 0::2]                    # (B, H, N, pairs_per_kp)
            x1 = x_kp[:, :, :, 1::2]

            x_rot = torch.empty_like(x_kp)
            x_rot[:, :, :, 0::2] = x0 * cos_a - x1 * sin_a
            x_rot[:, :, :, 1::2] = x0 * sin_a + x1 * cos_a

            x_out[:, :, :, start:end] = x_rot

        return x_out

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N_tok, D)
        coords: (B, N_tok, 4)  — heatmap distances in [0, 1]; zeros for no-heatmap cameras

        Returns: (B, N_tok, D)
        """
        B, N_tok, D = tokens.shape

        # ---- Self-attention with heatmap RoPE ----
        residual  = tokens
        tokens_n  = self.norm1(tokens)

        Q = self.W_Q(tokens_n).reshape(B, N_tok, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(tokens_n).reshape(B, N_tok, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(tokens_n).reshape(B, N_tok, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE only to Q and K — V carries content, not position
        Q = self._apply_rope(Q, coords)
        K = self._apply_rope(K, coords)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                 # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N_tok, D)
        out = self.out_proj(out)
        tokens = residual + out

        # ---- FFN ----
        tokens = tokens + self.ffn(self.norm2(tokens))

        return tokens


class DINOHeatmapRoPEEncoder(VisualTokenEncoder):
    """
    Post-DINOv2 transformer where Q and K are rotated by heatmap-derived RoPE
    coordinates instead of the standard grid-based position.

    Architecture
    ------------
    1. DINOv2 (frozen) encodes all RGB cameras → (B, To*n_rgb*N_tok, D)
       (includes camera + temporal embeddings from DINOv2TokenEncoder)
    2. Heatmap avgpool extracts per-patch distance coords → (B, To*n_rgb*N_tok, 4)
       Cameras without a heatmap (e.g. wrist) get coords = 0 → R(0) = identity
       → degrades to standard self-attention for those tokens.
    3. num_rope_layers × HeatmapRoPEAttentionLayer applied to ALL tokens jointly
       (cross-camera, cross-timestep attention with heatmap-relative RoPE).

    Key property: Q·K dot product = f(content_i, content_j, coord_j - coord_i).
    No content-position cross terms; no implicit attention bias.

    YAML (visual_encoder_cfg)
    -------------------------
    model_name                   : e.g. facebook/dinov2-base
    frozen                       : bool  (default True)
    dino_vis_spatial_embed_type  : learned | sinusoidal | none  (default: none)
    heatmap_channels             : int   (default 4)
    patch_size                   : int   (default 14, must match DINOv2)
    num_rope_layers              : int   (default 2)
    num_heads                    : int   (default 8)
    ffn_dim                      : int   (default 2048)
    dropout                      : float (default 0.0)
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
        # DINOv2 cfg
        model_name: str = "facebook/dinov2-base",
        frozen: bool = True,
        dino_vis_spatial_embed_type: str = "none",
        # Heatmap cfg
        heatmap_channels: int = 4,
        patch_size: int = 14,
        # RoPE transformer cfg
        num_rope_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        rgb_keys     = [k for k in cam_keys if k.endswith("_image")]
        heatmap_keys = [k for k in cam_keys if "_heatmap" in k]
        assert rgb_keys, "DINOHeatmapRoPEEncoder: no '_image' keys in cam_keys"

        self.n_obs_steps = n_obs_steps
        self.embed_dim   = embed_dim
        self._rgb_keys   = rgb_keys
        self._patch_size = patch_size

        # Map cam_num → heatmap key (e.g. 0 → "cam0_heatmap_ghost")
        self._heatmap_map: dict = {}
        for h_key in heatmap_keys:
            cam_num = int("".join(filter(str.isdigit, h_key.split("_")[0])))
            self._heatmap_map[cam_num] = h_key

        self._rgb_cam_nums = [
            int("".join(filter(str.isdigit, k.split("_")[0])))
            for k in rgb_keys
        ]

        print(f"[DINOHeatmapRoPEEncoder] rgb_keys={rgb_keys}, "
              f"heatmap_keys={heatmap_keys}, heatmap_map={self._heatmap_map}, "
              f"num_rope_layers={num_rope_layers}")

        # ---- DINOv2 sub-encoder (RGB only) ----
        self.dino_encoder = DINOv2TokenEncoder(
            cam_keys=rgb_keys,
            n_obs_steps=n_obs_steps,
            embed_dim=embed_dim,
            crop_shape=crop_shape,
            in_channels=in_channels,
            image_size=image_size,
            model_name=model_name,
            frozen=frozen,
            vis_spatial_embed_type=dino_vis_spatial_embed_type,
        )

        # ---- Heatmap crop randomizer (must match RGB crop) ----
        self.heatmap_crop_randomizer = CropRandomizer(
            input_shape=(heatmap_channels, image_size, image_size),
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        # ---- RoPE attention layers ----
        self.rope_layers = nn.ModuleList([
            HeatmapRoPEAttentionLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_rope_layers)
        ])

    @property
    def num_tokens(self) -> int:
        return self.dino_encoder.num_tokens

    @property
    def token_dim(self) -> int:
        return self.embed_dim

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DINOHeatmapRoPEEncoder: use encode() directly.")

    def encode(self, nobs: dict) -> torch.Tensor:
        """Full pipeline. Returns (B, To * n_rgb * N_tok, embed_dim)."""

        # ---- Step 1: DINOv2 tokens for all RGB cameras ----
        # dino_encoder already handles crop, projection, cam+temporal embeds.
        tokens = self.dino_encoder.encode(nobs)     # (B, To*n_rgb*N_tok, D)

        B     = tokens.shape[0]
        To    = self.n_obs_steps
        n_rgb = len(self._rgb_keys)
        N_tok = tokens.shape[1] // (To * n_rgb)
        dev   = tokens.device

        # ---- Step 2: Build per-patch heatmap coords in the same token order ----
        # dino_encoder returns (B, To, n_rgb, N_tok, D) flattened, iterating
        # over cameras in self._rgb_keys order → match that order here.
        cam_coords = []
        for cam_num in self._rgb_cam_nums:
            if cam_num in self._heatmap_map:
                h_key   = self._heatmap_map[cam_num]
                h_data  = nobs[h_key][:, :To]               # (B, To, 4, H, W)
                h_flat  = h_data.reshape(B * To, *h_data.shape[2:])
                h_flat  = self.heatmap_crop_randomizer(h_flat)   # (B*To, 4, Hc, Wc)
                # AvgPool each channel over patch_size windows → (B*To, 4, N_h, N_w)
                coords  = F.avg_pool2d(
                    h_flat,
                    kernel_size=self._patch_size,
                    stride=self._patch_size,
                )
                coords  = coords.flatten(2).permute(0, 2, 1)    # (B*To, N_tok, 4)
                coords  = coords.reshape(B, To, N_tok, 4)
            else:
                # No heatmap → zero coords → R(0) = identity → standard attention
                coords = torch.zeros(B, To, N_tok, 4, device=dev, dtype=tokens.dtype)
            cam_coords.append(coords)

        # Stack to (B, To, n_rgb, N_tok, 4) then flatten to match token order
        all_coords = torch.stack(cam_coords, dim=2)             # (B, To, n_rgb, N_tok, 4)
        all_coords = all_coords.reshape(B, To * n_rgb * N_tok, 4)

        # ---- Step 3: Apply RoPE layers globally (cross-camera, cross-time) ----
        for layer in self.rope_layers:
            tokens = layer(tokens, all_coords)

        return tokens   # (B, To * n_rgb * N_tok, D)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_visual_encoder(encoder_type: str, encoder_cfg: dict) -> VisualTokenEncoder:
    """
    Instantiate a VisualTokenEncoder by type name.

    encoder_type : 'resnet' | 'resnet_prope' | 'dinov2' | 'dinov3'
    encoder_cfg  : kwargs forwarded to the constructor (includes injected params)
    """
    registry = {
        "resnet":       ResNetTokenEncoder,
        "resnet_prope": ResNetPRoPETokenEncoder,
        "dinov2":       DINOv2TokenEncoder,
        "dinov3":       DINOv3TokenEncoder,
        "dinov2_rgb_resnet_heatmap":   DINOv2ResnetTokenEncoder,
        "dinov2_hommi_style_heatmap":  DINOHoMMIStyleEmbedding,
        "vit_heatmap_pos_embedding":   ViTHeatmapPosEmbedding,
        "dino_heatmap_rope":           DINOHeatmapRoPEEncoder,
        "vit_heatmap_rope":            ViTHeatmapRoPEEmbedding,
    }
    if encoder_type not in registry:
        raise ValueError(
            f"Unknown visual_encoder_type '{encoder_type}'. "
            f"Choose from: {list(registry)}"
        )
    return registry[encoder_type](**encoder_cfg)
