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
    ):
        super().__init__()

        rgb_keys     = [k for k in cam_keys if k.endswith("_image")]
        heatmap_keys = [k for k in cam_keys if k.endswith("_heatmap")]
        assert rgb_keys,     "DINOv2ResnetTokenEncoder: no '_image' keys in cam_keys"
        assert heatmap_keys, "DINOv2ResnetTokenEncoder: no '_heatmap' keys in cam_keys"

        print(f"[DINOv2ResnetTokenEncoder] rgb_keys={rgb_keys}, heatmap_keys={heatmap_keys}, "
              f"model_name={model_name}, frozen={frozen}, backbone={backbone}")

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

    @property
    def num_tokens(self) -> int:
        return self.dino_encoder.num_tokens + self.resnet_encoder.num_tokens

    @property
    def token_dim(self) -> int:
        return self.dino_encoder.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DINOv2ResnetTokenEncoder has two branches; use encode() directly."
        )

    def encode(self, nobs: dict) -> torch.Tensor:
        dino_toks   = self.dino_encoder.encode(nobs)
        resnet_toks = self.resnet_encoder.encode(nobs)
        return torch.cat([dino_toks, resnet_toks], dim=1)


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
        "dinov2_rgb_resnet_heatmap": DINOv2ResnetTokenEncoder
    }
    if encoder_type not in registry:
        raise ValueError(
            f"Unknown visual_encoder_type '{encoder_type}'. "
            f"Choose from: {list(registry)}"
        )
    return registry[encoder_type](**encoder_cfg)
