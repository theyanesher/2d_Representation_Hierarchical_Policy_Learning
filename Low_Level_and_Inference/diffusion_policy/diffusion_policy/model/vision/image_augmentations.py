"""
Image augmentation modules for visual encoders.

- ``RandomResizedCropAug``: batched random-resized-crop on GPU tensors.
- ``ImageAugmentor``: photometric augmentations (brightness, contrast, etc.).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms.v2.functional as tvf


class RandomResizedCropAug(nn.Module):
    """
    Batched random-resized-crop on GPU tensors.

    Training: samples a random crop region (area in ``scale``, aspect ratio in
    ``ratio``), then bilinear-resizes to ``output_size``.  Returns per-image
    crop parameters so the caller can adjust camera intrinsics.

    Eval: center-crops to ``output_size``

    Aspect ratio is sampled in log-space so that ``ratio=r`` and ``ratio=1/r``
    are equally likely (symmetric around 1.0).
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        scale: Tuple[float, float] = (0.75, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
    ):
        super().__init__()
        self.output_size = output_size
        self.scale = scale
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    def forward(self, x: Tensor):
        """
        Args:
            x: (B, To, C, H, W) — crop params sampled per B, applied
               identically across all To timesteps.
        Returns:
            out:    (B, To, C, out_H, out_W)
            params: dict with ``top, left, crop_h, crop_w, scale_h, scale_w``
                    — each a (B,) tensor.
        """
        B, To, C, H, W = x.shape
        out_h, out_w = self.output_size

        if self.training:
            area = H * W
            target_area = (
                torch.empty(B, device=x.device).uniform_(self.scale[0], self.scale[1])
                * area
            )
            log_r = torch.empty(B, device=x.device).uniform_(
                self.log_ratio[0], self.log_ratio[1]
            )
            ar = torch.exp(log_r)

            crop_w = torch.round(torch.sqrt(target_area * ar)).to(torch.long).clamp(1, W)
            crop_h = torch.round(torch.sqrt(target_area / ar)).to(torch.long).clamp(1, H)

            max_top = (H - crop_h).clamp(min=0).float()
            max_left = (W - crop_w).clamp(min=0).float()
            top = (torch.rand(B, device=x.device) * max_top).to(torch.long)
            left = (torch.rand(B, device=x.device) * max_left).to(torch.long)

            # Loop over B only; x[i] is (To, C, H, W) so tvf.resized_crop
            # applies the same crop to every timestep in one call.
            crops = []
            for i in range(B):
                crops.append(tvf.resized_crop(
                    x[i],
                    top=top[i].item(),
                    left=left[i].item(),
                    height=crop_h[i].item(),
                    width=crop_w[i].item(),
                    size=self.output_size,
                ))
            out = torch.stack(crops, dim=0)  # (B, To, C, out_h, out_w)

            scale_h = out_h / crop_h.float()
            scale_w = out_w / crop_w.float()
        else:
            # Center crop
            out = tvf.center_crop(
                x.reshape(B * To, C, H, W), self.output_size
            ).reshape(B, To, C, out_h, out_w)
            t = (H - out_h) // 2
            l = (W - out_w) // 2
            top = torch.full((B,), t, dtype=torch.long, device=x.device)
            left = torch.full((B,), l, dtype=torch.long, device=x.device)
            crop_h = torch.full((B,), out_h, dtype=torch.long, device=x.device)
            crop_w = torch.full((B,), out_w, dtype=torch.long, device=x.device)
            scale_h = torch.ones(B, device=x.device)
            scale_w = torch.ones(B, device=x.device)

        return out, {
            "top": top,
            "left": left,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "scale_h": scale_h,
            "scale_w": scale_w,
        }


class ImageAugmentor(nn.Module):
    """
    Photometric augmentations

    Random parameters are sampled once per (batch, camera) and applied
    identically across all observation timesteps so temporal signal is preserved.

    Expects input in **[0, 1]**.  Only active during ``training``.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        blur_kernel_size: int = 5,
        blur_sigma: Tuple[float, float] = (0.1, 2.0),
        blur_p: float = 0.2,
        posterize_bits: int = 4,
        posterize_p: float = 0.1,
        sharpness_factor: float = 1.5,
        sharpness_p: float = 0.2,
        grayscale_p: float = 0.05,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.blur_p = blur_p
        self.posterize_bits = posterize_bits
        self.posterize_p = posterize_p
        self.sharpness_factor = sharpness_factor
        self.sharpness_p = sharpness_p
        self.grayscale_p = grayscale_p

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, To, S, C, H, W) in [0,1].
               Params are sampled per (B, S) and broadcast across To.

        Returns:
            (B, To, S, C, H, W) augmented, in [0,1].
        """
        if not self.training:
            return x

        B, To, S, C, H, W = x.shape
        device = x.device


        # --- colour jitter ---
        # Factors are (B, 1, S, 1, 1, 1) so they broadcast across To.
        if self.brightness > 0:
            f = torch.empty(B, 1, S, 1, 1, 1, device=device).uniform_(
                1 - self.brightness, 1 + self.brightness
            )
            x = (x * f).clamp_(0, 1)

        if self.contrast > 0:
            f = torch.empty(B, 1, S, 1, 1, 1, device=device).uniform_(
                1 - self.contrast, 1 + self.contrast
            )
            mean = x.mean(dim=(-3, -2, -1), keepdim=True)
            x = (mean + f * (x - mean)).clamp_(0, 1)

        if self.saturation > 0:
            f = torch.empty(B, 1, S, 1, 1, 1, device=device).uniform_(
                1 - self.saturation, 1 + self.saturation
            )
            gray = tvf.rgb_to_grayscale(
                x.reshape(-1, C, H, W)
            ).reshape(B, To, S, 1, H, W)
            x = (gray + f * (x - gray)).clamp_(0, 1)

        # For masked ops: sample per (B, S), expand across To, flatten
        # to index into (B*To*S, C, H, W).
        x_flat = x.reshape(-1, C, H, W)

        def _mask_bs(p: float) -> Tensor:
            """Sample per (B, S) and broadcast across To → flat (B*To*S,)."""
            m = torch.rand(B, S, device=device) < p
            return m[:, None, :].expand(B, To, S).reshape(-1)

        # --- gaussian blur ---
        if self.blur_p > 0:
            mask = _mask_bs(self.blur_p)
            if mask.any():
                sigma = torch.empty(1).uniform_(*self.blur_sigma).item()
                ks = [self.blur_kernel_size, self.blur_kernel_size]
                x_flat[mask] = tvf.gaussian_blur(x_flat[mask], ks, [sigma, sigma])

        # --- posterize ---
        if self.posterize_p > 0:
            mask = _mask_bs(self.posterize_p)
            if mask.any():
                subset = (x_flat[mask] * 255).to(torch.uint8)
                subset = tvf.posterize(subset, self.posterize_bits)
                x_flat[mask] = subset.to(x_flat.dtype).div_(255.0)

        # --- sharpness ---
        if self.sharpness_p > 0:
            mask = _mask_bs(self.sharpness_p)
            if mask.any():
                factor = torch.empty(1).uniform_(0, self.sharpness_factor).item()
                x_flat[mask] = tvf.adjust_sharpness(x_flat[mask], factor)

        # --- grayscale ---
        if self.grayscale_p > 0:
            mask = _mask_bs(self.grayscale_p)
            if mask.any():
                x_flat[mask] = tvf.rgb_to_grayscale(
                    x_flat[mask], num_output_channels=3
                )

        return x_flat.reshape(B, To, S, C, H, W)