"""
Min-SNR Weighting Strategy for Diffusion Model Training.

Reference:
    "Efficient Diffusion Training via Min-SNR Weighting Strategy"
    Hang et al., 2023 — https://arxiv.org/abs/2303.09556

Overview:
    Standard diffusion training applies uniform loss weights across all
    timesteps, but the optimization directions conflict across noise levels,
    slowing convergence. Min-SNR clamps the per-timestep loss weight to a
    maximum of gamma (default=5), balancing contributions from easy (low-noise)
    and hard (high-noise) timesteps.

    SNR(t) = alphas_cumprod[t] / (1 - alphas_cumprod[t])

    Weight formulas by prediction target:
        epsilon  : w(t) = min(SNR(t), gamma) / SNR(t)
        sample   : w(t) = min(SNR(t), gamma)
"""

import torch


def compute_snr(noise_scheduler, timesteps):
    """
    Computes the Signal-to-Noise Ratio (SNR) for given timesteps.

        SNR(t) = alphas_cumprod[t] / (1 - alphas_cumprod[t])

    Args:
        noise_scheduler : DDPMScheduler (or any scheduler exposing
                          `alphas_cumprod` of shape [num_train_timesteps]).
        timesteps       : (B,) LongTensor — sampled diffusion timesteps.

    Returns:
        snr : (B,) FloatTensor of SNR values at the sampled timesteps.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    alpha_t = alphas_cumprod[timesteps]
    snr = alpha_t / (1.0 - alpha_t)
    return snr


def compute_min_snr_weights(noise_scheduler, timesteps, gamma=5.0, prediction_type="epsilon"):
    """
    Computes per-timestep Min-SNR loss weights.

    Args:
        noise_scheduler : DDPMScheduler with `alphas_cumprod` attribute.
        timesteps       : (B,) LongTensor — sampled diffusion timesteps.
        gamma           : SNR clamp value. Default 5.0 (recommended in paper).
        prediction_type : One of "epsilon" or "sample".

    Returns:
        weights : (B,) FloatTensor of per-sample loss weights.

    Raises:
        ValueError if prediction_type is not "epsilon" or "sample".
    """
    snr = compute_snr(noise_scheduler, timesteps)

    if prediction_type == "epsilon":
        # w(t) = min(SNR(t), gamma) / SNR(t)
        weights = torch.clamp(snr, max=gamma) / snr
    elif prediction_type == "sample":
        # w(t) = min(SNR(t), gamma)
        weights = torch.clamp(snr, max=gamma)
    else:
        raise ValueError(
            f"Unsupported prediction_type '{prediction_type}' for Min-SNR weighting. "
            f"Supported: 'epsilon', 'sample'."
        )

    return weights


def apply_min_snr_weighted_loss(per_sample_loss, noise_scheduler, timesteps,
                                gamma=5.0, prediction_type="epsilon"):
    """
    Applies Min-SNR weighting to a per-sample loss tensor and returns
    the final scalar weighted loss.

    Args:
        per_sample_loss : (B,) FloatTensor — mean MSE loss per sample,
                          already reduced over all non-batch dimensions
                          (time, action/feature dims, etc.).
        noise_scheduler : DDPMScheduler with `alphas_cumprod` attribute.
        timesteps       : (B,) LongTensor — sampled diffusion timesteps
                          (same timesteps used to add noise in the forward pass).
        gamma           : SNR clamp value. Default 5.0.
        prediction_type : "epsilon" or "sample".

    Returns:
        loss : scalar FloatTensor — mean of Min-SNR-weighted per-sample losses.
    """
    weights = compute_min_snr_weights(
        noise_scheduler, timesteps, gamma=gamma, prediction_type=prediction_type
    )
    weighted_loss = (per_sample_loss * weights).mean()
    return weighted_loss
