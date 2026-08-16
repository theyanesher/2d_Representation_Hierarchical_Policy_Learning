"""
Goal-GMM auxiliary head
=======================
Per-token readout that turns grounded visual tokens into a Gaussian mixture over
candidate goal gripper poses, plus the negative log-likelihood used to train it.

Mixture structure (mirrors ArticubotNetwork / GoalRegressionModule):
    mu_n   = anchor_n + Delta_n        anchor broadcasts across all 4 keypoints
    pi     = softmax(logits over the N anchors)
    sigma  = NOT predicted; the fixed ladder below is summed over instead

The NLL is a verbatim port of ``ArticuBot.nll_loss`` — same dropped Gaussian
normaliser, same -10 clamp on the log mixing coefficients, same two-stage
masking. Keeping it identical means any behavioural difference is attributable to
the anchor set rather than to a reimplementation.
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_policy.model.flow_matching.helpers import SimpleMLP

FIXED_VARIANCE: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5)
UNIFORM_WEIGHTS_COEFF: float = 0.1


class GoalGMMHead(nn.Module):
    """(tokens, anchors) -> (per-anchor displacement, per-anchor weight logit).

    Applied independently to every token — all cross-token reasoning already
    happened in the RoPE4D trunk.

    The anchor XYZ is concatenated to the token because RoPE is purely relative
    and leaves no positional trace in the token itself, yet the head has to
    predict ``goal - anchor``.

    The output layer gets a SMALL (not zero) initialisation so that
    ``mu_n ~= anchor_n`` at step 0 — the right prior, and it keeps the initial
    NLL finite: default Linear init would emit ~1 m displacements, driving the
    sigma^2=0.01 exponent to ~-600.

    Exact zero-init would give the same prior but kills the gradient path: with
    ``W2 == 0``, ``dL/dh = W2^T delta == 0``, so nothing reaches the first layer
    or the trunk behind it. It recovers after one optimiser step, but the whole
    point of the auxiliary loss is to shape the trunk, so don't start it inert.
    """

    OUT_INIT_STD: float = 1e-3

    def __init__(self, token_dim: int, hidden_dim: int = 512, n_keypoints: int = 4):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.mlp = SimpleMLP(
            input_dim=token_dim + 3,
            hidden_dim=hidden_dim,
            output_dim=n_keypoints * 3 + 1,
        )
        nn.init.normal_(self.mlp.layer2.weight, std=self.OUT_INIT_STD)
        nn.init.zeros_(self.mlp.layer2.bias)

    def forward(self, tokens: Tensor, anchors: Tensor) -> Tuple[Tensor, Tensor]:
        """tokens: (B, N, D), anchors: (B, N, 3) -> (B, N, K, 3), (B, N)."""
        out = self.mlp(torch.cat([tokens, anchors], dim=-1))
        disp = out[..., :-1].reshape(*out.shape[:-1], self.n_keypoints, 3)
        return disp, out[..., -1]


def gmm_nll_loss(
    pred_displacement: Tensor,
    gt_displacement: Tensor,
    weights: Tensor,
    valid_mask: Tensor,
    variance: float,
    use_weights: bool = True,
) -> Tensor:
    """Negative log-likelihood of the goal under the anchored mixture.

    Args:
        pred_displacement: (B, N, K, 3) predicted Delta_n
        gt_displacement:   (B, N, K, 3) goal - anchor_n
        weights:           (B, N) raw logits (log_softmax applied here)
        valid_mask:        (B, N) bool, True where the anchor is usable
        variance:          scalar sigma^2 for this rung of the ladder
        use_weights:       False replaces the predicted weights with uniform
                           ones, which forces EVERY valid anchor to regress a
                           good displacement rather than only the favoured ones.

    Returns:
        scalar loss, averaged over rows that have at least one valid anchor.
    """
    batch_size, pcd_size = pred_displacement.shape[:2]

    # Rows with no valid anchor would make log_softmax all -inf -> NaN. Force one
    # slot on so the maths stays finite, then drop those rows from the mean.
    has_valid = valid_mask.any(dim=1)
    safe_mask = valid_mask.clone()
    safe_mask[~has_valid, 0] = True

    weights = weights.masked_fill(~safe_mask, float("-inf"))
    if use_weights is False:
        weights = weights.masked_fill(safe_mask, 1)

    diff = (pred_displacement - gt_displacement).reshape(batch_size, pcd_size, -1)
    exponent = -0.5 * torch.sum((diff ** 2) / variance, dim=2)   # (B, N)
    log_gaussians = exponent

    log_mixing_coeffs = torch.log_softmax(weights, dim=1)
    log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-10)

    masked_sum = log_gaussians + log_mixing_coeffs
    # NOTE: order matters. The clamp above RAISES masked entries from -inf to
    # -10, so they must be re-masked here or dead anchors rejoin the mixture.
    masked_sum = masked_sum.masked_fill(~safe_mask, -1e9)

    max_log = torch.max(masked_sum, dim=1, keepdim=True).values
    log_probs = max_log.squeeze(1) + torch.logsumexp(masked_sum - max_log, dim=1)

    n = has_valid.sum().clamp(min=1)
    return -(log_probs * has_valid).sum() / n


def goal_gmm_loss(
    pred_displacement: Tensor,
    gt_displacement: Tensor,
    weights: Tensor,
    valid_mask: Tensor,
    variances: Sequence[float] = FIXED_VARIANCE,
    uniform_weights_coeff: float = UNIFORM_WEIGHTS_COEFF,
) -> Tensor:
    """Full objective: the NLL summed over the variance ladder, each rung paired
    with a uniform-weights term. Ten terms in total for the default ladder."""
    loss = pred_displacement.new_zeros(())
    for var in variances:
        loss = loss + gmm_nll_loss(
            pred_displacement, gt_displacement, weights, valid_mask, var,
            use_weights=True,
        )
        loss = loss + uniform_weights_coeff * gmm_nll_loss(
            pred_displacement, gt_displacement, weights, valid_mask, var,
            use_weights=False,
        )
    return loss
