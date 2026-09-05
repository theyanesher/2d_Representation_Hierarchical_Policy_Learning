"""
Goal-regression auxiliary head
===============================
Deterministic alternative to ``GoalGMMHead``: instead of a mixture over every
grounded anchor, pool the same tokens into one feature per (sample, obs_step)
and regress a single subgoal directly. No mixture weights, no variance ladder
-- plain MSE.

The RoPE4D trunk already gives each token positional information via rotary
attention (not by concatenation), so unlike ``GoalGMMHead`` the pooled feature
does not need the anchor XYZ appended to it.

Two target frames, both plausible depending on what should shape the trunk:
    "absolute"            regress the goal keypoints in world coordinates.
    "relative_to_gripper" regress goal - present_gripper_pts, i.e. the
                           displacement the gripper still has to travel. Keeps
                           the regression target on the same scale regardless
                           of where in the workspace the gripper starts.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_policy.model.flow_matching.helpers import SimpleMLP

REGRESSION_FRAMES = ("absolute", "relative_to_gripper")


class GoalRegressionHead(nn.Module):
    """(tokens, valid) -> one predicted subgoal per row: (B, n_keypoints, 3).

    ``tokens``/``valid`` are the same per-(sample, obs_step) anchor stack
    ``GoalGMMHead`` reads (gripper keypoints + patch anchors); this head just
    mean-pools across the anchor axis instead of scoring each anchor.
    """

    def __init__(self, token_dim: int, hidden_dim: int = 512, n_keypoints: int = 4):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.mlp = SimpleMLP(
            input_dim=token_dim, hidden_dim=hidden_dim, output_dim=n_keypoints * 3,
        )

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        """tokens: (B, N, D), valid: (B, N) bool -> (B, n_keypoints, 3)."""
        w = valid.to(tokens.dtype).unsqueeze(-1)               # (B, N, 1)
        pooled = (tokens * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        out = self.mlp(pooled)                                  # (B, K*3)
        return out.reshape(out.shape[0], self.n_keypoints, 3)


def goal_regression_target(
    goal: Tensor, present_gripper: Tensor, frame: str,
) -> Tensor:
    """goal, present_gripper: (..., K, 3) -> the MSE target in ``frame``."""
    if frame == "absolute":
        return goal
    if frame == "relative_to_gripper":
        return goal - present_gripper
    raise ValueError(f"unknown aux_regression_frame {frame!r}, expected one of "
                      f"{REGRESSION_FRAMES}")


def goal_regression_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean-squared error over keypoints and xyz, averaged over rows."""
    return ((pred - target) ** 2).sum(dim=-1).mean()
