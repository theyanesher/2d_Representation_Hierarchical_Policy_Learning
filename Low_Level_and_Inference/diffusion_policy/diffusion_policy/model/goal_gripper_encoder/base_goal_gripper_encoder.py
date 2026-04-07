"""
BaseGoalGripperEncoder

Encodes goal gripper keypoints into a single DiT token per observation timestep.

Input : (B, To, 4, 3)  — 4 keypoints × 3D coords, To obs steps
Output: (B, To, D)     — one token per timestep, concatenated into the
                          DiT hidden_state sequence alongside state tokens.

Pipeline:
    flatten (4, 3) → 12  →  Linear(12, hidden_dim)  →  ReLU  →  Linear(hidden_dim, embed_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGoalGripperEncoder(nn.Module):
    """
    One token per observation timestep.

    Parameters
    ----------
    embed_dim  : int — must match the DiT input_embedding_dim
    hidden_dim : int — MLP hidden dimension (default: embed_dim)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, goal_pts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        goal_pts : (B, To, 4, 3)

        Returns
        -------
        tokens : (B, To, embed_dim)
        """
        B, To, _, _ = goal_pts.shape
        flat = goal_pts.reshape(B, To, 12)   # (B, To, 12)
        return self.encoder(flat)            # (B, To, D)
