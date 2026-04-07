"""
KeypointAwareGoalGripperEncoder

Encodes goal gripper keypoints into one DiT token per keypoint per timestep.

Input : (B, To, 4, 3)  — 4 keypoints × 3D coords, To obs steps
Output: (B, To * 4, D) — one token per keypoint per timestep, concatenated
                          into the DiT hidden_state sequence alongside state tokens.

Pipeline (per keypoint):
    Linear(3, hidden_dim)  →  ReLU  →  Linear(hidden_dim, embed_dim)
    + learned keypoint identity embedding (which of the 4 keypoints)
    + learned timestep embedding (which obs step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointAwareGoalGripperEncoder(nn.Module):
    """
    One token per keypoint per observation timestep.

    Parameters
    ----------
    embed_dim   : int — must match the DiT input_embedding_dim
    hidden_dim  : int — MLP hidden dimension (default: embed_dim)
    n_keypoints : int — number of gripper keypoints (default: 4)
    n_obs_steps : int — number of observation timesteps (for temporal embed)
    """

    def __init__(
        self,
        embed_dim: int,
        n_obs_steps: int,
        hidden_dim: int = None,
        n_keypoints: int = 4,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.n_keypoints = n_keypoints
        self.n_obs_steps = n_obs_steps

        # Project each 3D keypoint coordinate into embed_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Learned identity embed: which keypoint (0-3)
        self.keypoint_embed = nn.Embedding(n_keypoints, embed_dim)
        nn.init.normal_(self.keypoint_embed.weight, std=0.02)

        # Learned temporal embed: which obs step
        self.temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.temporal_embed.weight, std=0.02)

    def forward(self, goal_pts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        goal_pts : (B, To, 4, 3)

        Returns
        -------
        tokens : (B, To * 4, embed_dim)
        """
        B, To, K, _ = goal_pts.shape
        device = goal_pts.device

        # Encode each keypoint coordinate
        flat = goal_pts.reshape(B * To * K, 3)
        toks = self.point_encoder(flat)               # (B*To*K, D)
        toks = toks.reshape(B, To, K, -1)             # (B, To, K, D)

        # Add keypoint identity embedding
        kp_ids = torch.arange(K, device=device)       # (K,)
        toks = toks + self.keypoint_embed(kp_ids)     # broadcast over (B, To)

        # Add temporal embedding
        t_ids = torch.arange(To, device=device)       # (To,)
        toks = toks + self.temporal_embed(t_ids)[None, :, None, :]  # (1, To, 1, D)

        # Flatten To and K into sequence dimension
        return toks.reshape(B, To * K, -1)            # (B, To*K, D)
