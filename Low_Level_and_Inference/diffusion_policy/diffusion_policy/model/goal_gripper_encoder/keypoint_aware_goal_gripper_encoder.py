"""
KeypointAwareGoalGripperEncoder

Two modes controlled by use_gmm:

use_gmm=False (default) — goal_gripper mode:
    Input : (B, To, 4, 3)      — 4 keypoints × 3D coords, To obs steps
    Output: (B, To * 4, D)     — one token per keypoint per timestep

    Pipeline (per keypoint):
        Linear(3, hidden_dim) → ReLU → Linear(hidden_dim, embed_dim)
        + learned keypoint identity embedding (which of the 4 keypoints)
        + learned timestep embedding (which obs step)

use_gmm=True — GMM distribution mode:
    Input : goal_pts (B, To, N, 4, 3)  — N candidates, each with 4 keypoints
            weights  (B, To, N)         — softmax probability of each candidate
    Output: (B, To * N, D)             — one token per candidate per timestep

    Pipeline (per candidate):
        flatten (4, 3) → 12, append weight → 13
        Linear(13, hidden_dim) → ReLU → Linear(hidden_dim, embed_dim)
        + learned temporal embedding (which obs step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointAwareGoalGripperEncoder(nn.Module):
    """
    Parameters
    ----------
    embed_dim   : int  — must match the DiT input_embedding_dim
    n_obs_steps : int  — number of observation timesteps (for temporal embed)
    hidden_dim  : int  — MLP hidden dimension (default: embed_dim)
    n_keypoints : int  — number of gripper keypoints per goal (default: 4)
    use_gmm     : bool — if True, encode full GMM distribution (N candidates + weights)
                         instead of a single goal (4 keypoints)
    """

    def __init__(
        self,
        embed_dim: int,
        n_obs_steps: int,
        hidden_dim: int = None,
        n_keypoints: int = 4,
        use_gmm: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.n_keypoints = n_keypoints
        self.n_obs_steps = n_obs_steps
        self.use_gmm = use_gmm

        if use_gmm:
            # Each candidate: flatten (n_keypoints, 3) → n_keypoints*3, append weight → +1
            candidate_dim = n_keypoints * 3 + 1
            self.point_encoder = nn.Sequential(
                nn.Linear(candidate_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            # Encode each 3D keypoint independently
            self.point_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            # Learned identity embed: which keypoint (0 to n_keypoints-1)
            self.keypoint_embed = nn.Embedding(n_keypoints, embed_dim)
            nn.init.normal_(self.keypoint_embed.weight, std=0.02)

        # Learned temporal embed: which obs step (shared across both modes)
        self.temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.temporal_embed.weight, std=0.02)

    def forward(
        self,
        goal_pts: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        goal_pts : (B, To, 4, 3)         when use_gmm=False
                   (B, To, N, 4, 3)      when use_gmm=True
        weights  : (B, To, N)            required when use_gmm=True, ignored otherwise

        Returns
        -------
        tokens : (B, To * 4, D)          when use_gmm=False
                 (B, To * N, D)          when use_gmm=True
        """
        device = goal_pts.device

        if self.use_gmm:
            B, To, N, K, _ = goal_pts.shape
            assert weights is not None, "weights required when use_gmm=True"

            # Flatten (N, 4, 3) → (N, 12), append weight → (N, 13)
            flat = goal_pts.reshape(B, To, N, K * 3)              # (B, To, N, 12)
            w = weights.unsqueeze(-1)                              # (B, To, N, 1)
            flat = torch.cat([flat, w], dim=-1)                    # (B, To, N, 13)

            toks = self.point_encoder(flat.reshape(B * To * N, -1))  # (B*To*N, D)
            toks = toks.reshape(B, To, N, -1)                        # (B, To, N, D)

            # Add temporal embedding
            t_ids = torch.arange(To, device=device)
            toks = toks + self.temporal_embed(t_ids)[None, :, None, :]  # (1, To, 1, D)

            return toks.reshape(B, To * N, -1)                     # (B, To*N, D)

        else:
            B, To, K, _ = goal_pts.shape

            # Encode each keypoint coordinate independently
            flat = goal_pts.reshape(B * To * K, 3)
            toks = self.point_encoder(flat)                        # (B*To*K, D)
            toks = toks.reshape(B, To, K, -1)                     # (B, To, K, D)

            # Add keypoint identity embedding
            kp_ids = torch.arange(K, device=device)
            toks = toks + self.keypoint_embed(kp_ids)             # broadcast over (B, To)

            # Add temporal embedding
            t_ids = torch.arange(To, device=device)
            toks = toks + self.temporal_embed(t_ids)[None, :, None, :]  # (1, To, 1, D)

            return toks.reshape(B, To * K, -1)                    # (B, To*K, D)
