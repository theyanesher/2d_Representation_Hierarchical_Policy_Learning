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

use_gmm=True, use_weighted_cross_attention=False — GMM feature mode:
    Input : goal_pts (B, To, N, 4, 3)  — N candidates, each with 4 keypoints
            weights  (B, To, N)         — softmax probability of each candidate
    Output: (B, To * N, D)             — one token per candidate per timestep

    Pipeline (per candidate):
        flatten (4, 3) → 12, append weight → 13
        Linear(13, hidden_dim) → ReLU → Linear(hidden_dim, embed_dim)
        + learned temporal embedding (which obs step)

use_gmm=True, use_weighted_cross_attention=True — GMM weighted cross-attention mode:
    Input : goal_pts (B, To, N, 4, 3)
            weights  (B, To, N)
    Output: tokens  (B, To * N, D)   — encoded WITHOUT weight appended
            weights (B, To * N)      — flat weights for log-bias in cross-attention

    Pipeline (per candidate):
        flatten (4, 3) → 12
        Linear(12, hidden_dim) → ReLU → Linear(hidden_dim, embed_dim)
        + learned temporal embedding (which obs step)
    Weights are returned separately for use in WeightedCrossAttention.
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
        use_weighted_cross_attention: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.n_keypoints = n_keypoints
        self.n_obs_steps = n_obs_steps
        self.use_gmm = use_gmm
        self.use_weighted_cross_attention = use_weighted_cross_attention

        if use_gmm:
            if use_weighted_cross_attention:
                # Encode keypoints only — weight is returned separately for log-bias attention.
                candidate_dim = n_keypoints * 3
            else:
                # Append weight as an input feature.
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
    ):
        """
        Parameters
        ----------
        goal_pts : (B, To, 4, 3)         when use_gmm=False
                   (B, To, N, 4, 3)      when use_gmm=True
        weights  : (B, To, N)            required when use_gmm=True, ignored otherwise

        Returns
        -------
        use_gmm=False or (use_gmm=True, use_weighted_cross_attention=False):
            tokens : (B, To*4, D) or (B, To*N, D)

        use_gmm=True, use_weighted_cross_attention=True:
            (tokens, flat_weights) : (B, To*N, D), (B, To*N)
            flat_weights are the raw GMM probabilities flattened across (To, N),
            ready to be passed as log-bias in WeightedCrossAttention.
        """
        device = goal_pts.device

        if self.use_gmm:
            # import pdb; pdb.set_trace()
            B, To, N, K, _ = goal_pts.shape
            assert weights is not None, "weights required when use_gmm=True"

            flat = goal_pts.reshape(B, To, N, K * 3)              # (B, To, N, 12)

            if self.use_weighted_cross_attention:
                # Encode without weight feature; return weights separately.
                toks = self.point_encoder(flat.reshape(B * To * N, -1))  # (B*To*N, D)
                toks = toks.reshape(B, To, N, -1)                        # (B, To, N, D)

                t_ids = torch.arange(To, device=device)
                toks = toks + self.temporal_embed(t_ids)[None, :, None, :]

                flat_weights = weights.reshape(B, To * N)          # (B, To*N)
                # import pdb; pdb.set_trace()
                return toks.reshape(B, To * N, -1), flat_weights
            else:
                # Append weight as input feature → (N, 13)
                w = weights.unsqueeze(-1)                          # (B, To, N, 1)
                flat = torch.cat([flat, w], dim=-1)                # (B, To, N, 13)

                toks = self.point_encoder(flat.reshape(B * To * N, -1))  # (B*To*N, D)
                toks = toks.reshape(B, To, N, -1)                        # (B, To, N, D)

                t_ids = torch.arange(To, device=device)
                toks = toks + self.temporal_embed(t_ids)[None, :, None, :]

                return toks.reshape(B, To * N, -1)                 # (B, To*N, D)

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
