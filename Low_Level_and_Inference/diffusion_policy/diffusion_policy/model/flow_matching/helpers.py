# Flow matching helper modules for FlowMatchingDiTImagePolicy.
# Extracted from Isaac-GR00T's simple_flow_matching_action_head.py.
# Only depends on torch — no gr00t package imports needed.

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape (B, T, embedding_dim) given timesteps (B, T)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class ActionEncoder(nn.Module):
    """
    Encode a noisy action trajectory + diffusion timestep into the DiT token space.

    Mirrors MultiEmbodimentActionEncoder from GR00T but uses plain nn.Linear
    (single embodiment).

    Input:
        actions   : (B, T, action_dim)
        timesteps : (B,)  discrete timestep per batch item
    Output:
        (B, T, hidden_size)
    """

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        ts = timesteps.unsqueeze(1).expand(-1, T).float()   # (B, T)
        a = self.W1(actions)                                 # (B, T, D)
        tau = self.pos_encoding(ts).to(a.dtype)              # (B, T, D)
        x = swish(self.W2(torch.cat([a, tau], dim=-1)))     # (B, T, D)
        return self.W3(x)                                    # (B, T, D)


def make_2d_sinusoidal_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    """
    2-D sinusoidal positional embedding for an h×w spatial grid.
    Returns a fixed (non-trainable) tensor of shape (h*w, dim).
    First dim//2 channels encode the row index; last dim//2 encode the column index.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2

    def _1d_sin_cos(length: int, channels: int) -> torch.Tensor:
        half_ch = channels // 2
        exponent = -torch.arange(half_ch, dtype=torch.float32) * (
            torch.log(torch.tensor(10000.0)) / half_ch
        )
        pos = torch.arange(length, dtype=torch.float32)
        freqs = torch.outer(pos, exponent.exp())           # (length, half_ch)
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # (length, channels)

    row_embed = _1d_sin_cos(h, half).unsqueeze(1).expand(h, w, half)  # (h, w, dim//2)
    col_embed = _1d_sin_cos(w, half).unsqueeze(0).expand(h, w, half)  # (h, w, dim//2)
    return torch.cat([row_embed, col_embed], dim=-1).reshape(h * w, dim)  # (h*w, dim)


class SimpleMLP(nn.Module):
    """Two-layer ReLU MLP (works on any leading batch/sequence dims)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class PerComponentStateEncoder(nn.Module):
    """
    Encode the 10-D MimicGen Panda state as THREE sub-tokens per obs step:
        - position  (3,)   → token + learned [POS]  identity embedding
        - rotation  (6,)   → token + learned [ROT]  identity embedding   (6D rep: cols 1+2 of R)
        - gripper   (1,)   → token + learned [GRIP] identity embedding
    Each component also receives a learned temporal embedding (which obs step).

    Input:  state (B, To, 10)   layout = [pos(3), rot_col1(3), rot_col2(3), gripper(1)]
    Output: tokens (B, To*3, D)

    Compared to a single SimpleMLP(10→D→D) that produces (B, To, D):
      - 3× more state tokens (To*3 instead of To) → better attention bandwidth
      - explicit per-component structure (learned identity embeddings) → the network
        doesn't have to discover from flat dims that "indices 0-2 are position"

    Assumes the standard 10-D Panda EE-space layout (3 + 6 + 1). Asserts the input
    has the expected shape; raises if mismatched.
    """

    def __init__(
        self,
        embed_dim: int,
        n_obs_steps: int,
        hidden_dim: int = None,
        pos_dim: int = 3,
        rot_dim: int = 6,
        gripper_dim: int = 1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.pos_dim = pos_dim
        self.rot_dim = rot_dim
        self.gripper_dim = gripper_dim
        self.state_dim = pos_dim + rot_dim + gripper_dim
        self.n_obs_steps = n_obs_steps

        # Per-component encoders. Each is a small 2-layer MLP raw-dim → D.
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.rot_encoder = nn.Sequential(
            nn.Linear(rot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gripper_encoder = nn.Sequential(
            nn.Linear(gripper_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Learned identity embedding: which component (0=pos, 1=rot, 2=gripper).
        # Lets the network distinguish sub-tokens by their semantic role.
        self.component_embed = nn.Embedding(3, embed_dim)
        nn.init.normal_(self.component_embed.weight, std=0.02)

        # Learned temporal embedding: which obs step (0..n_obs_steps-1).
        self.temporal_embed = nn.Embedding(n_obs_steps, embed_dim)
        nn.init.normal_(self.temporal_embed.weight, std=0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (B, To, 10)

        Returns
        -------
        tokens : (B, To*3, D)  — order = [pos_t0, rot_t0, grip_t0, pos_t1, rot_t1, grip_t1, ...]
        """
        B, To, D = state.shape
        assert D == self.state_dim, (
            f"PerComponentStateEncoder expects state_dim={self.state_dim}, got {D}. "
            f"Reconfigure pos_dim/rot_dim/gripper_dim if your state layout differs."
        )
        assert To == self.n_obs_steps, (
            f"Expected n_obs_steps={self.n_obs_steps}, got To={To}."
        )

        pos     = state[..., 0 : self.pos_dim]                                    # (B, To, 3)
        rot     = state[..., self.pos_dim : self.pos_dim + self.rot_dim]          # (B, To, 6)
        gripper = state[..., self.pos_dim + self.rot_dim :]                       # (B, To, 1)

        pos_tok     = self.pos_encoder(pos)         # (B, To, D)
        rot_tok     = self.rot_encoder(rot)         # (B, To, D)
        gripper_tok = self.gripper_encoder(gripper) # (B, To, D)

        # (B, To, 3, D)
        tokens = torch.stack([pos_tok, rot_tok, gripper_tok], dim=2)

        # Identity embedding (which component)
        component_ids = torch.arange(3, device=state.device)
        tokens = tokens + self.component_embed(component_ids)[None, None, :, :]

        # Temporal embedding (which obs step)
        t_ids = torch.arange(To, device=state.device)
        tokens = tokens + self.temporal_embed(t_ids)[None, :, None, :]

        return tokens.reshape(B, To * 3, -1)        # (B, To*3, D)
