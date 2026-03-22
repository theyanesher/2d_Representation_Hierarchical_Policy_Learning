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
