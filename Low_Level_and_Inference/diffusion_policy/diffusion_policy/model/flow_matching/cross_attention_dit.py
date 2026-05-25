# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class GoalAdaLN(nn.Module):
    """
    Per-token AdaLN with two conditioning branches:
      - temb     (B, D)        → per-channel (γ_t,    β_t)      — standard AdaLN
      - goal_ctx (B, T, D)     → per-token   (γ_goal, β_goal)   — zero-init at start

    Final modulation (additive composition, à la DiT class+time conditioning):
        x_mod = LN(x) · (1 + γ_t + γ_goal) + (β_t + β_goal)

    At init the goal branch's Linear is zero-init'd, so γ_goal = β_goal = 0 and
    the behavior reduces exactly to standard temb-only AdaLN. The optimizer grows
    the goal modulation only if it actually reduces loss.

    Used by BasicTransformerBlock when `use_goal_adaln_modulation=True` — in that
    mode the WCA output is captured as goal_ctx and feeds *here* instead of being
    added to the residual stream.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        # temb branch: matches the default AdaLayerNorm init exactly, so at step 0
        # this layer behaves identically to a standard AdaLayerNorm conditioned on temb.
        self.linear_t = nn.Linear(embedding_dim, embedding_dim * 2)
        # goal branch: zero-init so γ_goal = β_goal = 0 at step 0 regardless of goal_ctx.
        self.linear_goal = nn.Linear(embedding_dim, embedding_dim * 2)
        nn.init.zeros_(self.linear_goal.weight)
        nn.init.zeros_(self.linear_goal.bias)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, D)
        temb: torch.Tensor,                    # (B, D)
        goal_ctx: Optional[torch.Tensor] = None,  # (B, T, D) — None on self-attn blocks
    ) -> torch.Tensor:
        # temb → per-channel (γ_t, β_t), broadcast over T
        scale_t, shift_t = self.linear_t(self.silu(temb)).chunk(2, dim=-1)      # (B, D), (B, D)
        if goal_ctx is None:
            # No goals available (e.g. self-attn block) — behave as standard temb-AdaLN.
            x = self.norm(x) * (1 + scale_t[:, None, :]) + shift_t[:, None, :]
        else:
            # goal_ctx → per-token (γ_goal, β_goal). Linear applies pointwise across T.
            scale_g, shift_g = self.linear_goal(self.silu(goal_ctx)).chunk(2, dim=-1)  # (B, T, D), (B, T, D)
            x = self.norm(x) * (1 + scale_t[:, None, :] + scale_g) + (shift_t[:, None, :] + shift_g)
        return x


class StateAdaLN(nn.Module):
    """
    AdaLN with two conditioning branches:
      - temb      (B, D)  → per-channel (γ_t, β_t)         — standard AdaLN
      - state_ctx (B, D)  → per-channel (γ_state, β_state) — zero contribution at init

    Final modulation (additive composition, like DiT's class+time conditioning):
        x_mod = LN(x) · (1 + γ_t + α_state · γ_state) + (β_t + α_state · β_state)

    Used as `norm1` inside BasicTransformerBlock when use_state_adaln=True.

    Two parameterizations control how the state branch starts at "no contribution":

    use_gate=False (default — Linear-zero-init parameterization):
        - `linear_state.weight` and `.bias` are zero-init'd.
        - `α_state` is implicit = 1 (no parameter allocated).
        - At step 0: state branch outputs 0 → behavior = standard temb-AdaLN.
        - During training: `linear_state` grows from zero (both magnitude and direction
          come from the same projection's gradients).

    use_gate=True (ReZero parameterization):
        - `linear_state` uses default random init (direction is "ready" from step 0).
        - `α_state = nn.Parameter(zeros(1))` is the gate; init = 0.
        - At step 0: state branch outputs 0 because α=0 → behavior = standard temb-AdaLN.
        - During training: `α_state` grows from zero (controls magnitude); the linear's
          random direction explores freely.
        - Pros: decoupled magnitude/direction; single-scalar diagnostic per block
          ("how strongly does this block want state modulation"); implicit per-block
          sparsity (blocks that don't want state can train α → 0).

    The two parameterizations are NOT used together — zero-init linear with also
    zero-init gate would freeze the entire state branch forever (dead path).
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        use_gate: bool = False,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.silu = nn.SiLU()
        # temb branch — same shape as standard AdaLayerNorm's linear so at step 0,
        # combined with zero state contribution, behaves identically.
        self.linear_t = nn.Linear(embedding_dim, embedding_dim * 2)
        self.linear_state = nn.Linear(embedding_dim, embedding_dim * 2)
        if use_gate:
            # ReZero: keep default random init on linear_state (direction is "ready"),
            # and let the scalar gate (init=0) control magnitude.
            self.state_gate = nn.Parameter(torch.zeros(1))
        else:
            # Zero-init the linear; no explicit gate parameter.
            nn.init.zeros_(self.linear_state.weight)
            nn.init.zeros_(self.linear_state.bias)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, D)
        temb: torch.Tensor,                       # (B, D)
        state_ctx: Optional[torch.Tensor] = None, # (B, D) — pre-pooled state summary
    ) -> torch.Tensor:
        scale_t, shift_t = self.linear_t(self.silu(temb)).chunk(2, dim=-1)        # (B, D), (B, D)
        if state_ctx is None:
            # No state context available — behave as standard temb-AdaLN.
            x = self.norm(x) * (1 + scale_t[:, None, :]) + shift_t[:, None, :]
        else:
            scale_s, shift_s = self.linear_state(self.silu(state_ctx)).chunk(2, dim=-1)
            if self.use_gate:
                # ReZero gate (per-block scalar) scales the state branch's contribution.
                scale_s = self.state_gate * scale_s
                shift_s = self.state_gate * shift_s
            x = self.norm(x) * (1 + scale_t[:, None, :] + scale_s[:, None, :]) \
                            + (shift_t[:, None, :] + shift_s[:, None, :])
        return x


class WeightedCrossAttention(nn.Module):
    """
    Multi-head cross-attention with per-key log-prior from GMM weights.

    Implements:
        logits_{i,j} = (Q_i · K_j) / sqrt(d_k) + log(w_j)
        attn         = softmax(logits, dim=-1)
        out_i        = sum_j  attn_{i,j} · V_j

    This is equivalent to multiplying the un-normalised attention scores by w_j
    before the softmax, i.e. high-probability GMM candidates attract more
    attention regardless of the query content.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = num_heads * head_dim

        self.to_q   = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k   = nn.Linear(key_dim,   inner_dim, bias=bias)
        self.to_v   = nn.Linear(key_dim,   inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=bias)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, T_q, query_dim)
        encoder_hidden_states: torch.Tensor,  # (B, N_k, key_dim)
        weights: torch.Tensor,                # (B, N_k)  GMM probabilities
    ) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        B, T_q, _ = hidden_states.shape
        N_k = encoder_hidden_states.shape[1]
        H, Dh = self.num_heads, self.head_dim

        Q = self.to_q(hidden_states).view(B, T_q, H, Dh).transpose(1, 2)   # (B,H,T_q,Dh)
        K = self.to_k(encoder_hidden_states).view(B, N_k, H, Dh).transpose(1, 2)  # (B,H,N_k,Dh)
        V = self.to_v(encoder_hidden_states).view(B, N_k, H, Dh).transpose(1, 2)  # (B,H,N_k,Dh)

        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,H,T_q,N_k)

        # Add log-weight bias: boost high-probability keys, suppress low-probability ones.
        log_w = torch.log(weights.clamp(min=1e-8))        # (B, N_k)
        logits = logits + log_w[:, None, None, :]          # broadcast over H and T_q

        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)                        # (B,H,T_q,Dh)
        out = out.transpose(1, 2).contiguous().view(B, T_q, H * Dh)
        # import pdb; pdb.set_trace()
        return self.to_out(out)                            # (B,T_q,query_dim)


class GoalAuxiliaryAttention(nn.Module):
    """
    Joint self-attention over TWO sequences (a "main" stream and an "auxiliary"
    goal stream) with separate per-stream Q/K/V/out projections.

    Each stream goes through its own Wq/Wk/Wv (so the projections are tuned to
    each stream's statistics), the per-head Q/K/V are concatenated along the
    sequence dimension, a single softmax runs over the joined sequence, then
    the output is split back and sent through per-stream Wout projections.

    Shapes:
        x1: (B, T1, D)   — stream 1 (main / action stream)
        x2: (B, T2, D)   — stream 2 (auxiliary / goal stream)
        return: (out1, out2) with shapes (B, T1, D), (B, T2, D)

    The single softmax means stream-1 queries compete against ALL keys (own and
    auxiliary) in one normalization — that's the key property vs. running two
    separate attentions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = num_heads * head_dim

        # Per-stream Q/K/V projections (separate weights → each stream learns
        # its own "way of asking and answering questions").
        self.to_q1 = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k1 = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=bias)
        self.to_q2 = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k2 = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=bias)

        # Per-stream output projections.
        self.to_out1 = nn.Linear(inner_dim, dim, bias=bias)
        self.to_out2 = nn.Linear(inner_dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        B, T1, _ = x1.shape
        _, T2, _ = x2.shape
        H, Dh = self.num_heads, self.head_dim

        # Per-stream Q/K/V → (B, H, T, Dh)
        Q1 = self.to_q1(x1).view(B, T1, H, Dh).transpose(1, 2)
        K1 = self.to_k1(x1).view(B, T1, H, Dh).transpose(1, 2)
        V1 = self.to_v1(x1).view(B, T1, H, Dh).transpose(1, 2)
        Q2 = self.to_q2(x2).view(B, T2, H, Dh).transpose(1, 2)
        K2 = self.to_k2(x2).view(B, T2, H, Dh).transpose(1, 2)
        V2 = self.to_v2(x2).view(B, T2, H, Dh).transpose(1, 2)

        # Concat along sequence dim — one softmax over (T1 + T2) keys.
        Q = torch.cat([Q1, Q2], dim=2)        # (B, H, T1+T2, Dh)
        K = torch.cat([K1, K2], dim=2)
        V = torch.cat([V1, V2], dim=2)

        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, H, T1+T2, T1+T2)
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, V)            # (B, H, T1+T2, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T1 + T2, H * Dh)

        # Split back to each stream and run its own Wout.
        out1 = self.to_out1(out[:, :T1])
        out2 = self.to_out2(out[:, T1:])
        return out1, out2


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        # Goal cross-attention: runs before visual cross-attention when enabled.
        use_goal_cross_attention: bool = False,
        goal_cross_attention_dim: Optional[int] = None,
        # When True, goal cross-attention uses WeightedCrossAttention (log-bias from GMM weights)
        # instead of standard Attention. Requires use_goal_cross_attention=True.
        use_weighted_cross_attention: bool = False,
        # When True (and use_goal_cross_attention=True), goal-CA and visual-CA run in
        # parallel: both read the *pre-block* hidden_states through their own AdaLN,
        # their outputs are concatenated along the feature dim and projected back to D
        # via fuse_proj, then added as a single residual. The serial path
        # (goal-CA → visual-CA) is used when False.
        use_parallel_cross_attentions: bool = False,
        # When True (and use_goal_cross_attention=True), the goal-CA output is multiplied
        # by a learnable scalar α_goal (init=0, no temb dependence) before being added to
        # the residual stream (serial path) or concatenated into fuse_proj (parallel path).
        # Vision residual and FF are never gated. At init the block effectively becomes
        # "visual CA + FF only", forcing vision to carry the action signal from step 0.
        use_gated_goal_residual: bool = False,
        # When True (and use_goal_cross_attention=True), the WCA output is NOT added to the
        # residual stream. Instead it feeds GoalAdaLN as per-token (γ_goal, β_goal)
        # conditioning for the visual-CA's norm. Visual CA is the only path that writes
        # to the residual → hard architectural ceiling on the goal stream's influence.
        # Mutually exclusive with use_parallel_cross_attentions / use_gated_goal_residual.
        use_goal_adaln_modulation: bool = False,
        # When True, the block's `norm1` becomes a StateAdaLN that adds a per-channel
        # state-derived modulation on top of the standard temb modulation. The state
        # branch is zero-init so at step 0 behavior matches standard AdaLN; the
        # optimizer grows state modulation only if it reduces loss. Gives state a
        # global modulation pathway (every block, every norm) on top of being a token
        # in hidden_states. Mutually exclusive with use_goal_adaln_modulation (both
        # would want to replace norm1).
        use_state_adaln: bool = False,
        # When True (requires use_state_adaln=True), switch StateAdaLN's parameterization
        # to ReZero-style: random-init linear_state + per-block learnable scalar gate
        # α_state init=0. Gate controls magnitude, linear controls direction. Each block
        # gets its own gate (so all 12 blocks have independent gates — blocks that don't
        # want state modulation can train α → 0, giving implicit per-block sparsity).
        use_gate_on_state_adaln: bool = False,
        # When True, register PyTorch backward hooks on the goal-CA output and visual-CA
        # output, recording ‖∂L/∂out‖ per block per backward pass into
        # `self._last_goal_grad_norm` and `self._last_vis_grad_norm`. Read by the
        # workspace training loop to log mean per-block grad norms each epoch. Hooks
        # are no-ops in eval/no_grad mode (`tensor.requires_grad` is False).
        log_attention_grad_norms: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type
        self.use_goal_cross_attention = use_goal_cross_attention
        self.use_weighted_cross_attention = use_weighted_cross_attention
        self.use_parallel_cross_attentions = use_parallel_cross_attentions
        self.use_gated_goal_residual = use_gated_goal_residual
        self.use_goal_adaln_modulation = use_goal_adaln_modulation
        self.use_state_adaln = use_state_adaln
        self.use_gate_on_state_adaln = use_gate_on_state_adaln
        self.log_attention_grad_norms = log_attention_grad_norms

        # Local sanity: Idea 1 is mutually exclusive with the residual-write modes.
        if use_goal_adaln_modulation:
            assert use_goal_cross_attention, (
                "use_goal_adaln_modulation=True requires use_goal_cross_attention=True"
            )
            assert not use_parallel_cross_attentions, (
                "use_goal_adaln_modulation is incompatible with use_parallel_cross_attentions"
            )
            assert not use_gated_goal_residual, (
                "use_goal_adaln_modulation is incompatible with use_gated_goal_residual"
            )
        # State-AdaLN and goal-AdaLN both want to replace `norm1`. Forbid combining
        # (a unified MultiCondAdaLN could be added later if both are desired).
        if use_state_adaln:
            assert not use_goal_adaln_modulation, (
                "use_state_adaln is incompatible with use_goal_adaln_modulation "
                "(both want to replace norm1)"
            )
        # ReZero gate only makes sense as a flavor of StateAdaLN.
        if use_gate_on_state_adaln:
            assert use_state_adaln, (
                "use_gate_on_state_adaln=True requires use_state_adaln=True "
                "(the gate is part of StateAdaLN's ReZero parameterization)"
            )
        # Backward-hook buffers. Set by hooks during loss.backward(); read by the
        # workspace training loop after backward() and before optimizer.step().
        self._last_goal_grad_norm = None
        self._last_vis_grad_norm  = None

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn (or visual-CA depending on block role)
        if use_goal_adaln_modulation and norm_type == "ada_norm":
            # GoalAdaLN reduces to standard temb-AdaLN at init (goal branch zero-init'd),
            # then learns per-token (γ_goal, β_goal) modulation on top.
            self.norm1 = GoalAdaLN(dim, norm_elementwise_affine=norm_elementwise_affine, norm_eps=norm_eps)
        elif use_state_adaln and norm_type == "ada_norm":
            # StateAdaLN reduces to standard temb-AdaLN at init (state branch zero-contribution),
            # then learns per-channel (γ_state, β_state) modulation on top.
            # use_gate=True switches to the ReZero parameterization (random-init linear +
            # per-block scalar gate init=0). use_gate=False keeps zero-init linear.
            self.norm1 = StateAdaLN(
                dim,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                use_gate=use_gate_on_state_adaln,
            )
        elif norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Goal cross-attention (optional) — fires before visual cross-attention.
        # Uses AdaLayerNorm when norm_type=="ada_norm" so it gets the same
        # timestep conditioning as the other norms in this block.
        if use_goal_cross_attention:
            if norm_type == "ada_norm":
                self.norm_goal = AdaLayerNorm(dim)
            else:
                self.norm_goal = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            if use_weighted_cross_attention:
                # WeightedCrossAttention: log-biases attention logits by GMM weights.
                self.attn_goal = WeightedCrossAttention(
                    query_dim=dim,
                    key_dim=goal_cross_attention_dim or dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                )
            else:
                self.attn_goal = Attention(
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    cross_attention_dim=goal_cross_attention_dim or dim,
                    upcast_attention=upcast_attention,
                    out_bias=attention_out_bias,
                )

            # Fuse projection for parallel cross-attentions: concat(goal_out, visual_out)
            # → Linear(2D, D). Only allocated when both flags are on.
            if use_parallel_cross_attentions:
                self.fuse_proj = nn.Linear(2 * dim, dim, bias=attention_out_bias)

            # ReZero-style learnable scalar gate on the goal residual. Init=0 so the
            # goal stream contributes nothing at step 0 — vision and FF carry the
            # model. Optimizer grows α_goal only if WCA actually reduces loss.
            if use_gated_goal_residual:
                self.goal_residual_gate = nn.Parameter(torch.zeros(1))

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    # Backward-hook callbacks. Bound to `self`, so PyTorch's autograd registers
    # them once per forward and the closure stays tiny.
    def _record_goal_grad_norm(self, grad: torch.Tensor) -> None:
        self._last_goal_grad_norm = grad.detach().norm().item()

    def _record_vis_grad_norm(self, grad: torch.Tensor) -> None:
        self._last_vis_grad_norm = grad.detach().norm().item()

    def _maybe_register_goal_grad_hook(self, t: torch.Tensor) -> None:
        if self.log_attention_grad_norms and t.requires_grad:
            t.register_hook(self._record_goal_grad_norm)

    def _maybe_register_vis_grad_hook(self, t: torch.Tensor) -> None:
        if self.log_attention_grad_norms and t.requires_grad:
            t.register_hook(self._record_vis_grad_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        goal_hidden_states: Optional[torch.Tensor] = None,
        goal_weights: Optional[torch.Tensor] = None,  # (B, N_goals) — required when use_weighted_cross_attention
        state_ctx: Optional[torch.Tensor] = None,     # (B, D) — pooled state summary for StateAdaLN
    ) -> torch.Tensor:
        # Reset hook buffers each forward so the workspace doesn't read stale
        # values when this block was skipped (e.g. self-attn block has no goal).
        if self.log_attention_grad_norms:
            self._last_goal_grad_norm = None
            self._last_vis_grad_norm  = None
        # Idea 1: goal-AdaLN modulation. WCA still runs to produce goal_ctx, but its
        # output is NOT added to the residual stream. Instead it feeds GoalAdaLN as
        # per-token (γ_goal, β_goal) conditioning for the visual-CA's norm. Only
        # visual CA writes to the residual → hard architectural ceiling on the goal
        # stream's influence. self.norm1 is allocated as GoalAdaLN when this flag is on.
        adaln_mod = (
            self.use_goal_adaln_modulation
            and self.use_goal_cross_attention
            and goal_hidden_states is not None
            and encoder_hidden_states is not None
        )

        # Parallel cross-attentions: both goal-CA and visual-CA read the *pre-block*
        # hidden_states through their own AdaLN. Their outputs are concatenated along
        # the feature dim and projected back to D via fuse_proj, then added as a
        # single residual. Falls back to the serial path if either context stream
        # is missing at runtime (e.g. odd self-attn blocks).
        parallel = (
            self.use_parallel_cross_attentions
            and self.use_goal_cross_attention
            and goal_hidden_states is not None
            and encoder_hidden_states is not None
        )

        if adaln_mod:
            # ---- Goal branch: WCA still runs, output captured as goal_ctx ----
            if self.norm_type == "ada_norm":
                norm_goal_in = self.norm_goal(hidden_states, temb)
            else:
                norm_goal_in = self.norm_goal(hidden_states)

            if self.use_weighted_cross_attention:
                assert goal_weights is not None, (
                    "goal_weights must be provided when use_weighted_cross_attention=True"
                )
                goal_ctx = self.attn_goal(
                    norm_goal_in,
                    encoder_hidden_states=goal_hidden_states,
                    weights=goal_weights,
                )
            else:
                goal_ctx = self.attn_goal(
                    norm_goal_in,
                    encoder_hidden_states=goal_hidden_states,
                )
            self._maybe_register_goal_grad_hook(goal_ctx)

            # ---- Visual branch: GoalAdaLN takes (x, temb, goal_ctx) ----
            # self.norm1 is GoalAdaLN here. goal_ctx becomes per-token (γ_goal, β_goal)
            # added to the temb-derived (γ_t, β_t) before applying to LN(x).
            norm_vis_in = self.norm1(hidden_states, temb, goal_ctx)
            if self.pos_embed is not None:
                norm_vis_in = self.pos_embed(norm_vis_in)

            vis_out = self.attn1(
                norm_vis_in,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
            self._maybe_register_vis_grad_hook(vis_out)
            if self.final_dropout:
                vis_out = self.final_dropout(vis_out)

            # Only visual CA writes to the residual. WCA output already consumed by GoalAdaLN.
            hidden_states = vis_out + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
        elif parallel:
            # ---- Goal branch (WCA or standard CA) ----
            if self.norm_type == "ada_norm":
                norm_goal_in = self.norm_goal(hidden_states, temb)
            else:
                norm_goal_in = self.norm_goal(hidden_states)

            if self.use_weighted_cross_attention:
                assert goal_weights is not None, (
                    "goal_weights must be provided when use_weighted_cross_attention=True"
                )
                goal_out = self.attn_goal(
                    norm_goal_in,
                    encoder_hidden_states=goal_hidden_states,
                    weights=goal_weights,
                )
            else:
                goal_out = self.attn_goal(
                    norm_goal_in,
                    encoder_hidden_states=goal_hidden_states,
                )
            self._maybe_register_goal_grad_hook(goal_out)

            # ---- Visual branch (standard CA, reads same pre-block x) ----
            if self.norm_type == "ada_norm":
                # StateAdaLN accepts an extra state_ctx kwarg; standard AdaLayerNorm/GoalAdaLN do not.
                if self.use_state_adaln:
                    norm_vis_in = self.norm1(hidden_states, temb, state_ctx)
                else:
                    norm_vis_in = self.norm1(hidden_states, temb)
            else:
                norm_vis_in = self.norm1(hidden_states)
            if self.pos_embed is not None:
                norm_vis_in = self.pos_embed(norm_vis_in)

            vis_out = self.attn1(
                norm_vis_in,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
            self._maybe_register_vis_grad_hook(vis_out)

            # ---- ReZero-style asymmetric gate on the goal half (vision stays ungated) ----
            if self.use_gated_goal_residual:
                goal_out = self.goal_residual_gate * goal_out

            # ---- Fuse: concat along feature dim, project back to D ----
            fused = self.fuse_proj(torch.cat([goal_out, vis_out], dim=-1))
            if self.final_dropout:
                fused = self.final_dropout(fused)
            hidden_states = hidden_states + fused
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
        else:
            # 1. Goal cross-attention (only fires when flag is on AND goal tokens provided)
            if self.use_goal_cross_attention and goal_hidden_states is not None:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm_goal(hidden_states, temb)
                else:
                    norm_hidden_states = self.norm_goal(hidden_states)

                if self.use_weighted_cross_attention:
                    assert goal_weights is not None, (
                        "goal_weights must be provided when use_weighted_cross_attention=True"
                    )
                    attn_output = self.attn_goal(
                        norm_hidden_states,
                        encoder_hidden_states=goal_hidden_states,
                        weights=goal_weights,
                    )
                else:
                    attn_output = self.attn_goal(
                        norm_hidden_states,
                        encoder_hidden_states=goal_hidden_states,
                    )
                self._maybe_register_goal_grad_hook(attn_output)

                if self.final_dropout:
                    attn_output = self.final_dropout(attn_output)
                # ReZero-style asymmetric gate on the goal residual only.
                if self.use_gated_goal_residual:
                    attn_output = self.goal_residual_gate * attn_output
                hidden_states = attn_output + hidden_states
                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)

            # 2. Visual cross-attention (or self-attention when encoder_hidden_states is None)
            if self.norm_type == "ada_norm":
                # StateAdaLN accepts an extra state_ctx kwarg; AdaLayerNorm/GoalAdaLN do not.
                if self.use_state_adaln:
                    norm_hidden_states = self.norm1(hidden_states, temb, state_ctx)
                else:
                    norm_hidden_states = self.norm1(hidden_states, temb)
            else:
                norm_hidden_states = self.norm1(hidden_states)

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
            # Only record vis grad when attn1 is in visual-CA mode. Self-attn
            # blocks (encoder_hidden_states is None) are not "visual" — skip.
            if encoder_hidden_states is not None:
                self._maybe_register_vis_grad_hook(attn_output)
            if self.final_dropout:
                attn_output = self.final_dropout(attn_output)

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class GoalAuxiliaryStreamBlock(nn.Module):
    """
    One block of the dual-stream goal-auxiliary architecture.

    Maintains TWO hidden_states streams (both shaped like (state + action) tokens):
      stream 1 — the "main" / action stream. Its final state gets decoded.
      stream 2 — the "auxiliary" / goal stream. Discarded at the very end.

    Per block:
        # 1. Each stream pulls from its own context (in parallel — independent)
        x1 = x1 + CA (AdaLN(x1, temb), visual_tokens)
        x2 = x2 + WCA(AdaLN(x2, temb), goal_tokens, goal_weights)

        # 2. Joint self-attention with separate per-stream Q/K/V/out
        out1, out2 = GoalAuxiliaryAttention(AdaLN_sa1(x1), AdaLN_sa2(x2))
        x1 = x1 + out1
        x2 = x2 + out2

        # 3. Per-stream feed-forward
        x1 = x1 + FFN_1(LN(x1))
        x2 = x2 + FFN_2(LN(x2))

    Goal info reaches stream 1 only via the V-mixing in joint attention's softmax —
    no direct goal residual into stream 1. (Diagnostic-wise this is *more* indirect
    than the existing serial WCA, but it does NOT structurally cap goal bandwidth
    the way Idea 1 does; see discussion in chat.)
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,         # for visual CA's K/V dim
        goal_cross_attention_dim: Optional[int] = None,    # for WCA's K/V dim
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        norm_type: str = "ada_norm",
        final_dropout: bool = False,
        attention_out_bias: bool = True,
        use_weighted_cross_attention: bool = True,
        log_attention_grad_norms: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.norm_type = norm_type
        self.use_weighted_cross_attention = use_weighted_cross_attention
        self.log_attention_grad_norms = log_attention_grad_norms
        # Hook buffers — same convention as BasicTransformerBlock so the workspace
        # logger picks them up without changes.
        self._last_goal_grad_norm = None
        self._last_vis_grad_norm = None

        # ----- Stream-1 visual cross-attention -----
        if norm_type == "ada_norm":
            self.norm_ca1 = AdaLayerNorm(dim)
        else:
            self.norm_ca1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn_ca1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim or dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # ----- Stream-2 goal cross-attention (WCA or standard CA) -----
        if norm_type == "ada_norm":
            self.norm_wca2 = AdaLayerNorm(dim)
        else:
            self.norm_wca2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        if use_weighted_cross_attention:
            self.attn_wca2 = WeightedCrossAttention(
                query_dim=dim,
                key_dim=goal_cross_attention_dim or dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
            )
        else:
            self.attn_wca2 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=goal_cross_attention_dim or dim,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )

        # ----- Joint self-attention (per-stream Q/K/V/out) -----
        if norm_type == "ada_norm":
            self.norm_sa1 = AdaLayerNorm(dim)
            self.norm_sa2 = AdaLayerNorm(dim)
        else:
            self.norm_sa1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.norm_sa2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.joint_attn = GoalAuxiliaryAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # ----- Per-stream FFN -----
        self.norm_ff1 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.norm_ff2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff1 = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self.ff2 = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    # Backward-hook callbacks (same convention as BasicTransformerBlock).
    def _record_goal_grad_norm(self, grad: torch.Tensor) -> None:
        self._last_goal_grad_norm = grad.detach().norm().item()

    def _record_vis_grad_norm(self, grad: torch.Tensor) -> None:
        self._last_vis_grad_norm = grad.detach().norm().item()

    def _maybe_register_goal_grad_hook(self, t: torch.Tensor) -> None:
        if self.log_attention_grad_norms and t.requires_grad:
            t.register_hook(self._record_goal_grad_norm)

    def _maybe_register_vis_grad_hook(self, t: torch.Tensor) -> None:
        if self.log_attention_grad_norms and t.requires_grad:
            t.register_hook(self._record_vis_grad_norm)

    def forward(
        self,
        x1: torch.Tensor,                                # (B, T, D)  main stream
        x2: torch.Tensor,                                # (B, T, D)  auxiliary stream
        temb: torch.Tensor,                              # (B, D)
        visual_tokens: torch.Tensor,                     # (B, T_v, D)
        goal_tokens: Optional[torch.Tensor] = None,      # (B, T_g, D)
        goal_weights: Optional[torch.Tensor] = None,     # (B, T_g)
    ):
        # Reset hook buffers per forward (so we don't read stale values).
        if self.log_attention_grad_norms:
            self._last_goal_grad_norm = None
            self._last_vis_grad_norm = None

        # ---- 1. Stream-1: visual CA ----
        x1_norm = self.norm_ca1(x1, temb) if self.norm_type == "ada_norm" else self.norm_ca1(x1)
        ca_out = self.attn_ca1(x1_norm, encoder_hidden_states=visual_tokens)
        self._maybe_register_vis_grad_hook(ca_out)
        if self.final_dropout is not None:
            ca_out = self.final_dropout(ca_out)
        x1 = x1 + ca_out

        # ---- 2. Stream-2: goal WCA (skip cleanly if goals missing) ----
        if goal_tokens is not None:
            x2_norm = self.norm_wca2(x2, temb) if self.norm_type == "ada_norm" else self.norm_wca2(x2)
            if self.use_weighted_cross_attention:
                assert goal_weights is not None, (
                    "goal_weights must be provided when use_weighted_cross_attention=True"
                )
                wca_out = self.attn_wca2(x2_norm, encoder_hidden_states=goal_tokens, weights=goal_weights)
            else:
                wca_out = self.attn_wca2(x2_norm, encoder_hidden_states=goal_tokens)
            self._maybe_register_goal_grad_hook(wca_out)
            if self.final_dropout is not None:
                wca_out = self.final_dropout(wca_out)
            x2 = x2 + wca_out

        # ---- 3. Joint self-attention (per-stream Q/K/V/out, ONE softmax) ----
        x1_norm = self.norm_sa1(x1, temb) if self.norm_type == "ada_norm" else self.norm_sa1(x1)
        x2_norm = self.norm_sa2(x2, temb) if self.norm_type == "ada_norm" else self.norm_sa2(x2)
        sa_out1, sa_out2 = self.joint_attn(x1_norm, x2_norm)
        if self.final_dropout is not None:
            sa_out1 = self.final_dropout(sa_out1)
            sa_out2 = self.final_dropout(sa_out2)
        x1 = x1 + sa_out1
        x2 = x2 + sa_out2

        # ---- 4. Per-stream FFN ----
        x1 = x1 + self.ff1(self.norm_ff1(x1))
        x2 = x2 + self.ff2(self.norm_ff2(x2))
        return x1, x2


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        # Goal cross-attention: adds a cross-attn to goal tokens before every
        # visual cross-attention block (even-indexed blocks when interleave_self_attention=True).
        use_goal_cross_attention: bool = False,
        goal_cross_attention_dim: Optional[int] = None,
        # When True, the goal cross-attention uses WeightedCrossAttention (log-bias from GMM
        # weights) instead of standard scaled-dot-product attention.
        # Requires use_goal_cross_attention=True and goal_weights passed to forward().
        use_weighted_cross_attention: bool = False,
        # When True (and use_goal_cross_attention=True), each cross-attn block runs
        # goal-CA and visual-CA in parallel rather than serially. See BasicTransformerBlock.
        use_parallel_cross_attentions: bool = False,
        # When True (and use_goal_cross_attention=True), each cross-attn block multiplies
        # its WCA/goal-CA output by a learnable scalar α_goal init=0 before adding to the
        # residual (or concat-fusing). Asymmetric ReZero — vision residual stays ungated.
        use_gated_goal_residual: bool = False,
        # Idea 1. When True (and use_goal_cross_attention=True), the WCA output of each
        # cross-attn block is NOT added to the residual. Instead it feeds GoalAdaLN as
        # per-token (γ_goal, β_goal) conditioning for the visual-CA's norm. Only visual
        # CA writes to the residual. Mutually exclusive with use_parallel_cross_attentions
        # and use_gated_goal_residual (those modes write a goal residual; this one doesn't).
        use_goal_adaln_modulation: bool = False,
        # Dual-stream architecture. When True, replaces the BasicTransformerBlock chain
        # with GoalAuxiliaryStreamBlocks. Maintains two parallel hidden_states streams:
        #   stream 1 ("main"/action)      — cross-attends to visual_tokens
        #   stream 2 ("auxiliary"/goal)   — cross-attends to goal_tokens via WCA
        # Per block: stream-1 CA, stream-2 WCA, joint SA with per-stream Q/K/V/out,
        # per-stream FFN. At the end, only stream-1 is decoded; stream-2 is discarded.
        # Mutually exclusive with use_parallel_cross_attentions / use_gated_goal_residual
        # / use_goal_adaln_modulation. Requires use_goal_cross_attention=True. The
        # `interleave_self_attention` flag is ignored under this mode (the joint SA
        # already provides self-attn inside every block).
        use_goal_auxiliary_stream: bool = False,
        # When True, every BasicTransformerBlock's `norm1` becomes a StateAdaLN that
        # adds per-channel state-derived modulation on top of temb. State context
        # (a pooled (B, D) summary) is passed in DiT.forward via the `state_ctx` kwarg.
        # Mutually exclusive with use_goal_adaln_modulation. Only wired into
        # BasicTransformerBlock currently; not yet supported under
        # use_goal_auxiliary_stream (dual-stream block).
        use_state_adaln: bool = False,
        # When True (requires use_state_adaln=True), switch StateAdaLN to the ReZero
        # parameterization: random-init linear_state + per-block scalar gate α_state
        # init=0. With 12 blocks, you get 12 independent gates that the optimizer can
        # train to different values — implicit per-block sparsity for state modulation.
        use_gate_on_state_adaln: bool = False,
        # When True, every block registers backward hooks on its goal-CA / visual-CA
        # outputs and records ‖∂L/∂out‖ each backward pass. Read by the workspace
        # training loop to log per-block grad norm means per epoch.
        log_attention_grad_norms: bool = False,
    ):
        super().__init__()

        # Mutual exclusion among "what to do with the goal stream" modes.
        if use_goal_adaln_modulation:
            assert not use_parallel_cross_attentions, (
                "use_goal_adaln_modulation is mutually exclusive with "
                "use_parallel_cross_attentions (Idea 1 has no goal residual to fuse)."
            )
            assert not use_gated_goal_residual, (
                "use_goal_adaln_modulation is mutually exclusive with "
                "use_gated_goal_residual (Idea 1 has no goal residual to gate)."
            )
        if use_goal_auxiliary_stream:
            assert use_goal_cross_attention, (
                "use_goal_auxiliary_stream=True requires use_goal_cross_attention=True"
            )
            assert not use_parallel_cross_attentions, (
                "use_goal_auxiliary_stream is mutually exclusive with use_parallel_cross_attentions"
            )
            assert not use_gated_goal_residual, (
                "use_goal_auxiliary_stream is mutually exclusive with use_gated_goal_residual"
            )
            assert not use_goal_adaln_modulation, (
                "use_goal_auxiliary_stream is mutually exclusive with use_goal_adaln_modulation"
            )
            assert not use_state_adaln, (
                "use_state_adaln is not yet supported with use_goal_auxiliary_stream "
                "(StateAdaLN currently wired only into BasicTransformerBlock)"
            )
        if use_state_adaln:
            assert not use_goal_adaln_modulation, (
                "use_state_adaln is mutually exclusive with use_goal_adaln_modulation"
            )
        if use_gate_on_state_adaln:
            assert use_state_adaln, (
                "use_gate_on_state_adaln=True requires use_state_adaln=True"
            )

        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )

        if use_goal_auxiliary_stream:
            # Dual-stream mode: every block is a GoalAuxiliaryStreamBlock.
            # interleave_self_attention is intentionally ignored — joint SA inside
            # each block already does the self-attn role.
            self.transformer_blocks = nn.ModuleList(
                [
                    GoalAuxiliaryStreamBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=None,                    # visual K/V dim = D
                        goal_cross_attention_dim=goal_cross_attention_dim,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        upcast_attention=upcast_attention,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        norm_type=norm_type,
                        final_dropout=final_dropout,
                        use_weighted_cross_attention=use_weighted_cross_attention,
                        log_attention_grad_norms=log_attention_grad_norms,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        self.inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        positional_embeddings=positional_embeddings,
                        num_positional_embeddings=max_num_positional_embeddings,
                        final_dropout=final_dropout,
                        # Goal cross-attn only on cross-attention blocks (even idx when interleaving).
                        # With interleave_self_attention=False every block is a cross-attn block.
                        use_goal_cross_attention=(
                            use_goal_cross_attention
                            and (not interleave_self_attention or idx % 2 == 0)
                        ),
                        goal_cross_attention_dim=goal_cross_attention_dim,
                        use_weighted_cross_attention=(
                            use_weighted_cross_attention
                            and use_goal_cross_attention
                            and (not interleave_self_attention or idx % 2 == 0)
                        ),
                        use_parallel_cross_attentions=(
                            use_parallel_cross_attentions
                            and use_goal_cross_attention
                            and (not interleave_self_attention or idx % 2 == 0)
                        ),
                        use_gated_goal_residual=(
                            use_gated_goal_residual
                            and use_goal_cross_attention
                            and (not interleave_self_attention or idx % 2 == 0)
                        ),
                        use_goal_adaln_modulation=(
                            use_goal_adaln_modulation
                            and use_goal_cross_attention
                            and (not interleave_self_attention or idx % 2 == 0)
                        ),
                        # StateAdaLN applies to every block's norm1 (no idx gating);
                        # the modulation is timestep-conditioning-like.
                        use_state_adaln=use_state_adaln,
                        # ReZero gate flavor of StateAdaLN. Each block allocates its
                        # own scalar gate (12 total in the default 12-layer DiT).
                        use_gate_on_state_adaln=use_gate_on_state_adaln,
                        log_attention_grad_norms=log_attention_grad_norms,
                    )
                    for idx in range(num_layers)
                ]
            )

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,           # (B, T, D)
        encoder_hidden_states: torch.Tensor,   # (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        goal_hidden_states: Optional[torch.Tensor] = None,  # (B, G, D)
        goal_weights: Optional[torch.Tensor] = None,        # (B, G) — GMM probs for weighted cross-attn
        state_ctx: Optional[torch.Tensor] = None,           # (B, D) — pooled state summary for StateAdaLN
        return_all_hidden_states: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        if self.config.use_goal_auxiliary_stream:
            # Dual-stream loop: maintain two parallel hidden_states streams.
            # Both start from the same input. Only stream 1 is decoded at the end.
            # all_hidden_states tracks stream-1 only (for downstream consumers).
            # NOTE: state_ctx is currently NOT forwarded into GoalAuxiliaryStreamBlock
            # (the assertion in __init__ forbids that combination).
            x1 = hidden_states
            x2 = hidden_states.clone()
            for block in self.transformer_blocks:
                x1, x2 = block(
                    x1, x2,
                    temb=temb,
                    visual_tokens=encoder_hidden_states,
                    goal_tokens=goal_hidden_states,
                    goal_weights=goal_weights,
                )
                all_hidden_states.append(x1)
            hidden_states = x1   # discard stream 2
        else:
            # Process through transformer blocks
            for idx, block in enumerate(self.transformer_blocks):
                is_self_attn_block = idx % 2 == 1 and self.config.interleave_self_attention
                if is_self_attn_block:
                    # Self-attention block: no visual or goal cross-attention
                    hidden_states = block(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        temb=temb,
                        goal_hidden_states=None,
                        goal_weights=None,
                        state_ctx=state_ctx,
                    )
                else:
                    # Visual cross-attention block: optionally also cross-attend to goals
                    hidden_states = block(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=None,
                        temb=temb,
                        goal_hidden_states=goal_hidden_states,
                        goal_weights=goal_weights,
                        state_ctx=state_ctx,
                    )
                all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)

        return hidden_states


if __name__ == "__main__":
    import torch

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create test inputs
    batch_size = 2
    seq_len = 17
    encoder_seq_len = 48
    hidden_dim = 1024

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, hidden_dim)
    timestep = torch.arange(batch_size)  # Shape: (2,)

    # Initialize model
    # Create padding mask for encoder_hidden_states
    # Example: First 3 tokens are padded (0), rest are valid (1)
    encoder_attention_mask = torch.zeros((batch_size, encoder_seq_len), dtype=torch.long)
    encoder_attention_mask[:, 3:] = 1  # Set last 45 positions to 1
    model = DiT(
        num_attention_heads=16,
        attention_head_dim=64,
        num_layers=2,  # Using 2 layers for faster testing
        dropout=0.1,
    )

    # Run forward pass
    output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        timestep=timestep,
    )

    # Print shapes for verification
    print(f"Input hidden_states shape: {hidden_states.shape}")
    print(f"Input encoder_hidden_states shape: {encoder_hidden_states.shape}")
    print(f"Input timestep shape: {timestep.shape}")
    print(f"Output shape: {output.shape}")

    # Expected output shape should be (2, 17, 1024)
    assert output.shape == (
        batch_size,
        seq_len,
        hidden_dim,
    ), f"Expected shape {(batch_size, seq_len, hidden_dim)}, got {output.shape}"

    print("All tests passed!")
