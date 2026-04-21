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
        # 1. Self-Attn
        if norm_type == "ada_norm":
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        goal_hidden_states: Optional[torch.Tensor] = None,
        goal_weights: Optional[torch.Tensor] = None,  # (B, N_goals) — required when use_weighted_cross_attention
    ) -> torch.Tensor:
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

            if self.final_dropout:
                attn_output = self.final_dropout(attn_output)
            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # 2. Visual cross-attention (or self-attention when encoder_hidden_states is None)
        if self.norm_type == "ada_norm":
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
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )

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
        return_all_hidden_states: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

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
