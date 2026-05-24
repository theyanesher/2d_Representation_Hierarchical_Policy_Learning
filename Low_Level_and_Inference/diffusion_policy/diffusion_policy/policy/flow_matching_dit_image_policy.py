"""
FlowMatchingDiTImagePolicy
==========================
A diffusion_policy-compatible policy using the GR00T cross-attention DiT
with flow matching, but without the GR00T VL backbone.

Architecture
------------
Visual encoding  (replaces GR00T LLM backbone)
  visual_encoder : VisualTokenEncoder — self-contained encoder (owns crop,
                   projector, and positional embeddings).
  Each encoder produces: (B, n_obs_steps·n_cams·N_tokens, embed_dim)
  Used as DiT encoder_hidden_states (cross-attention context)

State encoding  (same design as SimpleFlowmatchingActionHead)
  state_encoder : SimpleMLP(state_dim → input_embedding_dim)
  → (B, n_obs_steps, D)  — DiT hidden_state prefix tokens

Action encoding  (flow matching)
  action_encoder : ActionEncoder(action_dim → input_embedding_dim)
  → (B, action_horizon, D)  — DiT hidden_state tokens

DiT backbone
  hidden_states        = cat([state_tokens, action_tokens], dim=1)
  encoder_hidden_states = visual_tokens
  → (B, n_obs_steps + action_horizon, hidden_size)

Action decoder
  action_decoder : SimpleMLP(hidden_size → action_dim)
  slice last action_horizon tokens → predicted velocity

Diffusion policy interface
  set_normalizer(normalizer)
  compute_loss(batch)       → scalar loss
  predict_action(obs_dict)  → {'action': ..., 'action_pred': ...}

Swapping the visual encoder
  Change visual_encoder_type and visual_encoder_cfg in the YAML:
    visual_encoder_type: resnet  →  visual_encoder_type: dinov2
    visual_encoder_cfg:
      backbone: resnet18          model_name: facebook/dinov2-base
      ...                         frozen: true
"""

from typing import Dict, Optional
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.obs_util import process_observations, process_obs_shape_meta

from diffusion_policy.model.flow_matching.cross_attention_dit import DiT
from diffusion_policy.model.flow_matching.helpers import ActionEncoder, SimpleMLP, PerComponentStateEncoder
from diffusion_policy.model.flow_matching.visual_encoders import build_visual_encoder
from diffusion_policy.model.goal_gripper_encoder.base_goal_gripper_encoder import BaseGoalGripperEncoder
from diffusion_policy.model.goal_gripper_encoder.keypoint_aware_goal_gripper_encoder import KeypointAwareGoalGripperEncoder


class FlowMatchingDiTImagePolicy(BaseImagePolicy):
    """
    Flow-matching DiT policy compatible with TrainDiffusionUnetHybridWorkspace.
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # ---- visual encoder ----
        visual_encoder_type: str = "resnet",
        visual_encoder_cfg: dict = None,
        use_separate_wrist_encoder: bool = False,  # separate backbone for cam2_image
        observation_mode: str = "image",
        crop_shape: tuple = (224, 224),   # (H, W) to random-crop each image
        # ---- flow matching ----
        num_inference_timesteps: int = 10,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        # ---- architecture ----
        input_embedding_dim: int = 512,   # must == DiT num_heads * head_dim
        hidden_size: int = 512,            # must == DiT output_dim
        add_pos_embed: bool = True,
        max_seq_len: int = 64,
        diffusion_model_cfg: dict = None,
        # ---- goal gripper encoder ----
        goal_gripper_encoder_type: str = "base",  # "base" or "keypoint_aware"
        # When True, goal tokens are fed to the DiT via dedicated cross-attention
        # (one cross-attn to goals before every visual cross-attn block) rather
        # than being prepended to hidden_states for self-attention.
        use_goal_cross_attention: bool = False,
        # When True (requires use_goal_cross_attention=True and has_gmm=True), the goal
        # cross-attention uses the GMM probability weights as a log-prior bias on attention
        # logits: logits += log(w_j). The encoder does NOT append weight as an input feature.
        use_weighted_cross_attention: bool = False,
        # When True (requires use_goal_cross_attention=True), each cross-attn block runs
        # the goal-CA and visual-CA in parallel from the *pre-block* hidden_states,
        # concatenates their outputs and projects back to D via fuse_proj, then adds as
        # a single residual. Composable with use_weighted_cross_attention.
        use_parallel_cross_attentions: bool = False,
        # When True (requires use_goal_cross_attention=True), each cross-attn block
        # multiplies its goal-CA output by a learnable scalar α_goal (init=0) before
        # adding to the residual / concat-fusing. ReZero-style asymmetric gate that
        # hands vision the head start in optimization. Composable with the flags above.
        use_gated_goal_residual: bool = False,
        # Idea 1. When True (requires use_goal_cross_attention=True), each cross-attn
        # block routes its WCA output through GoalAdaLN as per-token (γ_goal, β_goal)
        # conditioning for the visual-CA's norm instead of adding it to the residual.
        # Hard architectural constraint: goals can no longer write to the residual
        # stream — they can only modulate vision. Mutually exclusive with
        # use_parallel_cross_attentions and use_gated_goal_residual.
        use_goal_adaln_modulation: bool = False,
        # Dual-stream architecture (GoalAuxiliaryStream). When True, replaces the
        # standard DiT block chain with GoalAuxiliaryStreamBlocks: two parallel
        # streams (main = action, auxiliary = goal) with per-block CA / WCA,
        # joint self-attention (per-stream Q/K/V/out), and per-stream FFN. Only the
        # main stream is decoded at the end. Mutually exclusive with the other
        # goal-routing flags (parallel_cross_attentions / gated_goal_residual /
        # goal_adaln_modulation). Requires use_goal_cross_attention=True.
        use_goal_auxiliary_stream: bool = False,
        # When True, encode the 10-D state as 3 sub-tokens per obs step (pos, rot,
        # gripper) instead of a single token. Adds learned per-component identity
        # embeddings + temporal embeddings. Output is (B, To*3, D) instead of
        # (B, To, D). Assumes the standard MimicGen Panda EE-space layout
        # [pos(3), rot_6D(6), gripper(1)] = 10.
        use_per_component_state: bool = False,
        # When True, the DiT's BasicTransformerBlock norm1 becomes a StateAdaLN that
        # adds per-channel state-derived modulation on top of temb. Provides state
        # with a global modulation pathway across every block. The state branch is
        # zero-init so behavior matches standard AdaLN at step 0. Mutually exclusive
        # with use_goal_adaln_modulation. Currently not supported alongside
        # use_goal_auxiliary_stream.
        use_state_adaln: bool = False,
        # Diagnostic: register PyTorch backward hooks on the goal-CA and visual-CA
        # outputs of every cross-attn block to record ‖∂L/∂out‖. The workspace logs
        # per-block means each epoch. Small per-step overhead; safe to leave off
        # for production runs.
        log_attention_grad_norms: bool = False,
        # When set, keep only the top-K GMM candidates by weight before encoding.
        # Useful with use_goal_cross_attention=False (self-attention) to bound sequence length.
        # None = use all candidates.
        gmm_top_k: int = None,
        **kwargs,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # 1. Parse shape_meta                                                 #
        # ------------------------------------------------------------------ #
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, "Expected flat action vector."
        action_dim = action_shape[0]

        obs_shape_meta = copy.deepcopy(dict(shape_meta["obs"]))
        process_obs_shape_meta(observation_mode, shape_meta, obs_shape_meta)

        # Separate visual obs (cameras) from state obs (low-dim proprioception).
        # extrinsic/intrinsic keys are neither visual nor state — they are
        # consumed directly by ResNetPRoPETokenEncoder from nobs.
        visual_obs_key_shapes: Dict[str, list] = {}
        state_obs_key_shapes: Dict[str, list] = {}
        goal_gripper_key_shapes: Dict[str, list] = {}
        gmm_goals_key = None
        gmm_weights_key = None

        for key, attr in obs_shape_meta.items():
            shape = list(attr["shape"])
            stype = attr.get("type", "low_dim")
            if stype in ("rgb", "depth", "pointmap", "plucker", "heatmap", "ghost_heatmap"):
                visual_obs_key_shapes[key] = shape
            elif stype == "goal_gripper":
                goal_gripper_key_shapes[key] = shape
            elif stype == "gmm_goals":
                gmm_goals_key = key
            elif stype == "gmm_weights":
                gmm_weights_key = key
            elif stype == "low_dim":
                state_obs_key_shapes[key] = shape
            # extrinsic/intrinsic: skip — encoder reads them from nobs directly

        self.visual_obs_keys       = list(visual_obs_key_shapes.keys())
        self.state_obs_keys        = sorted(state_obs_key_shapes.keys())
        self.goal_gripper_obs_keys = list(goal_gripper_key_shapes.keys())
        self.gmm_goals_key         = gmm_goals_key
        self.gmm_weights_key       = gmm_weights_key
        self.n_cams = len(self.visual_obs_keys)

        # ------------------------------------------------------------------ #
        # 2. Visual token encoder                                              #
        # ------------------------------------------------------------------ #
        if not visual_obs_key_shapes:
            raise ValueError("No visual observations found in shape_meta.")

        first_shape = next(iter(visual_obs_key_shapes.values()))
        in_channels = first_shape[0]   # C (may be > 3 after early fusion)
        image_size  = first_shape[1]   # H (assumes square images)

        if visual_encoder_cfg is None:
            visual_encoder_cfg = {}

        # Params injected by the policy into every encoder constructor.
        # They always override any same-named key in visual_encoder_cfg.
        common_encoder_cfg = dict(
            n_obs_steps=n_obs_steps,
            embed_dim=input_embedding_dim,
            crop_shape=crop_shape,
            in_channels=in_channels,
            image_size=image_size,
        )

        # Determine which cam keys go to main vs. wrist encoder.
        wrist_cam_key = "cam2_image"
        use_wrist = use_separate_wrist_encoder and wrist_cam_key in visual_obs_key_shapes
        main_cam_keys = (
            [k for k in self.visual_obs_keys if k != wrist_cam_key]
            if use_wrist else self.visual_obs_keys
        )

        def _make_encoder_cfg(cam_keys):
            cfg = dict(**visual_encoder_cfg)   # user YAML values (lower priority)
            cfg.update(common_encoder_cfg)     # injected values win
            cfg["cam_keys"] = cam_keys
            if use_wrist and visual_encoder_type in ("resnet", "resnet_prope"):
                cfg["wrist_cam_key"] = wrist_cam_key
            return cfg

        print(f"[FlowMatchingDiTImagePolicy] visual_encoder_type={visual_encoder_type!r}, "
              f"use_separate_wrist_encoder={use_separate_wrist_encoder}, use_wrist={use_wrist}, "
              f"main_cam_keys={main_cam_keys}"
              + (f", wrist_cam_key={wrist_cam_key!r}" if use_wrist else ""))
        self.visual_encoder = build_visual_encoder(
            visual_encoder_type, _make_encoder_cfg(main_cam_keys)
        )
        import inspect
        self._encoder_accepts_t = 't' in inspect.signature(self.visual_encoder.encode).parameters

        # ------------------------------------------------------------------ #
        # 3. State encoder (low-dim → DiT hidden_state prefix tokens)         #
        # ------------------------------------------------------------------ #
        state_dim = int(sum(int(np.prod(v)) for v in state_obs_key_shapes.values()))
        self.has_state = state_dim > 0
        if self.has_state:
            if use_per_component_state:
                # Per-component (pos/rot/gripper) encoder → 3 sub-tokens per obs step.
                # Assumes standard MimicGen Panda 10-D layout: [pos(3), rot_6D(6), gripper(1)].
                assert state_dim == 10, (
                    f"use_per_component_state=True expects total state_dim=10 "
                    f"(layout: pos(3)+rot_6D(6)+gripper(1)); got state_dim={state_dim}"
                )
                self.state_encoder = PerComponentStateEncoder(
                    embed_dim=input_embedding_dim,
                    n_obs_steps=n_obs_steps,
                    hidden_dim=hidden_size,
                    pos_dim=3, rot_dim=6, gripper_dim=1,
                )
            else:
                self.state_encoder = SimpleMLP(
                    input_dim=state_dim,
                    hidden_dim=hidden_size,
                    output_dim=input_embedding_dim,
                )

        # ------------------------------------------------------------------ #
        # 3b. Goal gripper encoder                                             #
        # ------------------------------------------------------------------ #
        self.has_goal_gripper = len(self.goal_gripper_obs_keys) > 0
        self.has_gmm = gmm_goals_key is not None and gmm_weights_key is not None
        self.use_goal_cross_attention = use_goal_cross_attention
        self.use_weighted_cross_attention = use_weighted_cross_attention
        self.use_parallel_cross_attentions = use_parallel_cross_attentions
        self.use_gated_goal_residual = use_gated_goal_residual
        self.use_goal_adaln_modulation = use_goal_adaln_modulation
        self.use_goal_auxiliary_stream = use_goal_auxiliary_stream
        self.use_per_component_state = use_per_component_state
        self.use_state_adaln = use_state_adaln
        self.log_attention_grad_norms = log_attention_grad_norms
        self.gmm_top_k = gmm_top_k

        if self.has_gmm:
            self.goal_gripper_encoder = KeypointAwareGoalGripperEncoder(
                embed_dim=input_embedding_dim,
                n_obs_steps=n_obs_steps,
                use_gmm=True,
                use_weighted_cross_attention=use_weighted_cross_attention,
            )
            print(f"[FlowMatchingDiTImagePolicy] GMM goal encoder: "
                  f"goals_key={gmm_goals_key!r}, weights_key={gmm_weights_key!r}, "
                  f"use_goal_cross_attention={use_goal_cross_attention}, "
                  f"use_weighted_cross_attention={use_weighted_cross_attention}, "
                  f"use_parallel_cross_attentions={use_parallel_cross_attentions}, "
                  f"use_gated_goal_residual={use_gated_goal_residual}, "
                  f"use_goal_adaln_modulation={use_goal_adaln_modulation}, "
                  f"use_goal_auxiliary_stream={use_goal_auxiliary_stream}, "
                  f"use_per_component_state={use_per_component_state}, "
                  f"use_state_adaln={use_state_adaln}")
        elif self.has_goal_gripper:
            if goal_gripper_encoder_type == "keypoint_aware":
                self.goal_gripper_encoder = KeypointAwareGoalGripperEncoder(
                    embed_dim=input_embedding_dim,
                    n_obs_steps=n_obs_steps,
                )
            else:  # "base"
                self.goal_gripper_encoder = BaseGoalGripperEncoder(
                    embed_dim=input_embedding_dim,
                )
            print(f"[FlowMatchingDiTImagePolicy] goal_gripper_encoder_type={goal_gripper_encoder_type!r}, "
                  f"use_goal_cross_attention={use_goal_cross_attention}, "
                  f"keys={self.goal_gripper_obs_keys}")

        # ------------------------------------------------------------------ #
        # 4. Action encoder (noisy actions + timestep → DiT hidden tokens)    #
        # ------------------------------------------------------------------ #
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_size=input_embedding_dim,
        )

        # ------------------------------------------------------------------ #
        # 5. DiT backbone                                                      #
        # Constraints: num_heads * head_dim == input_embedding_dim            #
        #              output_dim           == hidden_size                    #
        # ------------------------------------------------------------------ #
        if diffusion_model_cfg is None:
            diffusion_model_cfg = {}
        self.model = DiT(**diffusion_model_cfg)

        # ------------------------------------------------------------------ #
        # 6. Action decoder                                                    #
        # ------------------------------------------------------------------ #
        self.action_decoder = SimpleMLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=action_dim,
        )

        # ------------------------------------------------------------------ #
        # 7. Learned positional embedding for action tokens                   #
        # ------------------------------------------------------------------ #
        self.add_pos_embed = add_pos_embed
        if add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len, input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # ------------------------------------------------------------------ #
        # 8. Flow matching noise schedule (Beta distribution)                 #
        # ------------------------------------------------------------------ #
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
        self.noise_s = noise_s
        self.num_timestep_buckets = num_timestep_buckets

        # ------------------------------------------------------------------ #
        # 9. Bookkeeping                                                       #
        # ------------------------------------------------------------------ #
        self.action_dim = action_dim
        self.action_horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_timesteps = num_inference_timesteps
        self.observation_mode = observation_mode

        self.normalizer = LinearNormalizer()

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.visual_encoder.parameters()))

    # ===================================================================== #
    # Internals                                                               #
    # ===================================================================== #

    def _sample_time(self, batch_size: int, device, dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def _encode_obs(self, nobs: dict, batch_size: int, t: Optional[torch.Tensor] = None):
        """
        Convert a normalised obs dict into DiT-ready tokens.

        Parameters
        ----------
        t : (B,) flow timestep in [0, 1], or None.
            Passed to the visual encoder for dynamic RoPE frequency scaling.

        Returns
        -------
        visual_tokens       : (B, n_obs_steps · n_cams · N_tokens, embed_dim)
        state_tokens        : (B, n_obs_steps, embed_dim)  or  None
        goal_gripper_tokens : (B, n_obs_steps [* n_keypoints], embed_dim)  or  None
        goal_weights        : (B, n_obs_steps * N_candidates)  or  None
                              Only non-None when use_weighted_cross_attention=True.
        """
        # -- Visual tokens ------------------------------------------------- #
        if self._encoder_accepts_t:
            visual_tokens = self.visual_encoder.encode(nobs, t=t)
        else:
            visual_tokens = self.visual_encoder.encode(nobs)

        # -- State tokens -------------------------------------------------- #
        state_tokens = None
        if self.has_state:
            state_parts = [
                nobs[k][:, : self.n_obs_steps].reshape(batch_size, self.n_obs_steps, -1)
                for k in self.state_obs_keys
                if k in nobs
            ]
            state = torch.cat(state_parts, dim=-1)       # (B, To, state_dim)
            state_tokens = self.state_encoder(state)     # (B, To, D)

        # -- Goal gripper tokens ------------------------------------------- #
        goal_gripper_tokens = None
        goal_weights = None
        if self.has_gmm:
            goal_pts = nobs[self.gmm_goals_key][:, : self.n_obs_steps]    # (B, To, N, 4, 3)
            weights  = nobs[self.gmm_weights_key][:, : self.n_obs_steps]  # (B, To, N)

            # Keep only the top-K candidates by weight (per batch, per obs step).
            if self.gmm_top_k is not None:
                K = min(self.gmm_top_k, weights.shape[-1])
                topk_idx = torch.topk(weights, K, dim=-1).indices          # (B, To, K)
                # Gather goal_pts: expand idx to match (B, To, K, 4, 3)
                idx_pts = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
                    *topk_idx.shape, goal_pts.shape[-2], goal_pts.shape[-1]
                )
                goal_pts = torch.gather(goal_pts, 2, idx_pts)             # (B, To, K, 4, 3)
                weights  = torch.gather(weights, 2, topk_idx)             # (B, To, K)

            enc_out = self.goal_gripper_encoder(goal_pts, weights=weights)
            if self.use_weighted_cross_attention:
                goal_gripper_tokens, goal_weights = enc_out  # tokens + flat weights
            else:
                goal_gripper_tokens = enc_out
        elif self.has_goal_gripper:
            gripper_parts = [
                nobs[k][:, : self.n_obs_steps]
                for k in self.goal_gripper_obs_keys
                if k in nobs
            ]
            goal_pts = gripper_parts[0] if len(gripper_parts) == 1 else torch.cat(gripper_parts, dim=2)
            goal_gripper_tokens = self.goal_gripper_encoder(goal_pts)

        return visual_tokens, state_tokens, goal_gripper_tokens, goal_weights

    def _run_dit(
        self,
        action_features: torch.Tensor,
        visual_tokens: torch.Tensor,
        state_tokens,
        t_discretized: torch.Tensor,
        goal_gripper_tokens=None,
        goal_weights=None,
    ) -> torch.Tensor:
        """
        When use_goal_cross_attention=False (default):
          hidden_states        = cat([state_tokens, goal_gripper_tokens, action_features])
          goal_hidden_states   = None  (goal tokens participate in self-attention)

        When use_goal_cross_attention=True, use_weighted_cross_attention=False:
          hidden_states        = cat([state_tokens, action_features])
          goal_hidden_states   = goal_gripper_tokens  (standard cross-attn in DiT)

        When use_goal_cross_attention=True, use_weighted_cross_attention=True:
          hidden_states        = cat([state_tokens, action_features])
          goal_hidden_states   = goal_gripper_tokens  (WeightedCrossAttention in DiT)
          goal_weights         = (B, N_goals) GMM probs for log-bias

        encoder_hidden_states = visual_tokens (cross-attended in every even block)
        Returns last action_horizon output tokens: (B, action_horizon, hidden_size)
        """
        parts = []
        if self.has_state and state_tokens is not None:
            parts.append(state_tokens)

        dit_goal_hidden_states = None
        dit_goal_weights = None
        if goal_gripper_tokens is not None:
            if self.use_goal_cross_attention:
                dit_goal_hidden_states = goal_gripper_tokens
                if self.use_weighted_cross_attention:
                    dit_goal_weights = goal_weights
            else:
                parts.append(goal_gripper_tokens)

        parts.append(action_features)
        hidden_states = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        # State summary for StateAdaLN: mean-pool the state tokens to (B, D).
        # Only computed if the flag is on AND state tokens are available.
        state_ctx = None
        if self.use_state_adaln and self.has_state and state_tokens is not None:
            state_ctx = state_tokens.mean(dim=1)   # (B, T_state, D) → (B, D)

        model_output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=visual_tokens,
            timestep=t_discretized,
            goal_hidden_states=dit_goal_hidden_states,
            goal_weights=dit_goal_weights,
            state_ctx=state_ctx,
        )
        return model_output[:, -self.action_horizon :]

    # ===================================================================== #
    # Diffusion policy interface                                              #
    # ===================================================================== #

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, batch: dict) -> torch.Tensor:
        return self.compute_loss(batch)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        nobs    = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        device, dtype = nactions.device, nactions.dtype

        process_observations(nobs, self.observation_mode)

        # Sample t first so it can be passed to the visual encoder for dynamic RoPE.
        noise = torch.randn_like(nactions)
        t = self._sample_time(batch_size, device=device, dtype=dtype)

        visual_tokens, state_tokens, goal_gripper_tokens, goal_weights = self._encode_obs(nobs, batch_size, t=t)
        t_bc = t[:, None, None]                          # (B, 1, 1)
        noisy_actions   = (1 - t_bc) * noise + t_bc * nactions
        velocity_target = nactions - noise

        t_disc = (t * self.num_timestep_buckets).long()  # (B,)

        action_features = self.action_encoder(noisy_actions, t_disc)
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        dit_out      = self._run_dit(action_features, visual_tokens, state_tokens, t_disc, goal_gripper_tokens, goal_weights)
        pred_velocity = self.action_decoder(dit_out)
        # import pdb; pdb.set_trace()
        return F.mse_loss(pred_velocity, velocity_target)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]
        device, dtype = self.device, self.dtype
        # import pdb; pdb.set_trace()
        process_observations(nobs, self.observation_mode)

        # Precompute t-independent visual tokens once; reuse across denoising steps.
        # Encoders with dynamic RoPE expose precompute()+run_rope(); others just use encode().
        _has_rope = hasattr(self.visual_encoder, 'precompute')
        if _has_rope:
            raw_tokens, coords, B = self.visual_encoder.precompute(nobs)
        else:
            visual_tokens_static = self.visual_encoder.encode(nobs)

        # State tokens are t-independent — compute once separately.
        state_tokens = None
        if self.has_state:
            state_parts = [
                nobs[k][:, :self.n_obs_steps].reshape(batch_size, self.n_obs_steps, -1)
                for k in self.state_obs_keys
                if k in nobs
            ]
            state = torch.cat(state_parts, dim=-1)
            state_tokens = self.state_encoder(state)

        # Goal gripper tokens are also t-independent — compute once.
        goal_gripper_tokens = None
        goal_weights = None
        if self.has_gmm:
            goal_pts = nobs[self.gmm_goals_key][:, :self.n_obs_steps]    # (B, To, N, 4, 3)
            weights  = nobs[self.gmm_weights_key][:, :self.n_obs_steps]  # (B, To, N)

            if self.gmm_top_k is not None:
                K = min(self.gmm_top_k, weights.shape[-1])
                topk_idx = torch.topk(weights, K, dim=-1).indices
                idx_pts = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
                    *topk_idx.shape, goal_pts.shape[-2], goal_pts.shape[-1]
                )
                goal_pts = torch.gather(goal_pts, 2, idx_pts)
                weights  = torch.gather(weights, 2, topk_idx)

            enc_out = self.goal_gripper_encoder(goal_pts, weights=weights)
            if self.use_weighted_cross_attention:
                goal_gripper_tokens, goal_weights = enc_out
            else:
                goal_gripper_tokens = enc_out
        elif self.has_goal_gripper:
            gripper_parts = [
                nobs[k][:, :self.n_obs_steps]
                for k in self.goal_gripper_obs_keys
                if k in nobs
            ]
            goal_pts = gripper_parts[0] if len(gripper_parts) == 1 else torch.cat(gripper_parts, dim=2)
            goal_gripper_tokens = self.goal_gripper_encoder(goal_pts)

        # Start from pure noise.
        actions = torch.randn(
            batch_size, self.action_horizon, self.action_dim,
            dtype=dtype, device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((batch_size,), fill_value=t_disc, device=device)

            # Re-run RoPE layers with current t (dynamic freq scaling).
            t_tensor      = torch.full((batch_size,), fill_value=t_cont, dtype=dtype, device=device)
            if _has_rope:
                visual_tokens = self.visual_encoder.run_rope(raw_tokens, coords, B, t=t_tensor)
            else:
                visual_tokens = visual_tokens_static

            action_features = self.action_encoder(actions, timesteps)
            if self.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            dit_out       = self._run_dit(action_features, visual_tokens, state_tokens, timesteps, goal_gripper_tokens, goal_weights)
            pred_velocity = self.action_decoder(dit_out)
            actions       = actions + dt * pred_velocity

        action_pred = self.normalizer["action"].unnormalize(actions)
        return {
            "action":      action_pred[:, : self.n_action_steps],
            "action_pred": action_pred,
        }

    # ===================================================================== #
    # Properties                                                              #
    # ===================================================================== #

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
