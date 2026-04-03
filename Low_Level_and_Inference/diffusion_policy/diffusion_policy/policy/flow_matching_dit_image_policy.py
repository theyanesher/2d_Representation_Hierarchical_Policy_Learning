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
from diffusion_policy.model.flow_matching.helpers import ActionEncoder, SimpleMLP
from diffusion_policy.model.flow_matching.visual_encoders import build_visual_encoder


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

        for key, attr in obs_shape_meta.items():
            shape = list(attr["shape"])
            stype = attr.get("type", "low_dim")
            if stype in ("rgb", "depth", "pointmap", "plucker", "heatmap", "ghost_heatmap"):
                visual_obs_key_shapes[key] = shape
            elif stype == "low_dim":
                state_obs_key_shapes[key] = shape
            # extrinsic/intrinsic: skip — encoder reads them from nobs directly

        self.visual_obs_keys = list(visual_obs_key_shapes.keys())
        self.state_obs_keys  = sorted(state_obs_key_shapes.keys())
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
            self.state_encoder = SimpleMLP(
                input_dim=state_dim,
                hidden_dim=hidden_size,
                output_dim=input_embedding_dim,
            )

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
        visual_tokens : (B, n_obs_steps · n_cams · N_tokens, embed_dim)
        state_tokens  : (B, n_obs_steps, embed_dim)  or  None
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

        return visual_tokens, state_tokens

    def _run_dit(
        self,
        action_features: torch.Tensor,
        visual_tokens: torch.Tensor,
        state_tokens,
        t_discretized: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states        = cat([state_tokens, action_features])
        encoder_hidden_states = visual_tokens
        Returns last action_horizon output tokens: (B, action_horizon, hidden_size)
        """
        if self.has_state and state_tokens is not None:
            hidden_states = torch.cat([state_tokens, action_features], dim=1)
        else:
            hidden_states = action_features

        model_output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=visual_tokens,
            timestep=t_discretized,
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

        visual_tokens, state_tokens = self._encode_obs(nobs, batch_size, t=t)
        t_bc = t[:, None, None]                          # (B, 1, 1)
        noisy_actions   = (1 - t_bc) * noise + t_bc * nactions
        velocity_target = nactions - noise

        t_disc = (t * self.num_timestep_buckets).long()  # (B,)

        action_features = self.action_encoder(noisy_actions, t_disc)
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        dit_out      = self._run_dit(action_features, visual_tokens, state_tokens, t_disc)
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

            dit_out       = self._run_dit(action_features, visual_tokens, state_tokens, timesteps)
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
