"""
FlowMatchingDiTGoalGMMPolicy
============================
Approach 2 ablation: the low-level policy is NOT given the goal. Instead the goal
supervises the visual representation through an auxiliary GMM head.

Differences from ``FlowMatchingDiTImagePolicy``:
  1. ``goal_gripper_pts`` is a training TARGET, never an input. The DiT's
     hidden_states stay ``[state_tokens, action_features]``.
  2. The plain DINOv2 encoder is replaced by ``DINOv2RoPE4DGroundedEncoder``,
     which gives each patch token a world-frame 3D anchor and fuses patches with
     the gripper keypoints via 4D-RoPE self-attention.
  3. An auxiliary head reads the SAME grounded tokens the DiT cross-attends to
     and is trained to predict the goal. Two kinds:
       - ``aux_head_type="gmm"``: ``GoalGMMHead``, trained with the ArticuBot
         goal-GMM NLL (a mixture over every anchor).
       - ``aux_head_type="regression"``: ``GoalRegressionHead``, trained with
         plain MSE to a single pooled subgoal prediction. ``aux_regression_frame``
         picks the target: ``"absolute"`` (world-frame goal keypoints) or
         ``"relative_to_gripper"`` (goal - present_gripper_pts).

Everything downstream of the fork is untouched: the DiT, action encoder/decoder,
flow-matching schedule, and the DiT's own state encoder are all as before, and
none of them receive gradient from the auxiliary loss. The only channel by which
the auxiliary loss can influence the policy is the grounded visual tokens.

Set ``aux_gmm_loss_weight: null`` to get the no-auxiliary-loss control arm (same
architecture, flow-matching loss only) -- ``aux_head_type`` is then irrelevant.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor

from diffusion_policy.common.obs_util import process_observations
from diffusion_policy.model.flow_matching.goal_gmm_head import (
    FIXED_VARIANCE, GoalGMMHead, goal_gmm_loss,
)
from diffusion_policy.model.flow_matching.goal_regression_head import (
    REGRESSION_FRAMES, GoalRegressionHead, goal_regression_loss,
    goal_regression_target,
)
from diffusion_policy.model.flow_matching.grounded_encoder import (
    DINOv2RoPE4DGroundedEncoder,
)
from diffusion_policy.policy.flow_matching_dit_image_policy import (
    FlowMatchingDiTImagePolicy,
)


class FlowMatchingDiTGoalGMMPolicy(FlowMatchingDiTImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # ---- auxiliary goal head ----
        # None disables the head entirely (the architecture-matched control arm).
        aux_gmm_loss_weight: Optional[float] = 0.01,
        aux_gmm_hidden_dim: int = 512,
        aux_goal_key: str = "goal_gripper_pts",
        gripper_key: str = "present_gripper_pts",
        # "gmm" (mixture NLL, default) or "regression" (single pooled subgoal, MSE).
        aux_head_type: str = "gmm",
        # only used when aux_head_type == "regression": "absolute" or
        # "relative_to_gripper" -- see goal_regression_head.py.
        aux_regression_frame: str = "absolute",
        # ---- grounded encoder ----
        patch_size: int = 14,
        n_trunk_layers: int = 2,
        xyz_scale: float = 5.0,
        time_scale: float = 1.0,
        base_frequency: float = 100.0,
        **kwargs,
    ):
        # The parent builds a visual encoder from every rgb/depth-typed key, which
        # is not what we want (depth is geometry, not a camera). It is replaced
        # below; the transient backbone is discarded.
        super().__init__(
            shape_meta=shape_meta,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            **kwargs,
        )

        if aux_head_type not in ("gmm", "regression"):
            raise ValueError(f"aux_head_type must be 'gmm' or 'regression', got "
                              f"{aux_head_type!r}")
        if aux_regression_frame not in REGRESSION_FRAMES:
            raise ValueError(f"aux_regression_frame must be one of "
                              f"{REGRESSION_FRAMES}, got {aux_regression_frame!r}")

        self.aux_goal_key = aux_goal_key
        self.gripper_key = gripper_key
        self.aux_gmm_loss_weight = aux_gmm_loss_weight
        self.aux_regression_frame = aux_regression_frame
        self.n_keypoints = int(shape_meta["obs"][aux_goal_key]["shape"][0])

        # -- 1. The goal is a target, never an input ------------------------- #
        # Both goal_gripper_pts and present_gripper_pts are 'goal_gripper'-typed
        # so the dataset serves them raw; neither may reach the DiT.
        self.goal_gripper_obs_keys = []
        self.has_goal_gripper = False
        self.has_gmm = False
        if hasattr(self, "goal_gripper_encoder"):
            del self.goal_gripper_encoder

        # -- 2. Grounded visual encoder -------------------------------------- #
        rgb_cam_keys = [
            k for k, a in shape_meta["obs"].items() if a.get("type") == "rgb"
        ]
        assert rgb_cam_keys, "no rgb-typed obs keys found"
        rgb_cam_keys = sorted(rgb_cam_keys)
        img_shape = list(shape_meta["obs"][rgb_cam_keys[0]]["shape"])

        enc_cfg = dict(kwargs.get("visual_encoder_cfg") or {})
        enc_cfg.update(dict(
            cam_keys=rgb_cam_keys,
            n_obs_steps=n_obs_steps,
            embed_dim=kwargs.get("input_embedding_dim", 512),
            crop_shape=kwargs.get("crop_shape", (224, 224)),
            in_channels=img_shape[0],
            image_size=img_shape[1],
            gripper_key=gripper_key,
            n_total_steps=n_obs_steps + horizon,
            n_keypoints=self.n_keypoints,
            patch_size=patch_size,
            n_trunk_layers=n_trunk_layers,
            xyz_scale=xyz_scale,
            time_scale=time_scale,
            base_frequency=base_frequency,
            num_heads=kwargs["diffusion_model_cfg"]["num_attention_heads"],
            head_dim=kwargs["diffusion_model_cfg"]["attention_head_dim"],
        ))
        self.visual_encoder = DINOv2RoPE4DGroundedEncoder(**enc_cfg)

        # -- 3. Auxiliary head ----------------------------------------------- #
        self.gmm_head = None
        self.regression_head = None
        self.aux_head_type = aux_head_type if aux_gmm_loss_weight is not None else None
        if self.aux_head_type == "gmm":
            self.gmm_head = GoalGMMHead(
                token_dim=kwargs.get("input_embedding_dim", 512),
                hidden_dim=aux_gmm_hidden_dim,
                n_keypoints=self.n_keypoints,
            )
        elif self.aux_head_type == "regression":
            self.regression_head = GoalRegressionHead(
                token_dim=kwargs.get("input_embedding_dim", 512),
                hidden_dim=aux_gmm_hidden_dim,
                n_keypoints=self.n_keypoints,
            )

        print(
            f"[FlowMatchingDiTGoalGMMPolicy] goal removed from DiT input; "
            f"aux_gmm_loss_weight={aux_gmm_loss_weight}, aux_head_type={aux_head_type}, "
            + (f"aux_regression_frame={aux_regression_frame}, "
               if self.aux_head_type == "regression" else "")
            + (f"variances={list(FIXED_VARIANCE)}, " if self.aux_head_type == "gmm" else "")
            + f"n_trunk_layers={n_trunk_layers}, xyz_scale={xyz_scale}, "
            f"time_scale={time_scale}, cams={rgb_cam_keys}"
        )

    # ===================================================================== #
    def _state_tokens(self, nobs: dict, batch_size: int) -> Optional[Tensor]:
        """DiT state tokens — unchanged from the baseline, and never touched by
        the auxiliary loss (the trunk has its own gripper encoder)."""
        if not self.has_state:
            return None
        parts = [
            nobs[k][:, : self.n_obs_steps].reshape(batch_size, self.n_obs_steps, -1)
            for k in self.state_obs_keys
            if k in nobs
        ]
        return self.state_encoder(torch.cat(parts, dim=-1))

    def _gmm_anchor_stack(
        self,
        vis_tokens: Tensor, vis_xyz: Tensor, vis_valid: Tensor,
        grip_tokens: Tensor, grip_xyz: Tensor,
    ):
        """Fold the obs-step axis into the batch: one mixture per (sample, step).

        Grouping per obs step matters because ``goal_gripper_pts`` is stored per
        timestep and jumps at phase boundaries; a single pooled mixture would
        supervise one step's anchors against the other step's goal. Folding the
        obs-step axis into the batch makes the per-mixture softmax fall out.

        Returns (tokens, anchors, valid), each with leading dim ``B * To`` and
        anchor order [gripper keypoints ; patch anchors] — gripper first,
        mirroring ArticuBot's ``prepare_scene_pcd``.
        """
        B, To, K = vis_tokens.shape[0], self.n_obs_steps, self.n_keypoints
        D = vis_tokens.shape[-1]
        n_per_step = vis_tokens.shape[1] // To      # n_cams * N_tok

        # Token order is (obs_step, cam, patch), so obs_step is outermost and this
        # reshape groups correctly.
        vt = vis_tokens.reshape(B * To, n_per_step, D)
        vx = vis_xyz.reshape(B * To, n_per_step, 3)
        vv = vis_valid.reshape(B * To, n_per_step)

        gt_tok = grip_tokens.reshape(B * To, K, D)
        gx = grip_xyz.reshape(B * To, K, 3)
        gv = torch.ones(B * To, K, dtype=torch.bool, device=vt.device)

        tokens = torch.cat([gt_tok, vt], dim=1)
        anchors = torch.cat([gx, vx], dim=1)
        valid = torch.cat([gv, vv], dim=1)
        return tokens, anchors, valid

    def _compute_goal_gmm_loss(
        self,
        vis_tokens: Tensor, vis_xyz: Tensor, vis_valid: Tensor,
        grip_tokens: Tensor, grip_xyz: Tensor,
        goal: Tensor,
    ) -> Tensor:
        """One mixture per obs step over [gripper keypoints ; patch anchors]."""
        B, To, K = vis_tokens.shape[0], self.n_obs_steps, self.n_keypoints
        tokens, anchors, valid = self._gmm_anchor_stack(
            vis_tokens, vis_xyz, vis_valid, grip_tokens, grip_xyz,
        )
        goal_t = goal[:, :To].reshape(B * To, K, 3).to(anchors.dtype)

        pred_disp, logits = self.gmm_head(tokens, anchors)
        gt_disp = goal_t[:, None, :, :] - anchors[:, :, None, :]
        return goal_gmm_loss(pred_disp, gt_disp, logits, valid)

    def _compute_goal_regression_loss(
        self,
        vis_tokens: Tensor, vis_xyz: Tensor, vis_valid: Tensor,
        grip_tokens: Tensor, grip_xyz: Tensor,
        goal: Tensor,
    ) -> Tensor:
        """One pooled subgoal prediction per obs step, scored with MSE.

        Same token/valid stack ``_compute_goal_gmm_loss`` reads (gripper
        keypoints + patch anchors); the anchors themselves are unused here
        since the head pools instead of scoring per-anchor.
        """
        B, To, K = vis_tokens.shape[0], self.n_obs_steps, self.n_keypoints
        tokens, _, valid = self._gmm_anchor_stack(
            vis_tokens, vis_xyz, vis_valid, grip_tokens, grip_xyz,
        )
        goal_t = goal[:, :To].reshape(B * To, K, 3).to(tokens.dtype)
        present_t = grip_xyz.reshape(B * To, K, 3).to(tokens.dtype)

        pred = self.regression_head(tokens, valid)
        target = goal_regression_target(goal_t, present_t, self.aux_regression_frame)
        return goal_regression_loss(pred, target)

    # ===================================================================== #
    def compute_loss(self, batch: dict) -> Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        device, dtype = nactions.device, nactions.dtype

        process_observations(nobs, self.observation_mode)

        # raw_obs carries metric depth, camera matrices and world-frame gripper
        # keypoints; the normalizer would put depth and state in the wrong frame.
        vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz = \
            self.visual_encoder.encode_with_positions(nobs, batch["obs"])
        state_tokens = self._state_tokens(nobs, batch_size)

        noise = torch.randn_like(nactions)
        t = self._sample_time(batch_size, device=device, dtype=dtype)
        t_bc = t[:, None, None]
        noisy_actions = (1 - t_bc) * noise + t_bc * nactions
        velocity_target = nactions - noise
        t_disc = (t * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy_actions, t_disc)
        if self.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device,
            )
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        dit_out = self._run_dit(action_features, vis_tok, state_tokens, t_disc)
        pred_velocity = self.action_decoder(dit_out)
        fm_loss = F.mse_loss(pred_velocity, velocity_target)

        if self.aux_head_type is None:
            return fm_loss

        goal = batch["obs"][self.aux_goal_key]
        if self.aux_head_type == "gmm":
            aux_loss = self._compute_goal_gmm_loss(
                vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz, goal,
            )
            aux_loss_name = "goal_gmm_loss"
        else:
            aux_loss = self._compute_goal_regression_loss(
                vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz, goal,
            )
            aux_loss_name = "goal_regression_loss"

        if wandb.run is not None:
            prefix = "train" if self.training else "val"
            wandb.log({
                f"{prefix}_fm_loss": fm_loss.item(),
                f"{prefix}_{aux_loss_name}": aux_loss.item(),
            }, commit=False)
        return fm_loss + self.aux_gmm_loss_weight * aux_loss

    # ===================================================================== #
    def _flow_sample(
        self,
        vis_tok: Tensor,
        state_tokens: Optional[Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Euler-integrate the flow from noise to a NORMALISED action chunk.

        The single implementation of the sampler — ``predict_action`` and
        ``predict_for_eval`` both go through here, so an offline visualisation
        can never drift from what the policy actually does at deployment.

        Args:
            vis_tok:      (N, n_vis, D) grounded visual tokens.
            state_tokens: (N, To, D) or None.
            generator:    optional RNG for reproducible initial noise.

        Returns:
            (N, action_horizon, action_dim) normalised actions.
        """
        n = vis_tok.shape[0]
        device, dtype = self.device, self.dtype

        actions = torch.randn(
            n, self.action_horizon, self.action_dim,
            dtype=dtype, device=device, generator=generator,
        )

        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((n,), fill_value=t_disc, device=device)

            action_features = self.action_encoder(actions, timesteps)
            if self.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device,
                )
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            dit_out = self._run_dit(action_features, vis_tok, state_tokens, timesteps)
            actions = actions + dt * self.action_decoder(dit_out)

        return actions

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]

        process_observations(nobs, self.observation_mode)

        # The trunk has no flow-timestep conditioning, so the grounded tokens are
        # computed once and reused across every Euler step. The GMM head is not
        # called at inference — it only ever shaped the trunk during training.
        vis_tok, _, _, _, _ = self.visual_encoder.encode_with_positions(nobs, obs_dict)
        state_tokens = self._state_tokens(nobs, batch_size)

        actions = self._flow_sample(vis_tok, state_tokens)

        action_pred = self.normalizer["action"].unnormalize(actions)
        return {
            "action": action_pred[:, : self.n_action_steps],
            "action_pred": action_pred,
        }

    # ===================================================================== #
    @torch.no_grad()
    def predict_for_eval(
        self,
        obs_dict: Dict[str, Tensor],
        n_action_samples: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Tensor]:
        """Diagnostic readout: action chunk(s) AND the auxiliary goal mixture.

        Inference-only. Not used in training or deployment — it exists so an
        offline/online evaluation can see the subgoal the auxiliary head believes
        in, which ``predict_action`` deliberately never computes (the head only
        ever shaped the trunk).

        One encoder pass feeds both consumers, so the mixture shown is exactly
        the one read off the tokens the DiT cross-attended to for these actions.

        Args:
            obs_dict:         (B, To, ...) raw (un-normalised) observations.
            n_action_samples: independent flow samples per observation. >1
                              exposes the policy's own multimodality; the
                              encoder still runs only once.
            generator:        optional RNG for reproducible sampling.

        Returns (all detached, on the policy's device):
            action_pred : (B, S, horizon, action_dim) un-normalised actions

            when ``aux_head_type == "gmm"``, additionally:
              anchors     : (B, To, N, 3)      world-frame mixture anchors,
                                               ordered [gripper K ; patches]
              valid       : (B, To, N)         bool, depth-valid anchors
              weights     : (B, To, N)         mixture weights pi (masked softmax)
              logits      : (B, To, N)         raw per-anchor logits
              mu          : (B, To, N, K, 3)   per-component goal keypoints
                                               (anchor + predicted displacement)
              n_gripper_anchors : int          how many leading anchors are the
                                               gripper keypoints

            when ``aux_head_type == "regression"``, additionally:
              subgoal_pred : (B, To, K, 3)     the single pooled subgoal
                                               prediction, always in WORLD
                                               coordinates (already added back
                                               to present_gripper_pts if
                                               ``aux_regression_frame ==
                                               "relative_to_gripper"``)

        These extra keys are absent when the policy was built with
        ``aux_gmm_loss_weight=null`` (the architecture-matched control arm).
        """
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        process_observations(nobs, self.observation_mode)

        vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz = \
            self.visual_encoder.encode_with_positions(nobs, obs_dict)
        state_tokens = self._state_tokens(nobs, B)

        # -- actions ---------------------------------------------------------- #
        S = max(int(n_action_samples), 1)
        vt = vis_tok if S == 1 else vis_tok.repeat_interleave(S, dim=0)
        st = state_tokens
        if st is not None and S > 1:
            st = st.repeat_interleave(S, dim=0)
        actions = self._flow_sample(vt, st, generator=generator)
        action_pred = self.normalizer["action"].unnormalize(actions)
        out = {"action_pred": action_pred.reshape(B, S, *action_pred.shape[1:])}

        if self.aux_head_type is None:
            return out
        if self.aux_head_type == "gmm":
            out.update(self._gmm_readout(
                vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz, batch_size=B))
        else:
            out.update(self._regression_readout(
                vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz, batch_size=B))
        return out

    @torch.no_grad()
    def _regression_readout(
        self,
        vis_tokens: Tensor, vis_xyz: Tensor, vis_valid: Tensor,
        grip_tokens: Tensor, grip_xyz: Tensor,
        batch_size: int,
    ) -> Dict[str, Tensor]:
        """The regression counterpart of ``_gmm_readout``: same anchor stack,
        just one pooled prediction instead of a mixture. Always reported in
        world coordinates, regardless of ``aux_regression_frame``."""
        B, To, K = batch_size, self.n_obs_steps, self.n_keypoints
        tokens, _, valid = self._gmm_anchor_stack(
            vis_tokens, vis_xyz, vis_valid, grip_tokens, grip_xyz,
        )
        pred = self.regression_head(tokens, valid)              # (B*To, K, 3)
        present = grip_xyz.reshape(B * To, K, 3)
        subgoal = pred if self.aux_regression_frame == "absolute" else present + pred
        return {"subgoal_pred": subgoal.reshape(B, To, K, 3)}

    @torch.no_grad()
    def _gmm_readout(
        self,
        vis_tokens: Tensor, vis_xyz: Tensor, vis_valid: Tensor,
        grip_tokens: Tensor, grip_xyz: Tensor,
        batch_size: int,
    ) -> Dict[str, Tensor]:
        """Run the auxiliary head and unfold the mixture back to (B, To, ...).

        The inference-side counterpart of ``_compute_goal_gmm_loss``: same anchor
        stack, same per-obs-step grouping, same masking — it just reports the
        mixture instead of scoring it against a goal.
        """
        B, To, K = batch_size, self.n_obs_steps, self.n_keypoints
        tokens, anchors, valid = self._gmm_anchor_stack(
            vis_tokens, vis_xyz, vis_valid, grip_tokens, grip_xyz,
        )
        disp, logits = self.gmm_head(tokens, anchors)        # (BTo,N,K,3), (BTo,N)
        mu = anchors[:, :, None, :] + disp                   # (BTo, N, K, 3)

        # Same masking the NLL uses, so a weight shown here is a weight the loss
        # saw: dead anchors are removed from the mixture, not merely downweighted.
        weights = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)

        N = anchors.shape[1]
        return {
            "anchors": anchors.reshape(B, To, N, 3),
            "valid":   valid.reshape(B, To, N),
            "logits":  logits.reshape(B, To, N),
            "weights": weights.reshape(B, To, N),
            "mu":      mu.reshape(B, To, N, K, 3),
            "n_gripper_anchors": K,
        }
