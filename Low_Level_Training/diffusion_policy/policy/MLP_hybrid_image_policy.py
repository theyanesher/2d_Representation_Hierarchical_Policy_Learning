from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class MLPHybridImagePolicy(BaseImagePolicy):
    """
    Behavioral cloning MLP baseline.

    Uses the same ResNet observation encoder as DiffusionUnetHybridImagePolicy,
    but replaces the diffusion UNet with a simple MLP that directly regresses
    the next n_action_steps of actions from the encoded observation.

    Training loss: MSE in normalized action space (committed steps only).
    Inference:     single forward pass — fully deterministic, no stochasticity.

    Use this as a debugging baseline to verify that the obs encoder + data
    pipeline can overfit to a small dataset before running the diffusion model.
    """

    def __init__(self,
            shape_meta: dict,
            horizon,
            n_action_steps,
            n_obs_steps,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            obs_encoder_group_norm=True,
            eval_fixed_crop=True,
            mlp_hidden_dims=(1024, 512, 256),
            lang_cond=True,
            **kwargs):
        super().__init__()

        # -------- parse shape_meta (identical to diffusion policy) --------
        self.lang_cond = lang_cond
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        obs_shape_meta = shape_meta['obs']
        obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                obs_config['rgb'].append(key)
            elif obs_type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        # -------- build robomimic obs encoder (identical to diffusion policy) --------
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')

        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features)
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * n_obs_steps  # flattened obs features

        # -------- MLP head --------
        output_dim = n_action_steps * action_dim
        layers = []
        in_dim = global_cond_dim
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        mlp = nn.Sequential(*layers)

        # -------- store --------
        self.obs_encoder = obs_encoder
        self.mlp = mlp
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond

        print("MLP params:    %e" % sum(p.numel() for p in self.mlp.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    # ========= helpers ============

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _encode_obs(self, nobs: dict, lang_emb, batch_size: int) -> torch.Tensor:
        """Encode first n_obs_steps frames -> (B, obs_feature_dim * n_obs_steps)."""
        To = self.n_obs_steps
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs, lang_emb=lang_emb)
        return nobs_features.reshape(batch_size, -1)  # (B, obs_feature_dim * n_obs_steps)

    # ========= inference ============

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], lang_emb) -> Dict[str, torch.Tensor]:
        """
        obs_dict: same structure as used by the diffusion policy
        Returns:
            result['action']      - (B, n_action_steps, action_dim)  unnormalized, ready to execute
            result['action_pred'] - dict with 'action' key (same as above)
        """
        nobs = self.normalizer.normalize(obs_dict, seperate_params_pos_ori=True)
        B = next(iter(nobs.values())).shape[0]

        global_cond = self._encode_obs(nobs, lang_emb, B)  # (B, global_cond_dim)

        pred_normalized = self.mlp(global_cond)             # (B, n_action_steps * action_dim)
        pred_normalized = pred_normalized.reshape(B, self.n_action_steps, self.action_dim)

        action_pred = self.normalizer.unnormalize(
            {"action": pred_normalized}, seperate_params_pos_ori=True)

        action = action_pred["action"]  # (B, n_action_steps, action_dim)
        return {
            'action': action,
            'action_pred': action_pred,
        }

    # ========= training ============

    def compute_loss(self, batch):
        """
        MSE between MLP prediction and GT committed action steps (normalized space).
        Committed window: batch['action'][:, n_obs_steps-1 : n_obs_steps-1+n_action_steps, :]
        """
        nobs = self.normalizer.normalize(batch['obs'], seperate_params_pos_ori=True)
        nactions = self.normalizer.normalize(
            {"action": batch['action']}, seperate_params_pos_ori=True)["action"]

        # committed GT steps in normalized space
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        nactions_committed = nactions[:, start:end, :]  # (B, n_action_steps, action_dim)

        batch_size = nactions.shape[0]
        global_cond = self._encode_obs(nobs, batch['obs_lang_emb'], batch_size)

        pred = self.mlp(global_cond)                                     # (B, n_action_steps * action_dim)
        pred = pred.reshape(batch_size, self.n_action_steps, self.action_dim)

        loss = F.mse_loss(pred, nactions_committed)
        return loss
