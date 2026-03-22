# Copyright (c) Sudeep Dasari, 2023
# Heavy inspiration taken from DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# and DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply

class DiTBlockPolicy(BaseImagePolicy):
    def __init__(
        self, agent, n_obs_steps, n_action_steps
    ):
        super().__init__()

        self.policy = hydra.utils.instantiate(agent)

        # inherited from BaseImagePolicy
        self.normalizer = LinearNormalizer()
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

    def predict_action(self, obs_dict: dict) -> dict:
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        # prepare agent inputs
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        To = self.n_obs_steps
        
        imgs = {k: v for k, v in nobs.items() if k.startswith('cam')}
        obs = nobs['state']

        # run agent
        naction_pred = self.policy.get_actions(imgs, obs)
        
        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        imgs = {k: v for k, v in nobs.items() if k.startswith('cam')}
        obs = nobs['state']
        
        # agent forward pass
        return self.policy(imgs, obs, nactions)
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @property
    def ac_chunk(self):
        return self.policy.ac_chunk

    @property
    def ac_dim(self):
        return self.policy.ac_dim