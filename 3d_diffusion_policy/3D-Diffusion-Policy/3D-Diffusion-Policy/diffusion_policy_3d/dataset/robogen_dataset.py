from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class RobogenDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.1,
            train_ratio=0.9,
            max_train_episodes=None,
            task_name=None,
            observation_mode='segmask',
            ):
        super().__init__()
        
        self.task_name = task_name
        self.observation_mode = observation_mode
        
        keys = ['state', 'action', 'point_cloud']
        if 'act3d' in observation_mode:
            keys += ['feature_map', 'gripper_pcd']
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
        
        self.val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
        self.val_mask[-int(self.replay_buffer.n_episodes*val_ratio):] = True
        
        train_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
        train_mask[:int(self.replay_buffer.n_episodes*train_ratio)] = True

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after 
            
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set
    

    def get_normalizer(self, mode='limits', **kwargs):
        # TODO: do we need to normalize the agent_pos and point cloud?
        # or just center point cloud to be at robot gripper?
        if 'act3d' not in self.observation_mode:
            data = {
                'action': self.replay_buffer['action'],
                'agent_pos': self.replay_buffer['state'][...,:],
                'point_cloud': self.replay_buffer['point_cloud'],
            }
        else:
            # only normalizes actions, to make sure that the relative attention makes sense
            data = {
                'action': self.replay_buffer['action'],
            }
            
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (T, agent_pos: 7) T is the horizon
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1280, 6)
       
        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1280, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }

        if 'act3d' in self.observation_mode:
            gripper_pcd = sample['gripper_pcd'][:,].astype(np.float32)
            feature_map = sample['feature_map'][:,].astype(np.float32)
            data['obs']['gripper_pcd'] = gripper_pcd
            data['obs']['feature_map'] = feature_map
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data