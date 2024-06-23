from typing import Dict
import torch
import numpy as np
import copy
import os
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.sampler import (get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from termcolor import cprint

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
            enumerate=False,
            **kwargs
            ):
        super().__init__()
        
        self.task_name = task_name
        self.observation_mode = observation_mode
        
        keys = ['state', 'action', 'point_cloud']
        if 'act3d' in observation_mode:
            keys += ['feature_map', 'gripper_pcd', 'pcd_mask']
            if 'goal' in observation_mode:
                keys += ['goal_gripper_pcd']
            if 'displacement_gripper_to_object' in observation_mode:
                keys += ['displacement_gripper_to_object']
        elif 'act3d_pointnet' == observation_mode:
            keys += ['gripper_pcd']
        
        # try to get kept_in_disk from kwargs, if not, set it to False
        if 'kept_in_disk' in kwargs:
            self.kept_in_disk = kwargs['kept_in_disk']
        else:
            self.kept_in_disk = False 
            
        self.load_per_step = kwargs.get('load_per_step', False)

        if self.kept_in_disk:
            cprint("loading dataset in disk, need a lot of I/O", "red")
            
        if not enumerate:
            from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=keys)
            self.val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            self.val_mask[-int(self.replay_buffer.n_episodes*val_ratio):] = True
            
            train_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            train_mask[:int(self.replay_buffer.n_episodes*train_ratio)] = True
        else:
            # import pdb; pdb.set_trace()

            # if type(zarr_path) != list:
            #     zarr_path = [zarr_path]
            all_zarr_paths = copy.deepcopy(zarr_path)
            
            all_paths = []
            for zarr_path in all_zarr_paths:
                all_subfolder = os.listdir(zarr_path)
                # import pdb; pdb.set_trace()
                for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                    if string in all_subfolder:
                        all_subfolder.remove(string)
                all_subfolder = sorted(all_subfolder)
                n_episodes = len(all_subfolder)
                num_load_episodes = kwargs.get('num_load_episodes', n_episodes)
                num_load_episodes = min(num_load_episodes, n_episodes)
                all_subfolder = all_subfolder[:num_load_episodes]
                zarr_paths = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]
                all_paths += zarr_paths
            
            if not self.kept_in_disk:
                from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
                self.replay_buffer = ReplayBuffer.copy_from_multiple_path(all_paths, keys=keys)
            else:
                from diffusion_policy_3d.common.replay_buffer_disk import ReplayBuffer
                self.replay_buffer = ReplayBuffer.copy_from_multiple_path(all_paths, keys=keys, load_per_step=self.load_per_step)
                self.action_welford = self.replay_buffer.action_welford
            
            self.val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            self.val_mask[-int(self.replay_buffer.n_episodes*val_ratio):] = True
            train_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            train_mask[:int(self.replay_buffer.n_episodes*train_ratio)] = True
            
        
        if not self.kept_in_disk:
            from diffusion_policy_3d.common.sampler import SequenceSampler
            self.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer, 
                sequence_length=horizon,
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=train_mask)
        else:
            from diffusion_policy_3d.common.sampler_disk import SequenceSampler
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
        if not self.kept_in_disk:
            from diffusion_policy_3d.common.sampler import SequenceSampler
            val_set.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer, 
                sequence_length=self.horizon,
                pad_before=self.pad_before, 
                pad_after=self.pad_after,
                episode_mask=self.val_mask
                )
        else:
            from diffusion_policy_3d.common.sampler_disk import SequenceSampler
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
        if not self.kept_in_disk:
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
        else:
            normalizer = LinearNormalizer()
            input_min = self.action_welford.get_min()
            input_max = self.action_welford.get_max()
            input_mean = self.action_welford.get_mean()
            input_std = self.action_welford.get_std()
            input_range = input_max - input_min
            range_eps = 1e-4
            output_min = -1
            output_max = 1
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            scale = torch.from_numpy(scale).float()
            offset = torch.from_numpy(offset).float()
            this_params = torch.nn.ParameterDict({
                'scale': scale,
                'offset': offset,
                'input_stats': torch.nn.ParameterDict({
                    'min': input_min,
                    'max': input_max,
                    'mean': input_mean,
                    'std': input_std
                })
            })
            for p in this_params.values():
                p.requires_grad = False
            normalizer.params_dict['action'] = this_params

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
            pcd_mask = sample['pcd_mask'][:,].astype(np.uint8)
            data['obs']['gripper_pcd'] = gripper_pcd
            data['obs']['feature_map'] = feature_map
            data['obs']['pcd_mask'] = pcd_mask
            if 'goal' in self.observation_mode:
                data['obs']['goal_gripper_pcd'] = sample['goal_gripper_pcd'][:,].astype(np.float32)
            if 'displacement_gripper_to_object' in self.observation_mode:
                data['obs']['displacement_gripper_to_object'] = sample['displacement_gripper_to_object'][:,].astype(np.float32)
        elif 'act3d_pointnet' == self.observation_mode:
            gripper_pcd = sample['gripper_pcd'][:,].astype(np.float32)
            data['obs']['gripper_pcd'] = gripper_pcd
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data