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
import random
import copy

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
            is_pickle=False,
            dataset_keys=None,
            augmentation_pcd=False,
            **kwargs
            ):
        super().__init__()
        
        self.task_name = task_name
        self.observation_mode = observation_mode
        self.augmentation_pcd = augmentation_pcd
        
        if dataset_keys is None:
            keys = ['state', 'action', 'point_cloud']
            if 'act3d' in observation_mode:
                keys += ['feature_map', 'gripper_pcd', 'pcd_mask']
                if 'goal' in observation_mode:
                    keys += ['goal_gripper_pcd']
                if 'displacement_gripper_to_object' in observation_mode:
                    keys += ['displacement_gripper_to_object']
            elif 'act3d_pointnet' == observation_mode:
                keys += ['gripper_pcd']
        else:
            cprint(f"specifying dataset_keys: {dataset_keys}", "red")
            keys = dataset_keys

        self.keys_ = keys
        self.is_pickle = is_pickle
        
        # try to get kept_in_disk from kwargs, if not, set it to False
        if 'kept_in_disk' in kwargs:
            self.kept_in_disk = kwargs['kept_in_disk']
        else:
            self.kept_in_disk = False 
            
        self.load_per_step = kwargs.get('load_per_step', False)

        self.only_reach_stage = kwargs.get('only_reach_stage', False)

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
            train_masks = []
            val_masks = []
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
                folder_train_mask = np.zeros(num_load_episodes, dtype=bool)
                folder_train_mask[:int(num_load_episodes*train_ratio)] = True
                train_masks.append(folder_train_mask)
                folder_val_mask = np.zeros(num_load_episodes, dtype=bool)
                folder_val_mask[-int(num_load_episodes*val_ratio):] = True
                val_masks.append(folder_val_mask)
            
            if not self.kept_in_disk:
                from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
                self.replay_buffer = ReplayBuffer.copy_from_multiple_path(all_paths, keys=keys)
            else:
                from diffusion_policy_3d.common.replay_buffer_disk import ReplayBuffer
                self.replay_buffer = ReplayBuffer.copy_from_multiple_path(all_paths, keys=keys, load_per_step=self.load_per_step, only_reach_stage=self.only_reach_stage, is_pickle=self.is_pickle)
                self.action_welford = self.replay_buffer.action_welford
            
            # self.val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            # self.val_mask[-int(self.replay_buffer.n_episodes*val_ratio):] = True
            # train_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            # train_mask[:int(self.replay_buffer.n_episodes*train_ratio)] = True
            train_mask = np.concatenate(train_masks)
            self.val_mask = np.concatenate(val_masks)

        
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
        agent_pos = copy.deepcopy(sample['state'][:,].astype(np.float32)) # (T, agent_pos: 7) T is the horizon
        point_cloud = copy.deepcopy(sample['point_cloud'][:,].astype(np.float32))
        action = copy.deepcopy(sample['action'].astype(np.float32))
        agent_pos = copy.deepcopy(agent_pos)
        if 'act3d' in self.observation_mode:
            gripper_pcd = copy.deepcopy(sample['gripper_pcd'][:,].astype(np.float32))
            feature_map = copy.deepcopy(sample['feature_map'][:,].astype(np.float32))
            pcd_mask = copy.deepcopy(sample['pcd_mask'][:,].astype(np.uint8))
            if 'goal' in self.observation_mode:
                goal_gripper_pcd = copy.deepcopy(sample['goal_gripper_pcd'][:,].astype(np.float32))
            if 'displacement_gripper_to_object' in self.observation_mode:
                displacement_gripper_to_object = copy.deepcopy(sample['displacement_gripper_to_object'][:,].astype(np.float32))
        if self.augmentation_pcd:
            point_cloud = pointcloud + np.random.normal(0, 0.005, point_cloud.shape)
        data = {
            'obs': {
                'point_cloud': point_cloud.astype(np.float32) 
                'agent_pos': agent_pos.astype(np.float32), 
            },
            'action': action.astype(np.float32)
        }

        if 'act3d' in self.observation_mode:
            data['obs']['gripper_pcd'] = gripper_pcd.astype(np.float32)
            data['obs']['feature_map'] = feature_map.astype(np.float32)
            data['obs']['pcd_mask'] = pcd_mask.astype(np.uint8)
            if 'goal' in self.observation_mode:
                data['obs']['goal_gripper_pcd'] = goal_gripper_pcd.astype(np.float32)
            if 'displacement_gripper_to_object' in self.observation_mode:
                data['obs']['displacement_gripper_to_object'] = displacement_gripper_to_object.astype(np.float32)

        return data

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data