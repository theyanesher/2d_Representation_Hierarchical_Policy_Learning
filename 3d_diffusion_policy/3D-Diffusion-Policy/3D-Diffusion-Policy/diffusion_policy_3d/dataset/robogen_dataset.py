from typing import Dict
import torch
import time
import numpy as np
import copy
import os
from tqdm import tqdm
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.sampler import (get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.Augmentations.aug_translation_xy import TranslationXY
from diffusion_policy_3d.dataset.Augmentations.aug_rotation_z import rotationZ
from diffusion_policy_3d.dataset.Augmentations.random_apply_numpy import RandomApplyNumpy
from termcolor import cprint
import random
import copy

import pybullet as p
from manipulation.utils import get_pc, get_pc_in_camera_frame, rotation_transfer_6D_to_matrix_batch, rotation_transfer_matrix_to_6D_batch, add_sphere, get_pixel_location, get_matrix_from_pos_rot

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
            augmentation_scale=False,
            scale_scene_by_pcd=False,
            use_absolute_waypoint=False,
            augmentation_rot=False,
            object_augmentation_high_level = False,
            mean_x_augmentation_high_level = None, 
            mean_y_augmentation_high_level = None, 
            std_x_augmentation_high_level = None, 
            std_y_augmentation_high_level = None,
            mean_angle_z_augmentation_high_level = None, 
            std_rot_z_augmentation_high_level = None,
            prob_x = None,
            prob_y = None,
            prob_rot_z = None,
            prediction_target='action',
            **kwargs
            ):
        super().__init__()

        self.task_name = task_name
        self.observation_mode = observation_mode
        self.augmentation_rot = augmentation_rot
        self.augmentation_pcd = augmentation_pcd
        self.augmentation_scale = augmentation_scale
        self.scale_scene_by_pcd = scale_scene_by_pcd
        self.use_absolute_waypoint = use_absolute_waypoint
        self.is_pickle = is_pickle
        self.object_augmentation_high_level = object_augmentation_high_level
        self.prediction_target = prediction_target
        
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
            for zarr_path in tqdm(all_zarr_paths):
                all_subfolder = os.listdir(zarr_path)
                # import pdb; pdb.set_trace()
                for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud', '.zgroup']:
                    if string in all_subfolder:
                        all_subfolder.remove(string)
                all_subfolder = sorted(all_subfolder)
                n_episodes = len(all_subfolder)
                num_load_episodes = kwargs.get('num_load_episodes', n_episodes)
                num_load_episodes = min(num_load_episodes, n_episodes)
                all_subfolder = all_subfolder[:num_load_episodes]
                # zarr_paths = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]
                zarr_paths = []
                for subfolder in all_subfolder:
                    if len(os.listdir(os.path.join(zarr_path, subfolder))) > 10:
                        zarr_paths.append(os.path.join(zarr_path, subfolder))
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
                cprint(f'keep in disk and load per step, load_per_step:{self.load_per_step}', 'green')
                from diffusion_policy_3d.common.replay_buffer_disk import ReplayBuffer
                self.replay_buffer = ReplayBuffer.copy_from_multiple_path(all_paths, keys=keys, load_per_step=self.load_per_step, 
                                                                        only_reach_stage=self.only_reach_stage, is_pickle=self.is_pickle,
                                                                        target_action=self.prediction_target)
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

        if self.object_augmentation_high_level:
            print("High level Augmentation Setup")
            trans_x = TranslationXY(mean_x_augmentation_high_level, mean_y_augmentation_high_level, std_x_augmentation_high_level, std_y_augmentation_high_level, True, False)
            trans_y = TranslationXY(mean_x_augmentation_high_level, mean_y_augmentation_high_level, std_x_augmentation_high_level, std_y_augmentation_high_level, False, True)
            rot_z = rotationZ(mean_angle_z_augmentation_high_level, std_rot_z_augmentation_high_level)
            #probs = [0.4, 0.6, 0.3]
            transforms_and_probs = [[trans_x,prob_x], [trans_y, prob_y], [rot_z, prob_rot_z]]
            self.rand_apply = RandomApplyNumpy(transforms_and_probs)
        # [Chialiang]   
        cprint('dataset has been loaded', 'green')
            
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
            normalizer.params_dict[self.prediction_target] = this_params

            # [DebugNormalize] [Chialiang]
            if self.augmentation_rot:
                value = self.action_welford.get_max_norm_3d()
                value = torch.from_numpy(value).float()
                additional_params = torch.nn.ParameterDict({
                    'max_norm_3d': value
                })
                for p in additional_params.values():
                    p.requires_grad = False

                normalizer.params_dict['additional_params'] = additional_params

        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):

        # get data
        agent_pos = copy.deepcopy(sample['state'][:,])
        point_cloud = copy.deepcopy(sample['point_cloud'][:,])
        action = copy.deepcopy(sample['action'])
        if self.object_augmentation_high_level:
            gripper_pcd = copy.deepcopy(sample['gripper_pcd'][:,])
            goal_gripper_pcd = copy.deepcopy(sample['goal_gripper_pcd'][:,])
        agent_pos_old = copy.deepcopy(agent_pos)

        # if 'act3d' in self.observation_mode:
        #     gripper_pcd = copy.deepcopy(sample['gripper_pcd'][:,])
        #     if 'mlp' not in self.observation_mode:

        #         pcd_mask = copy.deepcopy(sample['pcd_mask'][:,])
        #         feature_map = copy.deepcopy(sample['feature_map'][:,])

        #     if 'goal' in self.observation_mode:
        #         goal_gripper_pcd = copy.deepcopy(sample['goal_gripper_pcd'][:,])
            
        #     if 'displacement_gripper_to_object' in self.observation_mode:
        #         displacement_gripper_to_object = copy.deepcopy(sample['displacement_gripper_to_object'][:,])
        
        # elif 'act3d_pointnet' == self.observation_mode:
        #     gripper_pcd = copy.deepcopy(sample['gripper_pcd'][:,])

        # augmentation
        ###########################################
        debug = False
        if debug:
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/agent_pos_before.npy', agent_pos)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/point_cloud_before.npy', point_cloud)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/action_before.npy', action)
            # np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/feature_map_before.npy', feature_map)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/gripper_pcd_before.npy', gripper_pcd)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/goal_gripper_pcd_before.npy', goal_gripper_pcd)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/displacement_gripper_to_object_before.npy', displacement_gripper_to_object)
            start = time.time()
        ###########################################

        if self.augmentation_pcd:
            point_cloud = point_cloud + np.random.normal(0, 0.003, point_cloud.shape) # [AugTODO] add more 

        if self.augmentation_rot:
            # random rotation
            random_trans = np.identity(4)
            random_zrot = (np.random.rand() * 2 - 1) * 10 * np.pi / 180 # -10 degree to 10 degree in raduis
            
            ###########################################
            if debug:
                random_zrot = 45 * np.pi / 180 
            ###########################################

            random_rotmat = p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, random_zrot]))
            random_rotmat = np.asarray(random_rotmat).reshape(3, 3)
            random_trans[:3, :3] = random_rotmat

            # agent pos
            agent_pos_old = copy.deepcopy(agent_pos)
            agent_trans = np.identity(4).repeat(self.horizon, 1)
            pos_index = np.asarray([4*i+3 for i in range(self.horizon)]).astype(np.uint16)
            agent_trans[:3, pos_index] = agent_pos[:, :3].T
            rot_index = np.asarray([[4*i, 4*i+1, 4*i+2] for i in range(self.horizon)]).astype(np.uint16).reshape(-1)
            agent_trans[:3, rot_index] = rotation_transfer_6D_to_matrix_batch(agent_pos[:,3:9]) # should be 6D rotation representation
            agent_trans = random_trans @ agent_trans
            agent_pos[:, :3] = agent_trans[:3, pos_index].T
            agent_pos[:, 3:9] = rotation_transfer_matrix_to_6D_batch(agent_trans[:3, rot_index].T)

            # point cloud
            point_cloud_homo = np.ones((point_cloud.shape[0] * point_cloud.shape[1], 4))
            point_cloud_homo[:,:3] = point_cloud.reshape((-1, 3))
            point_cloud = (point_cloud_homo @ random_trans.T)[:, :3]
            point_cloud = point_cloud.reshape(self.horizon, -1, 3)

            # action
            action[:,:3] = action[:,:3] @ random_rotmat.T

            if 'act3d' in self.observation_mode:

                gripper_pcd_copy = copy.deepcopy(gripper_pcd)
                gripper_pcd_homo = np.ones((gripper_pcd.shape[0] * gripper_pcd.shape[1], 4))
                gripper_pcd_homo[:,:3] = gripper_pcd.reshape((-1, 3))
                gripper_pcd = (gripper_pcd_homo @ random_trans.T)[:, :3]
                gripper_pcd = gripper_pcd.reshape(self.horizon, -1, 3)

                # if 'mlp' not in self.observation_mode:
                #     feature_num = feature_map.shape[0] * feature_map.shape[1]
                #     feature_dim = feature_map.shape[2]
                #     feature_map = feature_map.reshape((-1, feature_dim))
                #     feature_map_homo = np.ones((feature_num, 4))
                #     feature_map_homo[:,:3] = feature_map[:,2:]
                #     feature_map[:,2:] = (feature_map_homo @ random_trans.T)[:, :3]
                #     feature_map = feature_map.reshape(self.horizon, -1, feature_dim)

                if 'goal' in self.observation_mode:
                    goal_gripper_pcd_homo = np.ones((goal_gripper_pcd.shape[0] * goal_gripper_pcd.shape[1], 4))
                    goal_gripper_pcd_homo[:,:3] = goal_gripper_pcd.reshape((-1, 3))
                    goal_gripper_pcd = (goal_gripper_pcd_homo @ random_trans.T)[:, :3]
                    goal_gripper_pcd = goal_gripper_pcd.reshape(self.horizon, -1, 3)
                
                if 'displacement_gripper_to_object' in self.observation_mode:
                    goal_gripper_to_pcd = gripper_pcd_copy + displacement_gripper_to_object
                    goal_gripper_to_pcd_homo = np.ones((goal_gripper_to_pcd.shape[0] * goal_gripper_to_pcd.shape[1], 4))
                    goal_gripper_to_pcd_homo[:,:3] = goal_gripper_to_pcd.reshape((-1, 3))
                    goal_gripper_to_pcd = (goal_gripper_to_pcd_homo @ random_trans.T)[:, :3]
                    goal_gripper_to_pcd = goal_gripper_to_pcd.reshape(self.horizon, -1, 3)
                    displacement_gripper_to_object = goal_gripper_to_pcd - gripper_pcd
            
            elif 'act3d_pointnet' == self.observation_mode:

                gripper_pcd_homo = np.ones((gripper_pcd.shape[0] * gripper_pcd.shape[1], 4))
                gripper_pcd_homo[:,:3] = gripper_pcd.reshape((-1, 3))
                gripper_pcd = (gripper_pcd_homo @ random_trans.T)[:, :3]
                gripper_pcd = gripper_pcd_homo.reshape(self.horizon, -1, 3)

        if self.object_augmentation_high_level:
            #print("APPLYING DATA AUGMENTATION")
            data = {
                'point_cloud': point_cloud.astype(np.float32), # T, 1280, 
                'agent_pos': agent_pos.astype(np.float32), # T, D_pos
                'gripper_pcd' : gripper_pcd.astype(np.float32),
                'goal_gripper_pcd': goal_gripper_pcd.astype(np.float32)
        }
            data = self.rand_apply(data)
            point_cloud = data["point_cloud"]
            gripper_pcd = data["gripper_pcd"]
            goal_gripper_pcd = data["goal_gripper_pcd"]



        # change to absolute waypoints
        if self.use_absolute_waypoint:

            absolute_action = copy.deepcopy(agent_pos)
            
            # position
            absolute_action[:, :3] += action[:, :3]
            
            # rotation
            current_rotations = rotation_transfer_6D_to_matrix_batch(agent_pos[:, 3:9]).T.reshape((-1, 3, 3)) # (H, 6) -> (3, 3*H) -> (3*H, 3) -> (H, 3, 3)
            current_rotations = np.transpose(current_rotations, (0, 2, 1)) # (H, 3, 3) make row vector column vector
            delta_rotations = rotation_transfer_6D_to_matrix_batch(action[:, 3:9]).T.reshape((-1, 3, 3)) # (H, 6) -> (3, 3*H) -> (3*H, 3) -> (H, 3, 3)
            delta_rotations = np.transpose(delta_rotations, (0, 2, 1)) # (H, 3, 3) make row vector column vector
            next_rotations = np.matmul(current_rotations, delta_rotations) # (H, 3, 3)
            row_1 = next_rotations[:, :3, 0] # (H, 3)
            row_2 = next_rotations[:, :3, 1] # (H, 3)
            rot_6d = np.hstack((row_1, row_2)) # (H, 6)
            absolute_action[:, 3:9] = rot_6d

            # eef
            absolute_action[:, 9] += action[:, 9]

            action = absolute_action

        if self.augmentation_scale:

            max_difference = 0.2
            random_scale = 1 + max_difference * (2 * np.random.rand() - 1) # [1 - max_difference, 1 + max_difference]

            point_cloud[...,:3] *= random_scale
            agent_pos[...,:3] *= random_scale
            action[...,:3] *= random_scale
            
            if 'act3d' in self.observation_mode:
                gripper_pcd[...,:3] *= random_scale
                if 'goal' in self.observation_mode:
                    goal_gripper_pcd[...,:3] *= random_scale
                if 'displacement_gripper_to_object' in self.observation_mode:
                    displacement_gripper_to_object[...,:3] *= random_scale
            
            elif 'act3d_pointnet' == self.observation_mode:
                gripper_pcd[...,:3] *= random_scale

        if self.scale_scene_by_pcd:

            max_scale = np.max(np.linalg.norm(point_cloud, axis=-1))

            point_cloud[...,:3] /= max_scale
            agent_pos[...,:3] /= max_scale
            action[...,:3] /= max_scale

            if 'act3d' in self.observation_mode:
                gripper_pcd[...,:3] /= max_scale
                if 'goal' in self.observation_mode:
                    goal_gripper_pcd[...,:3] /= max_scale
                if 'displacement_gripper_to_object' in self.observation_mode:
                    displacement_gripper_to_object[...,:3] /= max_scale
            
            elif 'act3d_pointnet' == self.observation_mode:
                gripper_pcd[...,:3]  /= max_scale
        if self.object_augmentation_high_level:
            data = {
            'obs': {
                'point_cloud': point_cloud.astype(np.float32), # T, 1280, 
                'agent_pos': agent_pos.astype(np.float32), # T, D_pos
                'gripper_pcd': gripper_pcd.astype(np.float32),
                'goal_gripper_pcd': goal_gripper_pcd.astype(np.float32)
                },
                'action': action.astype(np.float32)
            }
            for key in self.keys_:
                if key not in ['state', 'action', 'point_cloud', 'gripper_pcd', 'goal_gripper_pcd']:
                    data['obs'][key] = copy.deepcopy(sample[key][:,].astype(np.float32))
        else:
            # assign to dict
            data = {
                'obs': {
                    'point_cloud': point_cloud.astype(np.float32), # T, 1280, 
                    'agent_pos': agent_pos.astype(np.float32), # T, D_pos
                },
                'action': action.astype(np.float32)
            }

            # if 'act3d' in self.observation_mode:
            #     data['obs']['gripper_pcd'] = gripper_pcd.astype(np.float32)
            #     if 'mlp' not in self.observation_mode:
            #         data['obs']['feature_map'] = feature_map.astype(np.float32)
            #         data['obs']['pcd_mask'] = pcd_mask.astype(np.uint8)
            #     if 'goal' in self.observation_mode:
            #         data['obs']['goal_gripper_pcd'] = goal_gripper_pcd.astype(np.float32)
            #     if 'displacement_gripper_to_object' in self.observation_mode:
            #         data['obs']['displacement_gripper_to_object'] = displacement_gripper_to_object.astype(np.float32)
            for key in self.keys_:
                if key not in ['state', 'action', 'point_cloud']:
                    data['obs'][key] = copy.deepcopy(sample[key][:,].astype(np.float32))
                
        if self.prediction_target == 'delta_to_goal_gripper':
            data['obs']['delta_to_goal_gripper'] = data['obs']['goal_gripper_pcd'] - data['obs']['gripper_pcd']
            
        return data

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
