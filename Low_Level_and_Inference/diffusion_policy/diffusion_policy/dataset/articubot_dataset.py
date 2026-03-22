import os
import h5py
import numpy as np
import torch
import copy
from typing import Dict
from pathlib import Path
from tqdm import tqdm

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_depth_scalar_normalizer,
    array_to_stats,
    get_welford_online_action_normalizer,
    get_robot_state_normalizer_from_stat,
    get_identity_normalizer
)
from diffusion_policy.common.action_util import (
    hybrid_delta_to_hybrid_relative_actions,
    delta_to_relative_actions,
)

from common.data_utils import process_pointmap, process_plucker, process_depth

class ArticuBotDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            data_dir: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            action_mode='hybrid_delta',
            pointmap_frame='robot_frame'
        ):
        super().__init__()

        self.shape_meta = shape_meta
        self.data_dir = Path(data_dir)
        self.h5_paths = sorted(list(self.data_dir.glob("*.h5")))
        self.action_mode = action_mode
        self.pointmap_frame = pointmap_frame

        if self.action_mode not in ['hybrid_delta', 'hybrid_relative', 'delta', 'relative']:
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        if self.pointmap_frame not in ['robot_frame', 'gripper_frame']:
            raise ValueError(f"Unsupported pointmap_frame: {pointmap_frame}")
        # if self.pointmap_frame == 'gripper_frame':
        #     raise ValueError("Gripper frame pointmaps are currently not supported in this dataset implementation.")

        self.action_key = 'action/hybrid' if self.action_mode.startswith('hybrid') else 'action/delta' 

        if max_train_episodes is not None:
            self.h5_paths = self.h5_paths[:max_train_episodes]

        # -------------------------------------------------------------------------
        # 1. Dynamic Key Categorization
        # -------------------------------------------------------------------------
        self.rgb_keys = []
        self.depth_keys = []
        self.pointmap_keys = []
        self.plucker_keys = []
        self.lowdim_keys = []
        self.matrix_keys = []
        self.mast3r_keys = []
        self.heatmap_keys = []

        for key, attr in shape_meta['obs'].items():
            type_name = attr.get('type')
            
            if type_name == 'rgb':
                self.rgb_keys.append(key)
            elif type_name == 'depth':
                self.depth_keys.append(key)
            elif type_name == 'pointmap':
                self.pointmap_keys.append(key)
            elif type_name == 'heatmap':
                self.heatmap_keys.append(key)
            elif type_name == 'plucker':
                self.plucker_keys.append(key)
            elif type_name == 'low_dim':
                self.lowdim_keys.append(key)
            elif 'intrinsic' in key or 'extrinsic' in key:
                self.matrix_keys.append(key)
            elif 'mast3r' in key:
                self.mast3r_keys.append(key)
            else:
                # Default fallback
                self.lowdim_keys.append(key)

        # -------------------------------------------------------------------------
        # 2. Build ReplayBuffer (Keep data compressed in RAM)
        # -------------------------------------------------------------------------
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        
        print(f"Loading {len(self.h5_paths)} trajectories from {data_dir}...")
        for path in tqdm(self.h5_paths):
            with h5py.File(path, 'r') as f:
                traj_data = {
                    'action': f[self.action_key][:].astype(np.float32)
                }
                
                all_obs_keys = self.rgb_keys + self.depth_keys + self.heatmap_keys + \
                               self.pointmap_keys + self.plucker_keys + \
                               self.lowdim_keys + self.matrix_keys + \
                               self.mast3r_keys
                
                for key in all_obs_keys:
                    h5_path = f"obs/{key}"
                    if h5_path in f:
                        data = f[h5_path][:]
                        
                        # --- Load Raw Data (Do NOT decompress here) ---
                        if key in self.rgb_keys:
                            # Keep uint8
                            pass 
                        elif key in self.depth_keys:
                            # Keep uint16 (Compressed)
                            pass
                        elif key in self.pointmap_keys or key in self.plucker_keys:
                            # Keep int16 (Compressed)
                            pass
                        else:
                            # Lowdim/Matrix: Ensure float32
                            data = data.astype(np.float32)

                        traj_data[key] = data
                    else:
                        print(f"Warning: Key {h5_path} missing in {path}")
                
                if self.pointmap_frame == 'gripper_frame':
                    traj_data['gripper_to_world'] = f['obs/gripper_to_world'][:] # (T, 4, 4)

                self.replay_buffer.add_episode(traj_data)

        # -------------------------------------------------------------------------
        # 3. Setup Sampler
        # -------------------------------------------------------------------------
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in all_obs_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Action
        stat = array_to_stats(self.replay_buffer['action'])
        if 'relative' in self.action_mode:
            normalizer['action'] = get_welford_online_action_normalizer(
                horizon=self.horizon,
                action_dim=self.shape_meta['action']['shape'][0],
                max_samples=len(self.sampler) * self.n_obs_steps,
            )
        elif 'delta' in self.action_mode:
            normalizer['action'] = get_robot_state_normalizer_from_stat(stat)

        # Low Dim
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if 'state' in key or 'qpos' in key:
                normalizer[key] = get_robot_state_normalizer_from_stat(stat)
            else:
                normalizer[key] = get_range_normalizer_from_stat(stat)

        # RGB
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        # Depth (Decompress temporarily for stats)
        for key in self.depth_keys:
            # We must decompress the buffer to calculate correct statistics
            # Since buffer is uint16, raw stats would be 0-1600 instead of 0-1.6m
            raw_data = self.replay_buffer[key][:] 
            decompressed_data = process_depth(raw_data, compress=False)
            normalizer[key] = get_depth_scalar_normalizer(decompressed_data)
        
        for key in self.heatmap_keys:
            normalizer[key] = get_identity_normalizer()
        
        # Plucker / Pointmap (Identity)
        # Even though they are compressed in RAM, we assume the network inputs 
        # (after getitem) are metric/geometric, so we use IdentityNormalizer.
        for key in self.pointmap_keys + self.plucker_keys + self.matrix_keys:
            normalizer[key] = get_identity_normalizer()
        # Similarly for mast3r outs
        for key in self.mast3r_keys:
            normalizer[key] = get_identity_normalizer()

        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()

        # 1. RGB: (T, H, W, C) -> (T, C, H, W) -> float
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
            del data[key]

        # 2. Depth: (T, H, W) -> Decompress -> Unsqueeze -> (T, 1, H, W)
        for key in self.depth_keys:
            d_compressed = data[key][T_slice]
            d_float = process_depth(d_compressed, compress=False) # (T, H, W)
            if d_float.ndim == 3:
                d_float = d_float[:, None, :, :] # Add channel dim
            obs_dict[key] = d_float
            del data[key]

        for key in self.heatmap_keys:
            h_float = data[key][T_slice].astype(np.float32)  # (T, H, W) float16 on disk -> float32
            if h_float.ndim == 3:
                h_float = h_float[:, None, :, :]             # (T, 1, H, W)
            obs_dict[key] = h_float
            del data[key]

        # 3. Pointmaps: (T, 3, H, W) -> Decompress
        for key in self.pointmap_keys:
            pm_compressed = data[key][T_slice]
            obs_dict[key] = process_pointmap(pm_compressed, compress=False)
            if self.pointmap_frame == 'gripper_frame':
                gripper_to_world = data['gripper_to_world'][T_slice] # (T, 4, 4)
                world_to_gripper = np.linalg.inv(gripper_to_world)  # (T, 4, 4)
                R = world_to_gripper[:, :3, :3]  # (T, 3, 3)
                t = world_to_gripper[:, :3, 3]   # (T, 3)
                T, _, H, W = obs_dict[key].shape
                pts = obs_dict[key].reshape(T, 3, -1)          # (T, 3, H*W)
                mask = np.all(pts == 0, axis=1, keepdims=True)  # (T, 1, H*W)
                transformed = (R @ pts + t[:, :, None])
                obs_dict[key] = np.where(mask, 0.0, transformed).reshape(T, 3, H, W)
            del data[key]
        if self.pointmap_frame == 'gripper_frame':
            del data['gripper_to_world']

        # 4. Plucker: (T, 6, H, W) -> Decompress
        for key in self.plucker_keys:
            pl_compressed = data[key][T_slice]
            obs_dict[key] = process_plucker(pl_compressed, compress=False)
            del data[key]

        # 5. Low Dim & Matrices
        for key in self.lowdim_keys + self.matrix_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # 5b. Mast3r matches
        for key in self.mast3r_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # 6. Actions
        if self.action_mode == 'hybrid_relative':
            relative_action = hybrid_delta_to_hybrid_relative_actions(data['action'].astype(np.float32))
            action = torch.from_numpy(relative_action)
        elif self.action_mode == 'relative':
            relative_action = delta_to_relative_actions(data['action'].astype(np.float32))
            action = torch.from_numpy(relative_action)
        elif 'delta' in self.action_mode:
            action = torch.from_numpy(data['action'].astype(np.float32))

        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': action
        }
