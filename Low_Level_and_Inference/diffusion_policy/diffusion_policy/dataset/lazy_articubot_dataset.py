import copy
import numpy as np
import torch
from typing import Dict
from pathlib import Path

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.lazy_replay_buffer import LazyH5Buffer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_welford_image_normalizer,
    get_welford_robot_state_normalizer,
    array_to_stats,
    get_welford_online_action_normalizer,
    get_robot_state_normalizer_from_stat,
    get_online_range_robot_state_normalizer,
    get_identity_normalizer,
)
from diffusion_policy.common.action_util import (
    relative_to_delta_actions,
    delta_to_relative_actions,
    hybrid_relative_to_hybrid_delta_actions,
    hybrid_delta_to_hybrid_relative_actions,
)
from common.data_utils import process_pointmap, process_plucker, process_depth
from diffusion_policy.common.heatmap_augmentation import HeatmapRotationAugmentation


class LazyArticuBotDataset(BaseImageDataset):
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
            pointmap_frame='robot_frame',
            ghost_heatmap_last_channel_only=False,
            add_current_heatmap=False,
            heatmap_augmentation=None,
        ):
        super().__init__()
        self.ghost_heatmap_last_channel_only = ghost_heatmap_last_channel_only
        self.add_current_heatmap = add_current_heatmap

        # Build augmenters — one per heatmap type with the correct border fill.
        # border_fill=0.0 for Gaussian (no signal = keypoint not here)
        # border_fill=1.0 for ghost/distance-field (max distance = keypoint not here)
        aug_cfg = heatmap_augmentation or {}
        self.gaussian_heatmap_aug = HeatmapRotationAugmentation(
            border_fill=0.0, **aug_cfg
        ) if aug_cfg else None
        self.ghost_heatmap_aug = HeatmapRotationAugmentation(
            border_fill=1.0, **aug_cfg
        ) if aug_cfg else None

        self.shape_meta = shape_meta
        self.data_dir = Path(data_dir)
        self.action_mode = action_mode
        self.pointmap_frame = pointmap_frame

        if self.action_mode not in ['hybrid_delta', 'hybrid_relative', 'delta', 'relative']:
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        if self.pointmap_frame not in ['robot_frame', 'gripper_frame']:
            raise ValueError(f"Unsupported pointmap_frame: {pointmap_frame}")

        h5_paths = sorted(self.data_dir.glob("*.h5"))
        if max_train_episodes is not None:
            h5_paths = h5_paths[:max_train_episodes]

        # -----------------------------------------------------------------------
        # 1. Categorise observation keys by modality
        # -----------------------------------------------------------------------
        self.rgb_keys = []
        self.depth_keys = []
        self.heatmap_keys = []
        self.ghost_heatmap_keys = []
        self.pointmap_keys = []
        self.plucker_keys = []
        self.lowdim_keys = []
        self.matrix_keys = []
        self.goal_gripper_keys = []
        self.gmm_keys = []  # gmm_goals and gmm_weights — loaded as-is, identity normalizer

        for key, attr in shape_meta['obs'].items():
            type_name = attr.get('type')
            if type_name == 'rgb':
                self.rgb_keys.append(key)
            elif type_name == 'depth':
                self.depth_keys.append(key)
            elif type_name == 'pointmap':
                self.pointmap_keys.append(key)
            elif type_name == 'plucker':
                self.plucker_keys.append(key)
            elif type_name == 'heatmap':
                self.heatmap_keys.append(key)
            elif type_name == 'ghost_heatmap':
                self.ghost_heatmap_keys.append(key)
            elif type_name == 'goal_gripper':
                self.goal_gripper_keys.append(key)
            elif type_name in ('gmm_goals', 'gmm_weights'):
                self.gmm_keys.append(key)
            elif type_name == 'low_dim':
                self.lowdim_keys.append(key)
            elif 'intrinsic' in key or 'extrinsic' in key:
                self.matrix_keys.append(key)
            else:
                self.lowdim_keys.append(key)

        all_obs_keys = (
            self.rgb_keys + self.depth_keys + self.heatmap_keys +
            self.ghost_heatmap_keys + self.pointmap_keys + self.plucker_keys +
            self.lowdim_keys + self.matrix_keys + self.goal_gripper_keys +
            self.gmm_keys
        )
        if self.pointmap_frame == 'gripper_frame':
            all_obs_keys.append('gripper_to_world')
        if add_current_heatmap:
            for key in self.ghost_heatmap_keys:
                cam_prefix = key.split('_heatmap')[0]
                all_obs_keys.append(f'{cam_prefix}_present_heatmap_ghost')
            for key in self.heatmap_keys:
                cam_prefix = key.split('_heatmap')[0]
                all_obs_keys.append(f'{cam_prefix}_present_heatmap')

        # -----------------------------------------------------------------------
        # 2. Build LazyH5Buffer — reads only metadata (shapes/dtypes/lengths),
        #    no array data loaded into memory.
        # -----------------------------------------------------------------------
        self.action_key = 'action/hybrid' if self.action_mode.startswith('hybrid') else 'action/delta' 
        
        key_to_h5path = {'action': self.action_key}
        for key in all_obs_keys:
            key_to_h5path[key] = f'obs/{key}'

        print(f"Indexing {len(h5_paths)} trajectories from {data_dir} (lazy)...")
        self.replay_buffer = LazyH5Buffer(
            h5_paths=h5_paths,
            key_to_h5path=key_to_h5path,
        )

        # -----------------------------------------------------------------------
        # 3. Setup sampler — identical to the eager dataset
        # -----------------------------------------------------------------------
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask

        key_first_k = {}
        if n_obs_steps is not None:
            for key in all_obs_keys:
                key_first_k[key] = n_obs_steps
        # import pdb; pdb.set_trace()
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        print(f"Dataset initialized with {len(self.sampler)} training sequences.")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        # Disable heatmap augmentation for validation
        if val_set.gaussian_heatmap_aug is not None:
            val_set.gaussian_heatmap_aug = copy.copy(val_set.gaussian_heatmap_aug)
            val_set.gaussian_heatmap_aug.enabled = False
        if val_set.ghost_heatmap_aug is not None:
            val_set.ghost_heatmap_aug = copy.copy(val_set.ghost_heatmap_aug)
            val_set.ghost_heatmap_aug.enabled = False
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Action
        # [:] reads from disk one file at a time and concatenates — low memory
        # footprint for low-dim data; fine for action/state arrays.
        if 'relative' in self.action_mode:
            normalizer['action'] = get_welford_online_action_normalizer(
                horizon=self.horizon,
                action_dim=self.shape_meta['action']['shape'][0],
                max_samples=len(self.sampler) * self.n_obs_steps,
            )
        elif 'delta' in self.action_mode:
            # stat = array_to_stats(self.replay_buffer['action'][:])
            # normalizer['action'] = get_robot_state_normalizer_from_stat(stat)
            action_dim = self.shape_meta['action']['shape'][0]
            normalizer['action'] = get_online_range_robot_state_normalizer(
                state_dim=action_dim,
                max_samples=len(self.sampler) * self.n_obs_steps,
            )

        # Low dim
        for key in self.lowdim_keys:
            if 'state' in key or 'qpos' in key:
                state_dim = self.shape_meta['obs'][key]['shape'][0]
                normalizer[key] = get_online_range_robot_state_normalizer(
                    state_dim=state_dim,
                    max_samples=len(self.sampler) * self.n_obs_steps,
                )
            else:
                stat = array_to_stats(self.replay_buffer[key][:])
                normalizer[key] = get_range_normalizer_from_stat(stat)

        # RGB — fixed normalizer, no data needed
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        # Depth — online Welford normalizer, no data loaded upfront.
        # Stats are accumulated per-channel (1 channel) during the first
        # max_samples training steps.
        for key in self.depth_keys:
            normalizer[key] = get_welford_image_normalizer(
                num_channels=1,
                max_samples=len(self.sampler) * self.n_obs_steps,
            )

        for key in self.heatmap_keys:
            normalizer[key] = get_identity_normalizer()

        for key in self.ghost_heatmap_keys:
            normalizer[key] = get_identity_normalizer()

        for key in self.goal_gripper_keys:
            normalizer[key] = get_identity_normalizer()

        # GMM distribution keys — identity (probabilities and 3D coords, no normalization)
        for key in self.gmm_keys:
            normalizer[key] = get_identity_normalizer()

        # Plucker / Pointmap / Matrices — identity (geometric data)
        for key in self.pointmap_keys + self.plucker_keys + self.matrix_keys:
            normalizer[key] = get_identity_normalizer()

        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = {}

        # RGB: (T, H, W, C) -> (T, C, H, W), float32 [0, 1]
        # import pdb; pdb.set_trace()
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
            del data[key]

        # Depth: (T, H, W) -> decompress -> (T, 1, H, W)
        for key in self.depth_keys:
            d_float = process_depth(data[key][T_slice], compress=False)
            if d_float.ndim == 3:
                d_float = d_float[:, None, :, :]
            obs_dict[key] = d_float
            del data[key]

        for key in self.heatmap_keys:
            raw = data[key][T_slice]
            if raw.dtype == np.uint8:
                # New dataset: (T, H, W, C) uint8 -> (T, C, H, W) float32 [0, 1]
                h_float = np.moveaxis(raw.astype(np.float32) / 255., -1, 1)
            else:
                # Old dataset: (T, H, W) float16, already [0, 1]
                h_float = raw.astype(np.float32)
                if h_float.ndim == 3:
                    h_float = h_float[:, None, :, :]         # (T, 1, H, W)
            if self.gaussian_heatmap_aug is not None:
                h_float = self.gaussian_heatmap_aug(h_float)
            obs_dict[key] = h_float
            del data[key]

        # Ghost heatmap: (T, H, W, 4) uint8 on disk -> (T, 4, H, W) float32 [0, 1]
        for key in self.ghost_heatmap_keys:
            g = data[key][T_slice].astype(np.float32) / 255.  # (T, H, W, 4)
            if self.ghost_heatmap_last_channel_only:
                g = g[..., -1:]                                # (T, H, W, 1)
            goal = np.moveaxis(g, -1, 1)                       # (T, 4or1, H, W)
            if self.ghost_heatmap_aug is not None:
                goal = self.ghost_heatmap_aug(goal)
            if self.add_current_heatmap:
                # key e.g. "cam0_heatmap_ghost" -> present key "cam0_present_heatmap_ghost"
                cam_prefix = key.split('_heatmap')[0]          # e.g. "cam0"
                present_key = f"{cam_prefix}_present_heatmap_ghost"
                p = data[present_key][T_slice].astype(np.float32) / 255.
                present = np.moveaxis(p, -1, 1)                # (T, 4, H, W)
                obs_dict[key] = np.concatenate([goal, present], axis=1)  # (T, 8, H, W)
            else:
                obs_dict[key] = goal
            del data[key]

        # Gaussian heatmap: append current gripper heatmap channels if requested
        if self.add_current_heatmap:
            for key in self.heatmap_keys:
                cam_prefix = key.split('_heatmap')[0]          # e.g. "cam0"
                present_key = f"{cam_prefix}_present_heatmap"
                p = data[present_key][T_slice].astype(np.float32) / 255.
                present = np.moveaxis(p, -1, 1)                # (T, 4, H, W)
                obs_dict[key] = np.concatenate([obs_dict[key], present], axis=1)  # (T, 8, H, W)


        # Pointmap: (T, 3, H, W) -> decompress
        for key in self.pointmap_keys:
            obs_dict[key] = process_pointmap(data[key][T_slice], compress=False)
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

        # Plucker: (T, 6, H, W) -> decompress
        for key in self.plucker_keys:
            obs_dict[key] = process_plucker(data[key][T_slice], compress=False)
            del data[key]

        # Low dim & matrices
        for key in self.lowdim_keys + self.matrix_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # Goal gripper pts: (T, 4, 3) float32 — loaded as-is, no reshape
        for key in self.goal_gripper_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # GMM distribution: loaded as-is (no reshape, no normalization)
        # gmm_all_goals:   (T, 2044, 4, 3) float32
        # gmm_all_weights: (T, 2044)       float32
        for key in self.gmm_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # Actions
        # 6. Actions
        if self.action_mode == 'hybrid_relative':
            relative_action = hybrid_delta_to_hybrid_relative_actions(data['action'].astype(np.float32))
            action = torch.from_numpy(relative_action)
        elif self.action_mode == 'relative':
            relative_action = delta_to_relative_actions(data['action'].astype(np.float32))
            action = torch.from_numpy(relative_action)
        elif 'delta' in self.action_mode:
            action = torch.from_numpy(data['action'].astype(np.float32))

        # import pdb; pdb.set_trace()
        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': action,
        }
