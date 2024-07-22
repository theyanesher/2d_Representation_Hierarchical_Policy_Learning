from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path
from time import time
import numpy as np

import torch
from torch.utils.data import Dataset
import copy
import os
import zarr
from robogen_utils import get_gripper_pos_orient_from_4_points
from termcolor import cprint
from scipy.spatial.transform import Rotation as R


def rotation_transfer_6D_to_quaternion(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T
    r = R.from_matrix(rotate_matrix)
    quat = r.as_quat()
    return quat



class RobogenDataset(Dataset):
    """Robogen dataset."""

    def __init__(
        self,
        # required
        root,
        # dataset specification
        max_episode_length=5,
        max_episodes_per_task=100,
        num_iters=None,
        num_cameras=2,
        feature_map_channel=5,
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        start_episode_idx=0,
        end_episode_idx=1000,
        kept_in_disk=True,
        load_per_step=True,
        **kwargs,
    ):
        keys = ['state', 'action', 'point_cloud', 'feature_map', 'gripper_pcd', 'goal_gripper_pcd']
        cprint("===== Using Robogen Dataset, only keypose =====", "yellow")
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training

        self.kept_in_disk = kept_in_disk
        self.load_per_step = load_per_step

        cprint("="*50, "yellow")
        cprint("================== Hardcoded dataset paths ==================", "yellow")
        cprint("="*50, "yellow")
        # all_data_paths = [copy.deepcopy(root)]
        all_data_paths = [
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1/",
            os.environ["PROJECT_DIR"]+"/../dp3_demo/0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1/",
            ]
        cprint(f"Loading data from {all_data_paths}", "yellow")
        all_paths = []
        for data_path in all_data_paths:
            all_subfolder = os.listdir(data_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            n_episodes = len(all_subfolder)
            start_episode_idx = max(0, start_episode_idx)
            end_episode_idx = min(n_episodes, end_episode_idx)
            all_subfolder = all_subfolder[start_episode_idx:end_episode_idx]
            zarr_paths = [os.path.join(data_path, subfolder) for subfolder in all_subfolder]
            all_paths.extend(zarr_paths)

        if not self.kept_in_disk:
            self.build_replay_buffer(all_paths, keys)
        else:
            self.build_replay_buffer_in_disk(all_paths, keys)


    def build_replay_buffer_in_disk(self, all_paths, keys=None):
        self.all_path_list = all_paths
        episode_lengths = []
        for idx, path in enumerate(all_paths):
            if not self.load_per_step:
                group = zarr.open(path, mode='r')
                src_store = group.store
                src_root = zarr.group(src_store)
                if keys is None:
                    keys = src_root['data'].keys()
                
                episode_lengths.append(len(src_root['data'][keys[0]][:]))
            else:
                all_substeps = os.listdir(path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x))
                episode_lengths.append(len(all_substeps))

        self.episode_lengths = np.array(episode_lengths)
        self.accumulated_lengths = np.cumsum(self.episode_lengths)

    def __getitem__(self, idx):
        idx %= self.accumulated_lengths[-1]
        episode_idx = np.searchsorted(self.accumulated_lengths, idx, side='right')
        end_idx = idx - self.accumulated_lengths[episode_idx]
        ret_data = self._get_single_step_data(self.all_path_list[episode_idx], end_idx)
        return ret_data


    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self.accumulated_lengths[-1]
        

    def _get_single_step_data(self, episode_data_path, idx):
        try:
            ret_dict = {}
            all_steps = len(os.listdir(episode_data_path))
            idx = idx + all_steps if idx < 0 else idx

            # get data at idx
            step_path = os.path.join(episode_data_path, str(idx))
            # cprint(f"loading step_path: {step_path}", "blue")
            group = zarr.open(step_path, mode='r')
            src_store = group.store
            src_root = zarr.group(src_store)
            ret_dict['task'] = ['robogen']
            ret_dict['rgbs'] = torch.tensor(src_root['data']['feature_map'][:]).permute(0, 1, 4, 2, 3)
            ret_dict['pcds'] = torch.tensor(src_root['data']['feature_map'][:]).permute(0, 1, 4, 2, 3)[:, :, 2:5, ...]
            ret_dict['action'] = torch.tensor(self._get_8D_pose_from_4_point(src_root['data']['goal_gripper_pcd'][:][0])).unsqueeze(0)
            ret_dict['instr'] = torch.zeros((ret_dict['rgbs'].shape[0], 53, 512))
            ret_dict['curr_gripper'] = torch.tensor(self._get_8D_pose_from_state(src_root['data']['state'][:][0])).unsqueeze(0)

            # get history
            gripper_history = []
            for x in range(-2, 1):
                i = idx + x
                i = max(0, i)
                i = min(i, all_steps - 1)
                step_path = os.path.join(episode_data_path, str(i))
                group = zarr.open(step_path, mode='r')
                src_store = group.store
                src_root = zarr.group(src_store)
                gripper_history.append(torch.tensor(self._get_8D_pose_from_state(src_root['data']['state'][:][0])))

            ret_dict['curr_gripper_history'] = torch.stack(gripper_history).unsqueeze(0)
            ret_dict['trajectory'] = ret_dict['action'].unsqueeze(0)
            ret_dict['trajectory_mask'] = torch.zeros(ret_dict['trajectory'].shape[:-1])
        except:
            ret_dict = {
                "task": ['robogen'],
                "rgbs": torch.zeros((1, 2, 5, 256, 256)),
                "pcds": torch.zeros((1, 2, 3, 256, 256)),
                "action": torch.zeros((1, 8)),
                "instr": torch.zeros((1, 53, 512)),
                "curr_gripper": torch.zeros((1, 8)),
                "curr_gripper_history": torch.zeros((1, 3, 8)),
                "trajectory": torch.zeros((1, 1, 8)),
                "trajectory_mask": torch.zeros((1, 1))
            }
            ret_dict['action'][0, 6] = 1
            ret_dict['action'][0, 7] = 1
            ret_dict['curr_gripper'][0, 6] = 1
            ret_dict['curr_gripper'][0, 7] = 1
            ret_dict['curr_gripper_history'][:, :, 6] = 1
            ret_dict['curr_gripper_history'][:, :, 7] = 1
            ret_dict['trajectory'][:, :, 6] = 1
            ret_dict['trajectory'][:, :, 7] = 1 

        # for key, value in ret_dict.items():
        #     if key != 'task':
        #         cprint(f"{key}: {value.shape}", "red")
        return ret_dict



    def _get_8D_pose_from_4_point(self, gripper_pcd):
        # print("trying to get 8D pose with gripper_pcd: ", gripper_pcd)
        pos, orn = get_gripper_pos_orient_from_4_points(gripper_pcd)
        openness = np.array([1])
        return np.concatenate([pos, orn, openness])
    
    def _get_8D_pose_from_state(self, state):
        pos = state[:3]
        orn = rotation_transfer_6D_to_quaternion(state[3:9])
        openness = [state[9]]
        return np.concatenate([pos, orn, openness])
    
if __name__ == '__main__':
    dataset = RobogenDataset(root="/scratch/yufei/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point")
    for i in range(10):
        data = dataset[i]
        import pdb; pdb.set_trace()





        


