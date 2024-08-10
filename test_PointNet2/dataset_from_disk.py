import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import zarr
import os
from termcolor import cprint
import numpy as np
from tqdm import tqdm

class PointNetDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, all_obj_paths, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False):
        self.all_obj_paths = all_obj_paths
        self.beg_ratio = beg_ratio
        self.end_ratio = end_ratio
        
        if only_first_stage:
            cprint("======= ONLY FIRST STAGE =======", "red")

        if eval_episode is not None:
            cprint("======= EVAL MODE =======", "red")
            cprint(f"Only evaluating the first observation of {eval_episode} episodes", "red")

        self.all_zarr_paths = []
        for obj_path in all_obj_paths:
            all_subfolder = os.listdir(obj_path)
            for s in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if s in all_subfolder:
                    all_subfolder.remove(s)
            all_subfolder = sorted(all_subfolder)
            beg = int(beg_ratio * len(all_subfolder))
            end = int(end_ratio * len(all_subfolder))
            if eval_episode is not None:
                end = beg + eval_episode
            all_subfolder = all_subfolder[beg:end]
            self.all_zarr_paths += [os.path.join(obj_path, s) for s in all_subfolder]

        cprint("Preparing all zarr paths", "green")
        self.episode_lengths = []
        for idx, zarr_path in enumerate(tqdm(self.all_zarr_paths)):

            all_substeps = os.listdir(zarr_path)
            all_substeps = sorted(all_substeps, key=lambda x: int(x))

            first_goal = None

            for i, substep in enumerate(all_substeps):
                
                if eval_episode is not None and i >=1:
                    self.episode_lengths.append(i)
                    break

                substep_path = os.path.join(zarr_path, substep)
                group = zarr.open(substep_path, 'r')
                src_store = group.store
                src_root = zarr.group(src_store)

                action = src_root['data']['action'][:]

                current_goal = src_root['data']['goal_gripper_pcd'][:]
                if first_goal is None:
                    first_goal = current_goal
                elif only_first_stage and not np.allclose(first_goal, current_goal):
                    self.episode_lengths.append(i)
                    break

            if not only_first_stage and eval_episode is None:
                self.episode_lengths.append(len(all_substeps))

        self.episode_lengths = np.array(self.episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        cprint(f"Finished preparing all zarr paths with total datapoints: {self.accumulated_episode_lengths[-1]}", "green")

    def __len__(self):
        return self.accumulated_episode_lengths[-1]

    def __getitem__(self, idx):
        episode_idx = np.searchsorted(self.accumulated_episode_lengths, idx, side='right')
        start_idx = idx - self.accumulated_episode_lengths[episode_idx]

        if start_idx < 0:
            start_idx += self.episode_lengths[episode_idx]

        zarr_path = self.all_zarr_paths[episode_idx]
        
        step_path = os.path.join(zarr_path, str(start_idx))
        group = zarr.open(step_path, 'r')
        src_store = group.store
        src_root = zarr.group(src_store)
        pointcloud = src_root['data']['point_cloud'][:][0]
        gripper_pcd = src_root['data']['gripper_pcd'][:][0]
        goal_gripper_pcd = src_root['data']['goal_gripper_pcd'][:][0]

        return pointcloud, gripper_pcd, goal_gripper_pcd
        
def get_dataloader(all_obj_paths=None, batch_size=32, beg_ratio=0, end_ratio=0.9, shuffle=True, eval_episode=None, only_first_stage=False):
    if all_obj_paths is None:
        all_obj_paths = ["0705-obj-41510", "0705-obj-45448", "0705-obj-46462", "0705-obj-46732", "0705-obj-46801", "0705-obj-46874", "0705-obj-46922", "0705-obj-46966", "0705-obj-47570", "0705-obj-47578", "0705-obj-48700", "0705-obj-45526", "0705-obj-45661", "0705-obj-45694", "0705-obj-45780", "0705-obj-45910", "0705-obj-45961", "0705-obj-46408", "0705-obj-46417", "0705-obj-46440", "0705-obj-46490", "0705-obj-46762", "0705-obj-46825", "0705-obj-46893", "0705-obj-47235", "0705-obj-47281", "0705-obj-47315", "0705-obj-47529", "0705-obj-47669", "0705-obj-47944", "0705-obj-48063", "0705-obj-48177", "0705-obj-48356", "0705-obj-48623", "0705-obj-48876", "0705-obj-49025", "0705-obj-49062", "0705-obj-49132", "0705-obj-49133", "0712-obj-40417", "0712-obj-41085", "0712-obj-41452", "0712-obj-45162", "0712-obj-45176", "0712-obj-45194", "0712-obj-45203", "0712-obj-45248", "0712-obj-45271", "0712-obj-45290", "0712-obj-45305"]
        all_obj_paths = ["/scratch/chialiang/dp3_demo/" + s for s in all_obj_paths]
    dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        