# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
'''
Data loader for training M2T2.
'''
from PIL import Image
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import torch
from termcolor import cprint
from tqdm import tqdm

from m2t2.dataset_utils import (
    depth_to_xyz, jitter_gaussian, normalize_rgb, sample_points
)
from m2t2.all_data import *


def load_rgb_xyz(
    data_dir, robot_prob, world_coord, jitter_scale, grid_res, surface_range=0
):
    with open(f'{data_dir}/meta_data.pkl', 'rb') as f:
        meta_data = pickle.load(f)
    rgb = normalize_rgb(Image.open(f'{data_dir}/rgb.png')).permute(1, 2, 0)
    depth = np.load(f'{data_dir}/depth.npy')
    xyz = torch.from_numpy(
        depth_to_xyz(depth, meta_data['intrinsics'])
    ).float()
    seg = torch.from_numpy(np.array(Image.open(f'{data_dir}/seg.png')))
    label_map = meta_data['label_map']

    if torch.rand(()) > robot_prob:
        robot_mask = seg == label_map['robot']
        if 'robot_table' in label_map:
            robot_mask |= seg == label_map['robot_table']
        if 'object_label' in meta_data:
            robot_mask |= seg == label_map[meta_data['object_label']]
        depth[robot_mask] = 0
        seg[robot_mask] = 0
    xyz, rgb, seg = xyz[depth > 0], rgb[depth > 0], seg[depth > 0]
    cam_pose = torch.from_numpy(meta_data['camera_pose']).float()
    xyz_world = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]

    if 'scene_bounds' in meta_data:
        bounds = meta_data['scene_bounds']
        within = (xyz_world[:, 0] > bounds[0]) & (xyz_world[:, 0] < bounds[3]) \
            & (xyz_world[:, 1] > bounds[1]) & (xyz_world[:, 1] < bounds[4]) \
            & (xyz_world[:, 2] > bounds[2]) & (xyz_world[:, 2] < bounds[5])
        xyz_world, rgb, seg = xyz_world[within], rgb[within], seg[within]
        # Set z-coordinate of all points near table to 0
        xyz_world[np.abs(xyz_world[:, 2]) < surface_range, 2] = 0
        if not world_coord:
            world2cam = cam_pose.inverse()
            xyz = xyz_world @ world2cam[:3, :3].T + world2cam[:3, 3]
    if world_coord:
        xyz = xyz_world

    if jitter_scale > 0:
        table_mask = seg == label_map['table']
        if 'robot_table' in label_map:
            table_mask |= seg == label_map['robot_table']
        xyz[table_mask] = jitter_gaussian(
            xyz[table_mask], jitter_scale, jitter_scale
        )

    outputs = {
        'inputs': torch.cat([xyz - xyz.mean(dim=0), rgb], dim=1),
        'points': xyz,
        'seg': seg,
        'cam_pose': cam_pose
    }

    if 'object_label' in meta_data:
        obj_mask = seg == label_map[meta_data['object_label']]
        obj_xyz, obj_rgb = xyz_world[obj_mask], rgb[obj_mask]
        obj_xyz_grid = torch.unique(
            (obj_xyz[:, :2] / grid_res).round(), dim=0
        ) * grid_res
        bottom_center = obj_xyz.min(dim=0)[0]
        bottom_center[:2] = obj_xyz_grid.mean(dim=0)

        ee_pose = torch.from_numpy(meta_data['ee_pose']).float()
        inv_ee_pose = ee_pose.inverse()
        obj_xyz = obj_xyz @ inv_ee_pose[:3, :3].T + inv_ee_pose[:3, 3]
        outputs.update({
            'object_inputs': torch.cat([
                obj_xyz - obj_xyz.mean(dim=0), obj_rgb
            ], dim=1),
            'ee_pose': ee_pose,
            'bottom_center': bottom_center,
            'object_center': obj_xyz.mean(dim=0)
        })
    else:
        outputs.update({
            'object_inputs': torch.rand(1024, 6),
            'ee_pose': torch.eye(4),
            'bottom_center': torch.zeros(3),
            'object_center': torch.zeros(3)
        })
    return outputs, meta_data


def load_grasps(
    data, pts, seg, label_map, world2cam, contact_radius, offset_bins
):
    grasping_masks, matched_grasps = [], []
    contact_dirs = torch.zeros_like(pts)
    approach_dirs = torch.zeros_like(pts)
    offsets = torch.zeros_like(pts[:, 0])
    names = sorted(list(data['grasps'].keys()))
    for name in names:
        contacts = torch.from_numpy(data['grasp_contacts'][name]).float()
        grasps = torch.from_numpy(data['grasps'][name]).float()
        if world2cam is not None:
            # convert contacts and grasps to camera coordinate
            contacts = contacts @ world2cam[:3, :3].T + world2cam[:3, 3]
            grasps = world2cam @ grasps

        contact_dir = contacts[:, 1] - contacts[:, 0]
        offset = contact_dir.norm(dim=1)
        contact_dir = contact_dir / offset.unsqueeze(1)
        approach_dir = grasps[:, :3, 2]

        # Mx2x3 -> 2Mx3
        contacts = contacts.transpose(0, 1).reshape(-1, 3)
        contact_dir = torch.cat([contact_dir, -contact_dir])
        approach_dir = torch.cat([approach_dir, approach_dir])
        offset = torch.cat([offset, offset])
        grasps = torch.cat([grasps, grasps])

        mask = seg == label_map[name]
        if mask.sum() == 0:
            continue
        tree = KDTree(contacts.numpy())
        dist, idx = tree.query(pts[mask].numpy())
        matched = dist < contact_radius
        idx = idx[matched]
        grasps = grasps[idx]

        if matched.sum() > 0:
            pt_i = torch.where(mask)[0]
            contact_mask = torch.zeros_like(mask)
            contact_mask[pt_i[matched[:, 0]]] = 1
            contact_dirs[contact_mask] = contact_dir[idx]
            approach_dirs[contact_mask] = approach_dir[idx]
            offsets[contact_mask] = offset[idx]
            grasping_masks.append(contact_mask)
            matched_grasps.append(grasps)

    if len(grasping_masks) > 0:
        grasping_masks = torch.stack(grasping_masks).float()
    else:
        # No grasp, skip
        return {'invalid': True}
    contact_any_obj = grasping_masks.any(dim=0)
    contact_dirs = contact_dirs[contact_any_obj]
    approach_dirs = approach_dirs[contact_any_obj]
    offsets = offsets[contact_any_obj]
    outputs = {
        'names': names,
        'grasping_masks': grasping_masks,
        'contact_dirs': contact_dirs,
        'approach_dirs': approach_dirs,
        'grasps': matched_grasps
    }
    labels = torch.bucketize(offsets, torch.tensor(offset_bins)) - 1
    outputs['offsets'] = torch.clip(labels, 0, len(offset_bins) - 1)
    return outputs


def load_placements(
    data, pts, seg, meta_data, cam_pose, num_rotations, contact_radius
):
    place_pos = data['placements'][...]
    place_pos = np.concatenate([
        place_pos, np.full((place_pos.shape[0], 1), 0)
    ], axis=1)
    tree = KDTree(place_pos)
    if cam_pose is not None:
        pts = pts @ cam_pose[:3, :3].T + cam_pose[:3, 3]
    dist, idx = tree.query(pts.numpy())
    matched = dist < contact_radius
    indices = idx[matched]
    if len(indices) == 0:
        # No placement, skip
        return {'invalid': True}
    placement_region = torch.from_numpy(matched[:, 0])
    placement_masks = torch.zeros(pts.shape[0], num_rotations)
    success = data['placement_success'][...]
    skip = success.shape[1] // num_rotations
    success = success[:, ::skip]
    placement_masks[placement_region] = torch.from_numpy(
        success[indices]
    ).float()

    # Mark robot as non-placable region
    label_map = meta_data['label_map']
    object_label = meta_data['object_label']
    not_placable = (seg == label_map['robot']) \
                 | (seg == label_map[object_label])
    placement_region[not_placable] = 1

    outputs = {
        'placement_masks': placement_masks.T,
        'placement_region': placement_region.float()
    }
    return outputs


class PickPlaceDataset(Dataset):
    def __init__(
        self, root_dir, num_points, num_obj_points, world_coord,
        num_rotations, grid_res, jitter_scale, contact_radius,
        offset_bins, robot_prob
    ):
        self.root_dir = root_dir
        self.scenes = sorted(os.listdir(root_dir))
        self.num_points = num_points
        self.num_obj_points = num_obj_points
        self.world_coord = world_coord
        self.num_rotations = num_rotations
        self.grid_res = grid_res
        self.jitter_scale = jitter_scale
        self.robot_prob = robot_prob
        self.contact_radius = contact_radius
        self.offset_bins = offset_bins

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['num_points'] = cfg.num_points
        args['num_obj_points'] = cfg.num_object_points
        args['world_coord'] = cfg.world_coord
        args['num_rotations'] = cfg.num_rotations
        args['grid_res'] = cfg.grid_resolution
        args['jitter_scale'] = cfg.jitter_scale
        args['contact_radius'] = cfg.contact_radius
        args['offset_bins'] = cfg.offset_bins
        args['robot_prob'] = cfg.robot_prob
        return args

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        data_dir = f"{self.root_dir}/{self.scenes[idx]}"
        outputs, meta_data = load_rgb_xyz(
            data_dir, self.robot_prob, self.world_coord,
            self.jitter_scale, self.grid_res
        )
        pt_idx = sample_points(outputs['points'], self.num_points)
        outputs['inputs'] = outputs['inputs'][pt_idx]
        outputs['points'] = outputs['points'][pt_idx]
        outputs['seg'] = outputs['seg'][pt_idx]
        pt_idx = sample_points(outputs['object_inputs'], self.num_obj_points)
        outputs['object_inputs'] = outputs['object_inputs'][pt_idx]
        outputs['scene'] = self.scenes[idx]
        if 'object_label' in meta_data:
            outputs['task'] = 'place'
        else:
            outputs['task'] = 'pick'
        cam_pose = None if self.world_coord else outputs['cam_pose']
        world2cam = None if self.world_coord else outputs['cam_pose'].inverse()

        with open(f"{data_dir}/annotation.pkl", 'rb') as f:
            annotation = pickle.load(f)
        if outputs['task'] == 'pick':
            outputs.update(load_grasps(
                annotation, outputs['points'], outputs['seg'],
                meta_data['label_map'], world2cam,
                self.contact_radius, self.offset_bins
            ))
        else:
            outputs.update({
                'names': [],
                'grasping_masks': torch.zeros(0, self.num_points),
                'contact_dirs': torch.zeros(0, 3),
                'approach_dirs': torch.zeros(0, 3),
                'offsets': torch.zeros(0).long(),
                'grasps': []
            })
        if outputs['task'] == 'place':
            outputs.update(load_placements(
                annotation, outputs['points'], outputs['seg'], meta_data,
                cam_pose, self.num_rotations, self.contact_radius
            ))
        else:
            outputs.update({
                'placement_masks': torch.zeros(
                    self.num_rotations, self.num_points
                ),
                'placement_region': torch.zeros(self.num_points)
            })
        return outputs
    
class FakeArticubotDataset(Dataset):
    def __init__(self):
        self.input_pcd = torch.rand(100, 4500, 3)  # BxNx3 (concatenated scene and gripper points)
        self.goal_gripper_pcd = torch.rand(100, 1, 12)  # Bx1x12 (goal gripper points)
        
    def __len__(self):
        return len(self.input_pcd)
    
    def __getitem__(self, idx):
        return {
            'inputs': self.input_pcd[idx],
            'goal_gripper_pcd': self.goal_gripper_pcd[idx]
        }
        
class ArticuBotDataset(Dataset):
    def __init__(self, all_obj_paths, beg_ratio=0, end_ratio=0.9, use_all_data=False):
        self.all_obj_paths = all_obj_paths
        self.beg_ratio = beg_ratio
        self.end_ratio = end_ratio
        num_cats = 12
        categories = ['bucket', 'faucet', 'foldingchair', 'laptop', 'stapler', 'toilet']
        self.cat_counts = np.zeros(num_cats)  # +1 for the background category
        self.use_all_data = use_all_data

        # TODO for conditioning
        # for each trajectory of the object, record the grasping pose and opening pose. Store as a dictionary maybe, key is object_traj-id

        self.all_zarr_paths = []
        self.all_zarr_categories = []
        self.episode_idx_to_obj_id = {}
        self.obj_id_to_all_episodes_indices = {}
        episode_idx = 0
        for obj_path in all_obj_paths:
            all_subfolder = os.listdir(obj_path)
            cat_idx = 0
            for i, cat in enumerate(categories):
                if cat in obj_path:
                    cat_idx = i + 1
                    break
            if 'invert' in obj_path:
                if cat_idx == 0:
                    cat_idx = 7
                else:
                    cat_idx += 5
                    
            # storage furniture, bucket, faucet, foldingchair, laptop, stapler, toilet, invert storage furniture, invert foldingchair, invert laptop, invert stapler, invert toilet
            for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
                if s in all_subfolder:
                    all_subfolder.remove(s)
            all_subfolder = sorted(all_subfolder)
            beg = int(beg_ratio * len(all_subfolder))
            end = int(end_ratio * len(all_subfolder))
            if not self.use_all_data:
                end = min(end, 75)
            all_subfolder = all_subfolder[beg:end]
            self.all_zarr_paths += [os.path.join(obj_path, s) for s in all_subfolder]
            self.all_zarr_categories += [cat_idx for s in all_subfolder]
            this_obj_episode_beg = episode_idx
            for s in all_subfolder:
                self.episode_idx_to_obj_id[episode_idx] = obj_path
                episode_idx += 1
            this_obj_episode_end = episode_idx
            self.obj_id_to_all_episodes_indices[obj_path] = [i for i in range(this_obj_episode_beg, this_obj_episode_end)]            

        cprint('Preparing all zarr paths', 'green')
        self.episode_lengths = []
        self.episode_idx_to_grasp_frame_idx = {}
        self.episode_idx_to_open_frame_idx = {}
        for idx, zarr_path in enumerate(tqdm(self.all_zarr_paths)):
            cat_idx = self.all_zarr_categories[idx]
            all_substeps = os.listdir(zarr_path)
            all_substeps = [s for s in all_substeps if s.endswith('.pkl')]
            all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0]))
                
            first_goal = None
            
            self.episode_lengths.append(len(all_substeps))
            self.cat_counts[cat_idx] += len(all_substeps)
                
        self.episode_lengths = np.array(self.episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        cprint(f'Finished preparing all zarr paths with total datapoints: {self.accumulated_episode_lengths[-1]}', 'green')
        self.class_weights = [1.0 / count if count > 0 else 0.0 for count in self.cat_counts]
        self.class_weights = np.array(self.class_weights)
        num_existing_classes = np.sum(self.class_weights > 0)
        self.class_weights *= np.sum(self.cat_counts) / num_existing_classes  # normalize to have sum of weights equal to number of classes
        print(f'Class weights: {self.class_weights}')
        print(f'Cat_counts: {self.cat_counts}')

        self.cat_idxs = np.repeat(self.all_zarr_categories, self.episode_lengths)
        self.all_weights = self.class_weights[self.cat_idxs]
        self.all_weights /= np.sum(self.all_weights)  # normalize weights to sum to 1

    def __len__(self):
        return self.accumulated_episode_lengths[-1]
    
    def read_pickle_data(self, episode_idx, step_idx):
        step_path = os.path.join(self.all_zarr_paths[episode_idx], str(step_idx) + '.pkl')
        cat_idx = self.all_zarr_categories[episode_idx]
        weight = self.class_weights[cat_idx]
        with open(step_path, 'rb') as f:
            data = pickle.load(f)
        pointcloud = data['point_cloud'][:][0].astype(np.float32)
        gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
        goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)
        return pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, weight

    def __getitem__(self, idx):
        
        episode_idx = np.searchsorted(self.accumulated_episode_lengths, idx, side='right')
        start_idx = idx - self.accumulated_episode_lengths[episode_idx]

        if start_idx < 0:
            start_idx += self.episode_lengths[episode_idx]
            
        pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, weight = self.read_pickle_data(episode_idx, start_idx)

        # return pointcloud, gripper_pcd, goal_gripper_pcd, cat_idx, weight
        data = {
            "inputs": torch.from_numpy(np.concatenate([pointcloud, gripper_pcd], axis=0)),
            "goal_gripper_pcd": torch.from_numpy(goal_gripper_pcd),
        }
        
        return data


def collate(batch):
    # TODO: change this according to articubot keys
    batch = [data for data in batch if not data.get('invalid', False)]
    batch = {key: [data[key] for data in batch] for key in batch[0]}
    # import pdb; pdb.set_trace()
    
    ### an example batch data from m2t2
    """
    dict_keys(['inputs', 'points', 'seg', 'cam_pose', 'object_inputs', 'ee_pose', 'bottom_center', 
        'object_center', 'scene', 'task', 'names', 'grasping_masks', 'contact_dirs', 'approach_dirs', 
        'grasps', 'offsets', 'placement_masks', 'placement_region'])
    """
    
    ### for articubot:
    """
    inputs: BxNx3 (concatenated scene and gripper points), maybe with one hot
    goal_gripper_points: Bx1x12
    """
    
    ### M2T2
    if "goal_gripper_pcd" not in batch:
        if 'task' in batch:
            task = batch.pop('task')
            batch['task_is_pick'] = torch.stack([
                torch.tensor(t == 'pick') for t in task
            ])
            batch['task_is_place'] = torch.stack([
                torch.tensor(t == 'place') for t in task
            ])
        for key in batch:
            if key in [
                'inputs', 'points', 'seg', 'object_inputs', 'bottom_center',
                'cam_pose', 'ee_pose', 'placement_masks', 'placement_region',
                'lang_tokens'
            ]:
                batch[key] = torch.stack(batch[key])
            if key in [
                'contact_dirs', 'approach_dirs', 'offsets'
            ]:
                batch[key] = torch.cat(batch[key])
                
                
        """
        example batch['inputs'].shape: torch.Size([4, 16384, 6])
        contact_dirs: 430, 3 (concatenated across the batch, not sure why)
        """
                
    else:
        for key in batch:
            batch[key] = torch.stack(batch[key])

    
    # import pdb; pdb.set_trace()
    
    return batch


def get_dataset_paths(beg_ratio=0, end_ratio=0.9, use_all_data=False, dataset_prefix=None, num_train_objects=200):
    
    if dataset_prefix is None:
        dataset_prefix = '/project/flame/yufeiw2/RoboGen-sim2real/data/dp3_demo'
        
    num_train_objects = str(num_train_objects)
    
    print(" ", num_train_objects)
    print("num_train_objects: ", num_train_objects)
    print("num_train_objects: ", num_train_objects)
    print("num_train_objects: ", num_train_objects)
    print("num_train_objects: ", num_train_objects)
    
    if num_train_objects == 'test':
        data_name = [save_data_name_0]
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
        
    elif num_train_objects == 'articulated':
        data_name = [
            save_data_name_5, save_data_name_6, save_data_name_7, save_data_name_8, save_data_name_9,
            save_data_name_10, save_data_name_11, save_data_name_12, save_data_name_13, save_data_name_14, save_data_name_15, save_data_name_16, save_data_name_17, save_data_name_18, save_data_name_19,
            save_data_name_20, save_data_name_21, save_data_name_22, save_data_name_23, save_data_name_24, save_data_name_25, save_data_name_26, save_data_name_27, save_data_name_28, save_data_name_29,
            save_data_name_30, save_data_name_31, save_data_name_32, save_data_name_33, save_data_name_34, save_data_name_35, save_data_name_36, save_data_name_37, save_data_name_38, save_data_name_39,
            save_data_name_40, save_data_name_41, save_data_name_42, save_data_name_43, save_data_name_44, save_data_name_45, save_data_name_46, save_data_name_47, save_data_name_48, save_data_name_49,
        ] + articulated_new
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    elif num_train_objects == 'articulated_250':
        data_name = [
            save_data_name_5, save_data_name_6, save_data_name_7, save_data_name_8, save_data_name_9,
            save_data_name_10, save_data_name_11, save_data_name_12, save_data_name_13, save_data_name_14, save_data_name_15, save_data_name_16, save_data_name_17, save_data_name_18, save_data_name_19,
            save_data_name_20, save_data_name_21, save_data_name_22, save_data_name_23, save_data_name_24, save_data_name_25, save_data_name_26, save_data_name_27, save_data_name_28, save_data_name_29,
            save_data_name_30, save_data_name_31, save_data_name_32, save_data_name_33, save_data_name_34, save_data_name_35, save_data_name_36, save_data_name_37, save_data_name_38, save_data_name_39,
            save_data_name_40, save_data_name_41, save_data_name_42, save_data_name_43, save_data_name_44, save_data_name_45, save_data_name_46, save_data_name_47, save_data_name_48, save_data_name_49,
            save_data_name_50, save_data_name_51, save_data_name_52, save_data_name_53, save_data_name_54, save_data_name_55, save_data_name_56, save_data_name_57, save_data_name_58, save_data_name_59,
            save_data_name_60, save_data_name_61, save_data_name_62, save_data_name_63, save_data_name_64, save_data_name_65, save_data_name_66, save_data_name_67, save_data_name_68, save_data_name_69,
            save_data_name_70, save_data_name_71, save_data_name_72, save_data_name_73, save_data_name_74, save_data_name_75, save_data_name_76, save_data_name_77, save_data_name_78, save_data_name_79,
            save_data_name_80, save_data_name_81, save_data_name_82, save_data_name_83, save_data_name_84, save_data_name_85, save_data_name_86, save_data_name_87, save_data_name_88, save_data_name_89,
            save_data_name_90, save_data_name_91, save_data_name_92, save_data_name_93, save_data_name_94, save_data_name_95, save_data_name_96, save_data_name_97, save_data_name_98, save_data_name_99,
        ] + articulated_new
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    elif num_train_objects == 'articulated_full':
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_zarr_paths_part_3 = articulated_new
        all_zarr_paths_part_3 = ["{}/{}".format(dataset_prefix, name) for name in all_zarr_paths_part_3]
        all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2 + all_zarr_paths_part_3
    elif num_train_objects == 'full_and_close':
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_zarr_paths_part_3 = articulated_new
        all_zarr_paths_part_3 = ["{}/{}".format(dataset_prefix, name) for name in all_zarr_paths_part_3]
        all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2 + all_zarr_paths_part_3
        close_prefix = '/mnt/RoboGen_sim2real/data/dp3_demo/invert'
        close_prefix_2 = '/mnt/RoboGen_sim2real/data/dp3_demo/invert_new'
        close_names = os.listdir(close_prefix)
        close_names = [name for name in close_names if name[0].isalpha()]
        close_names_2 = os.listdir(close_prefix_2)
        close_obj_paths = [
            "{}/{}".format(close_prefix, close_names[i]) for i in range(len(close_names))
        ] + [
            "{}/{}".format(close_prefix_2, close_names_2[i]) for i in range(len(close_names_2))
        ]
        all_obj_paths += close_obj_paths
    elif num_train_objects == 'articulated_full_dagger':
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_zarr_paths_part_3 = articulated_new
        all_zarr_paths_part_3 = ["{}/{}".format(dataset_prefix, name) for name in all_zarr_paths_part_3]
        all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2 + all_zarr_paths_part_3
        dagger_prefix = '/mnt/RoboGen_sim2real/data/weighted_full_dagger'
        dagger_names = os.listdir(dagger_prefix)
        dagger_obj_paths = [
            "{}/{}".format(dagger_prefix, dagger_names[i]) for i in range(len(dagger_names))
        ]
        all_obj_paths += dagger_obj_paths
    elif num_train_objects == 'dagger':
        data_name = [
            save_data_name_5, save_data_name_6, save_data_name_7, save_data_name_8, save_data_name_9,
            save_data_name_10, save_data_name_11, save_data_name_12, save_data_name_13, save_data_name_14, save_data_name_15, save_data_name_16, save_data_name_17, save_data_name_18, save_data_name_19,
            save_data_name_20, save_data_name_21, save_data_name_22, save_data_name_23, save_data_name_24, save_data_name_25, save_data_name_26, save_data_name_27, save_data_name_28, save_data_name_29,
            save_data_name_30, save_data_name_31, save_data_name_32, save_data_name_33, save_data_name_34, save_data_name_35, save_data_name_36, save_data_name_37, save_data_name_38, save_data_name_39,
            save_data_name_40, save_data_name_41, save_data_name_42, save_data_name_43, save_data_name_44, save_data_name_45, save_data_name_46, save_data_name_47, save_data_name_48, save_data_name_49,
        ] + articulated_new
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
        dagger_prefix = '/mnt/RoboGen_sim2real/data/dagger'
        dagger_names = os.listdir(dagger_prefix)
        dagger_obj_paths = [
            "{}/{}".format(dagger_prefix, dagger_names[i]) for i in range(len(dagger_names))
        ]
        all_obj_paths += dagger_obj_paths
    elif num_train_objects == 'bucket':
        data_name = [
            "bucket_100444", "bucket_100452", "bucket_100454", "bucket_100460", "bucket_100461",
            "bucket_100462", "bucket_100469", "bucket_100472", "bucket_102352", "bucket_102365",
        ]
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    elif num_train_objects == 'faucet':
        data_name = [
            "faucet_148", "faucet_149", "faucet_152", "faucet_153", "faucet_154",
            "faucet_168", "faucet_811", "faucet_857", "faucet_960", "faucet_991",
        ]
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    elif num_train_objects == 'foldingchair':
        data_name = [
            "foldingchair_100520", "foldingchair_100521", "foldingchair_100526", "foldingchair_100562", "foldingchair_100586",
            "foldingchair_100590", "foldingchair_100599", "foldingchair_102263", "foldingchair_102269", "foldingchair_102314",
        ]
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    elif num_train_objects == 'laptop':
        data_name = [
            "laptop_9748", "laptop_9912", "laptop_9960", "laptop_9968", "laptop_9992",
            "laptop_9996", "laptop_10040", "laptop_10098", "laptop_10101", "laptop_10238",
        ]
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    elif num_train_objects == 'stapler':
        data_name = [
            "stapler_103095", "stapler_103099", "stapler_103100", "stapler_103104", "stapler_103111",
            "stapler_103292", "stapler_103293", "stapler_103297", "stapler_103299", "stapler_103301",
        ]
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    elif num_train_objects == 'toilet':
        data_name = [
            "toilet_101320", "toilet_102621", "toilet_102622", "toilet_102630", "toilet_102634",
            "toilet_102645", "toilet_102648", "toilet_102651", "toilet_102652", "toilet_102658",
        ]
        all_obj_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    elif num_train_objects == 'debug':
        all_obj_paths = [f'{dataset_prefix}/0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1']
    elif num_train_objects == '10':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(10)]
    elif num_train_objects == '50':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(50)]
    elif num_train_objects == '100':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(100)]
    elif num_train_objects == '200':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(200)]
    elif num_train_objects == '300':
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2
        
    elif num_train_objects == "camera_random_10_obj_high_level":
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_10_save_data_name_{}".format(i)]) for i in range(20)]
    elif num_train_objects == 'camera_random_50_obj_high_level':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_50_save_data_name_{}".format(i)]) for i in range(87)]
    elif num_train_objects == 'camera_random_100_obj_high_level':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_100_save_data_name_{}".format(i)]) for i in range(175)]
    elif num_train_objects == 'camera_random_200_obj_high_level':
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_200_save_data_name_{}".format(i)]) for i in range(350)]
    elif num_train_objects == 'camera_random_500_obj_high_level' or num_train_objects == "500_object_high_level":
        all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(462)]
        
    elif num_train_objects == '300_old':
        
        all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
        f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
        f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
        f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
        f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
        f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
        f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
        f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
        f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
        f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
        f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
        f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
        f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
        f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
        f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
        f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
        f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
        f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
        f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
        f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
        f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
        f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
        f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
        f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
        f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
        f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
        f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
        f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
        f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}']
    elif num_train_objects == '500':
        all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
        f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
        f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
        f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
        f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
        f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
        f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
        f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
        f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
        f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
        f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
        f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
        f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
        f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
        f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
        f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
        f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
        f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
        f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
        f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
        f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
        f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
        f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
        f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
        f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
        f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
        f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
        f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
        f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}',
        f'{dataset_prefix}/{save_data_name_287}', f'{dataset_prefix}/{save_data_name_288}', f'{dataset_prefix}/{save_data_name_289}', f'{dataset_prefix}/{save_data_name_290}', f'{dataset_prefix}/{save_data_name_291}', f'{dataset_prefix}/{save_data_name_292}', f'{dataset_prefix}/{save_data_name_293}', f'{dataset_prefix}/{save_data_name_294}', f'{dataset_prefix}/{save_data_name_295}', f'{dataset_prefix}/{save_data_name_296}', f'{dataset_prefix}/{save_data_name_297}', f'{dataset_prefix}/{save_data_name_298}', f'{dataset_prefix}/{save_data_name_299}', f'{dataset_prefix}/{save_data_name_300}', f'{dataset_prefix}/{save_data_name_301}', f'{dataset_prefix}/{save_data_name_302}', f'{dataset_prefix}/{save_data_name_303}', f'{dataset_prefix}/{save_data_name_304}', f'{dataset_prefix}/{save_data_name_305}', f'{dataset_prefix}/{save_data_name_306}', f'{dataset_prefix}/{save_data_name_307}', f'{dataset_prefix}/{save_data_name_308}', f'{dataset_prefix}/{save_data_name_309}', f'{dataset_prefix}/{save_data_name_310}', f'{dataset_prefix}/{save_data_name_311}', f'{dataset_prefix}/{save_data_name_312}', f'{dataset_prefix}/{save_data_name_313}', f'{dataset_prefix}/{save_data_name_314}', f'{dataset_prefix}/{save_data_name_315}', f'{dataset_prefix}/{save_data_name_316}', f'{dataset_prefix}/{save_data_name_317}', f'{dataset_prefix}/{save_data_name_318}', f'{dataset_prefix}/{save_data_name_319}', f'{dataset_prefix}/{save_data_name_320}', f'{dataset_prefix}/{save_data_name_321}', f'{dataset_prefix}/{save_data_name_322}', f'{dataset_prefix}/{save_data_name_323}', f'{dataset_prefix}/{save_data_name_324}', f'{dataset_prefix}/{save_data_name_325}', f'{dataset_prefix}/{save_data_name_326}', f'{dataset_prefix}/{save_data_name_327}', f'{dataset_prefix}/{save_data_name_328}', f'{dataset_prefix}/{save_data_name_329}', f'{dataset_prefix}/{save_data_name_330}', f'{dataset_prefix}/{save_data_name_331}', f'{dataset_prefix}/{save_data_name_332}', f'{dataset_prefix}/{save_data_name_333}', f'{dataset_prefix}/{save_data_name_334}', f'{dataset_prefix}/{save_data_name_335}', f'{dataset_prefix}/{save_data_name_336}', f'{dataset_prefix}/{save_data_name_337}', f'{dataset_prefix}/{save_data_name_338}', f'{dataset_prefix}/{save_data_name_339}', f'{dataset_prefix}/{save_data_name_340}', f'{dataset_prefix}/{save_data_name_341}', f'{dataset_prefix}/{save_data_name_342}', f'{dataset_prefix}/{save_data_name_343}', f'{dataset_prefix}/{save_data_name_344}', f'{dataset_prefix}/{save_data_name_345}', f'{dataset_prefix}/{save_data_name_346}', f'{dataset_prefix}/{save_data_name_347}', f'{dataset_prefix}/{save_data_name_348}', f'{dataset_prefix}/{save_data_name_349}', f'{dataset_prefix}/{save_data_name_350}', f'{dataset_prefix}/{save_data_name_351}', f'{dataset_prefix}/{save_data_name_352}', f'{dataset_prefix}/{save_data_name_353}', f'{dataset_prefix}/{save_data_name_354}', f'{dataset_prefix}/{save_data_name_355}', f'{dataset_prefix}/{save_data_name_356}', f'{dataset_prefix}/{save_data_name_357}', f'{dataset_prefix}/{save_data_name_358}', f'{dataset_prefix}/{save_data_name_359}', f'{dataset_prefix}/{save_data_name_360}', f'{dataset_prefix}/{save_data_name_361}', f'{dataset_prefix}/{save_data_name_362}', f'{dataset_prefix}/{save_data_name_363}', f'{dataset_prefix}/{save_data_name_364}', f'{dataset_prefix}/{save_data_name_365}', f'{dataset_prefix}/{save_data_name_366}', f'{dataset_prefix}/{save_data_name_367}', f'{dataset_prefix}/{save_data_name_368}', f'{dataset_prefix}/{save_data_name_369}', f'{dataset_prefix}/{save_data_name_370}', f'{dataset_prefix}/{save_data_name_371}', f'{dataset_prefix}/{save_data_name_372}', f'{dataset_prefix}/{save_data_name_373}', f'{dataset_prefix}/{save_data_name_374}', f'{dataset_prefix}/{save_data_name_375}', f'{dataset_prefix}/{save_data_name_376}', f'{dataset_prefix}/{save_data_name_377}', f'{dataset_prefix}/{save_data_name_378}', f'{dataset_prefix}/{save_data_name_379}', f'{dataset_prefix}/{save_data_name_380}', f'{dataset_prefix}/{save_data_name_381}', f'{dataset_prefix}/{save_data_name_382}', f'{dataset_prefix}/{save_data_name_383}', f'{dataset_prefix}/{save_data_name_384}', f'{dataset_prefix}/{save_data_name_385}', f'{dataset_prefix}/{save_data_name_386}', f'{dataset_prefix}/{save_data_name_387}', f'{dataset_prefix}/{save_data_name_388}', f'{dataset_prefix}/{save_data_name_389}', f'{dataset_prefix}/{save_data_name_390}', f'{dataset_prefix}/{save_data_name_391}', f'{dataset_prefix}/{save_data_name_392}', f'{dataset_prefix}/{save_data_name_393}', f'{dataset_prefix}/{save_data_name_394}', f'{dataset_prefix}/{save_data_name_395}', f'{dataset_prefix}/{save_data_name_396}', f'{dataset_prefix}/{save_data_name_397}', f'{dataset_prefix}/{save_data_name_398}', f'{dataset_prefix}/{save_data_name_399}', f'{dataset_prefix}/{save_data_name_400}', f'{dataset_prefix}/{save_data_name_401}', f'{dataset_prefix}/{save_data_name_402}', f'{dataset_prefix}/{save_data_name_403}', f'{dataset_prefix}/{save_data_name_404}', f'{dataset_prefix}/{save_data_name_405}', f'{dataset_prefix}/{save_data_name_406}', f'{dataset_prefix}/{save_data_name_407}', f'{dataset_prefix}/{save_data_name_408}', f'{dataset_prefix}/{save_data_name_409}', f'{dataset_prefix}/{save_data_name_410}', f'{dataset_prefix}/{save_data_name_411}', f'{dataset_prefix}/{save_data_name_412}', f'{dataset_prefix}/{save_data_name_413}', f'{dataset_prefix}/{save_data_name_414}', f'{dataset_prefix}/{save_data_name_415}', f'{dataset_prefix}/{save_data_name_416}', f'{dataset_prefix}/{save_data_name_417}', f'{dataset_prefix}/{save_data_name_418}', f'{dataset_prefix}/{save_data_name_419}', f'{dataset_prefix}/{save_data_name_420}', f'{dataset_prefix}/{save_data_name_421}', f'{dataset_prefix}/{save_data_name_422}', f'{dataset_prefix}/{save_data_name_423}', f'{dataset_prefix}/{save_data_name_424}', f'{dataset_prefix}/{save_data_name_425}', f'{dataset_prefix}/{save_data_name_426}', f'{dataset_prefix}/{save_data_name_427}', f'{dataset_prefix}/{save_data_name_428}', f'{dataset_prefix}/{save_data_name_429}', f'{dataset_prefix}/{save_data_name_430}', f'{dataset_prefix}/{save_data_name_431}', f'{dataset_prefix}/{save_data_name_432}', f'{dataset_prefix}/{save_data_name_433}', f'{dataset_prefix}/{save_data_name_434}', f'{dataset_prefix}/{save_data_name_435}', f'{dataset_prefix}/{save_data_name_436}', f'{dataset_prefix}/{save_data_name_437}', f'{dataset_prefix}/{save_data_name_438}', f'{dataset_prefix}/{save_data_name_439}', f'{dataset_prefix}/{save_data_name_440}', f'{dataset_prefix}/{save_data_name_441}', f'{dataset_prefix}/{save_data_name_442}', f'{dataset_prefix}/{save_data_name_443}', f'{dataset_prefix}/{save_data_name_444}', f'{dataset_prefix}/{save_data_name_445}', f'{dataset_prefix}/{save_data_name_446}', f'{dataset_prefix}/{save_data_name_447}', f'{dataset_prefix}/{save_data_name_448}', f'{dataset_prefix}/{save_data_name_449}', f'{dataset_prefix}/{save_data_name_450}', f'{dataset_prefix}/{save_data_name_451}', f'{dataset_prefix}/{save_data_name_452}', f'{dataset_prefix}/{save_data_name_453}', f'{dataset_prefix}/{save_data_name_454}', f'{dataset_prefix}/{save_data_name_455}', f'{dataset_prefix}/{save_data_name_456}', f'{dataset_prefix}/{save_data_name_457}', f'{dataset_prefix}/{save_data_name_458}', f'{dataset_prefix}/{save_data_name_459}', f'{dataset_prefix}/{save_data_name_460}', f'{dataset_prefix}/{save_data_name_461}', f'{dataset_prefix}/{save_data_name_462}',
        ]
    elif num_train_objects == '600':
        all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
        f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
        f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
        f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
        f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
        f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
        f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
        f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
        f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
        f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
        f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
        f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
        f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
        f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
        f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
        f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
        f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
        f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
        f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
        f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
        f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
        f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
        f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
        f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
        f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
        f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
        f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
        f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
        f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}',
        f'{dataset_prefix}/{save_data_name_287}', f'{dataset_prefix}/{save_data_name_288}', f'{dataset_prefix}/{save_data_name_289}', f'{dataset_prefix}/{save_data_name_290}', f'{dataset_prefix}/{save_data_name_291}', f'{dataset_prefix}/{save_data_name_292}', f'{dataset_prefix}/{save_data_name_293}', f'{dataset_prefix}/{save_data_name_294}', f'{dataset_prefix}/{save_data_name_295}', f'{dataset_prefix}/{save_data_name_296}', f'{dataset_prefix}/{save_data_name_297}', f'{dataset_prefix}/{save_data_name_298}', f'{dataset_prefix}/{save_data_name_299}', f'{dataset_prefix}/{save_data_name_300}', f'{dataset_prefix}/{save_data_name_301}', f'{dataset_prefix}/{save_data_name_302}', f'{dataset_prefix}/{save_data_name_303}', f'{dataset_prefix}/{save_data_name_304}', f'{dataset_prefix}/{save_data_name_305}', f'{dataset_prefix}/{save_data_name_306}', f'{dataset_prefix}/{save_data_name_307}', f'{dataset_prefix}/{save_data_name_308}', f'{dataset_prefix}/{save_data_name_309}', f'{dataset_prefix}/{save_data_name_310}', f'{dataset_prefix}/{save_data_name_311}', f'{dataset_prefix}/{save_data_name_312}', f'{dataset_prefix}/{save_data_name_313}', f'{dataset_prefix}/{save_data_name_314}', f'{dataset_prefix}/{save_data_name_315}', f'{dataset_prefix}/{save_data_name_316}', f'{dataset_prefix}/{save_data_name_317}', f'{dataset_prefix}/{save_data_name_318}', f'{dataset_prefix}/{save_data_name_319}', f'{dataset_prefix}/{save_data_name_320}', f'{dataset_prefix}/{save_data_name_321}', f'{dataset_prefix}/{save_data_name_322}', f'{dataset_prefix}/{save_data_name_323}', f'{dataset_prefix}/{save_data_name_324}', f'{dataset_prefix}/{save_data_name_325}', f'{dataset_prefix}/{save_data_name_326}', f'{dataset_prefix}/{save_data_name_327}', f'{dataset_prefix}/{save_data_name_328}', f'{dataset_prefix}/{save_data_name_329}', f'{dataset_prefix}/{save_data_name_330}', f'{dataset_prefix}/{save_data_name_331}', f'{dataset_prefix}/{save_data_name_332}', f'{dataset_prefix}/{save_data_name_333}', f'{dataset_prefix}/{save_data_name_334}', f'{dataset_prefix}/{save_data_name_335}', f'{dataset_prefix}/{save_data_name_336}', f'{dataset_prefix}/{save_data_name_337}', f'{dataset_prefix}/{save_data_name_338}', f'{dataset_prefix}/{save_data_name_339}', f'{dataset_prefix}/{save_data_name_340}', f'{dataset_prefix}/{save_data_name_341}', f'{dataset_prefix}/{save_data_name_342}', f'{dataset_prefix}/{save_data_name_343}', f'{dataset_prefix}/{save_data_name_344}', f'{dataset_prefix}/{save_data_name_345}', f'{dataset_prefix}/{save_data_name_346}', f'{dataset_prefix}/{save_data_name_347}', f'{dataset_prefix}/{save_data_name_348}', f'{dataset_prefix}/{save_data_name_349}', f'{dataset_prefix}/{save_data_name_350}', f'{dataset_prefix}/{save_data_name_351}', f'{dataset_prefix}/{save_data_name_352}', f'{dataset_prefix}/{save_data_name_353}', f'{dataset_prefix}/{save_data_name_354}', f'{dataset_prefix}/{save_data_name_355}', f'{dataset_prefix}/{save_data_name_356}', f'{dataset_prefix}/{save_data_name_357}', f'{dataset_prefix}/{save_data_name_358}', f'{dataset_prefix}/{save_data_name_359}', f'{dataset_prefix}/{save_data_name_360}', f'{dataset_prefix}/{save_data_name_361}', f'{dataset_prefix}/{save_data_name_362}', f'{dataset_prefix}/{save_data_name_363}', f'{dataset_prefix}/{save_data_name_364}', f'{dataset_prefix}/{save_data_name_365}', f'{dataset_prefix}/{save_data_name_366}', f'{dataset_prefix}/{save_data_name_367}', f'{dataset_prefix}/{save_data_name_368}', f'{dataset_prefix}/{save_data_name_369}', f'{dataset_prefix}/{save_data_name_370}', f'{dataset_prefix}/{save_data_name_371}', f'{dataset_prefix}/{save_data_name_372}', f'{dataset_prefix}/{save_data_name_373}', f'{dataset_prefix}/{save_data_name_374}', f'{dataset_prefix}/{save_data_name_375}', f'{dataset_prefix}/{save_data_name_376}', f'{dataset_prefix}/{save_data_name_377}', f'{dataset_prefix}/{save_data_name_378}', f'{dataset_prefix}/{save_data_name_379}', f'{dataset_prefix}/{save_data_name_380}', f'{dataset_prefix}/{save_data_name_381}', f'{dataset_prefix}/{save_data_name_382}', f'{dataset_prefix}/{save_data_name_383}', f'{dataset_prefix}/{save_data_name_384}', f'{dataset_prefix}/{save_data_name_385}', f'{dataset_prefix}/{save_data_name_386}', f'{dataset_prefix}/{save_data_name_387}', f'{dataset_prefix}/{save_data_name_388}', f'{dataset_prefix}/{save_data_name_389}', f'{dataset_prefix}/{save_data_name_390}', f'{dataset_prefix}/{save_data_name_391}', f'{dataset_prefix}/{save_data_name_392}', f'{dataset_prefix}/{save_data_name_393}', f'{dataset_prefix}/{save_data_name_394}', f'{dataset_prefix}/{save_data_name_395}', f'{dataset_prefix}/{save_data_name_396}', f'{dataset_prefix}/{save_data_name_397}', f'{dataset_prefix}/{save_data_name_398}', f'{dataset_prefix}/{save_data_name_399}', f'{dataset_prefix}/{save_data_name_400}', f'{dataset_prefix}/{save_data_name_401}', f'{dataset_prefix}/{save_data_name_402}', f'{dataset_prefix}/{save_data_name_403}', f'{dataset_prefix}/{save_data_name_404}', f'{dataset_prefix}/{save_data_name_405}', f'{dataset_prefix}/{save_data_name_406}', f'{dataset_prefix}/{save_data_name_407}', f'{dataset_prefix}/{save_data_name_408}', f'{dataset_prefix}/{save_data_name_409}', f'{dataset_prefix}/{save_data_name_410}', f'{dataset_prefix}/{save_data_name_411}', f'{dataset_prefix}/{save_data_name_412}', f'{dataset_prefix}/{save_data_name_413}', f'{dataset_prefix}/{save_data_name_414}', f'{dataset_prefix}/{save_data_name_415}', f'{dataset_prefix}/{save_data_name_416}', f'{dataset_prefix}/{save_data_name_417}', f'{dataset_prefix}/{save_data_name_418}', f'{dataset_prefix}/{save_data_name_419}', f'{dataset_prefix}/{save_data_name_420}', f'{dataset_prefix}/{save_data_name_421}', f'{dataset_prefix}/{save_data_name_422}', f'{dataset_prefix}/{save_data_name_423}', f'{dataset_prefix}/{save_data_name_424}', f'{dataset_prefix}/{save_data_name_425}', f'{dataset_prefix}/{save_data_name_426}', f'{dataset_prefix}/{save_data_name_427}', f'{dataset_prefix}/{save_data_name_428}', f'{dataset_prefix}/{save_data_name_429}', f'{dataset_prefix}/{save_data_name_430}', f'{dataset_prefix}/{save_data_name_431}', f'{dataset_prefix}/{save_data_name_432}', f'{dataset_prefix}/{save_data_name_433}', f'{dataset_prefix}/{save_data_name_434}', f'{dataset_prefix}/{save_data_name_435}', f'{dataset_prefix}/{save_data_name_436}', f'{dataset_prefix}/{save_data_name_437}', f'{dataset_prefix}/{save_data_name_438}', f'{dataset_prefix}/{save_data_name_439}', f'{dataset_prefix}/{save_data_name_440}', f'{dataset_prefix}/{save_data_name_441}', f'{dataset_prefix}/{save_data_name_442}', f'{dataset_prefix}/{save_data_name_443}', f'{dataset_prefix}/{save_data_name_444}', f'{dataset_prefix}/{save_data_name_445}', f'{dataset_prefix}/{save_data_name_446}', f'{dataset_prefix}/{save_data_name_447}', f'{dataset_prefix}/{save_data_name_448}', f'{dataset_prefix}/{save_data_name_449}', f'{dataset_prefix}/{save_data_name_450}', f'{dataset_prefix}/{save_data_name_451}', f'{dataset_prefix}/{save_data_name_452}', f'{dataset_prefix}/{save_data_name_453}', f'{dataset_prefix}/{save_data_name_454}', f'{dataset_prefix}/{save_data_name_455}', f'{dataset_prefix}/{save_data_name_456}', f'{dataset_prefix}/{save_data_name_457}', f'{dataset_prefix}/{save_data_name_458}', f'{dataset_prefix}/{save_data_name_459}', f'{dataset_prefix}/{save_data_name_460}', f'{dataset_prefix}/{save_data_name_461}', f'{dataset_prefix}/{save_data_name_462}',
        f'{dataset_prefix}/{save_data_name_463}',f'{dataset_prefix}/{save_data_name_464}',f'{dataset_prefix}/{save_data_name_465}',f'{dataset_prefix}/{save_data_name_466}',f'{dataset_prefix}/{save_data_name_467}',f'{dataset_prefix}/{save_data_name_468}',f'{dataset_prefix}/{save_data_name_469}',f'{dataset_prefix}/{save_data_name_470}',f'{dataset_prefix}/{save_data_name_471}',f'{dataset_prefix}/{save_data_name_472}',f'{dataset_prefix}/{save_data_name_473}',f'{dataset_prefix}/{save_data_name_474}',f'{dataset_prefix}/{save_data_name_475}',f'{dataset_prefix}/{save_data_name_476}',f'{dataset_prefix}/{save_data_name_477}',f'{dataset_prefix}/{save_data_name_478}',f'{dataset_prefix}/{save_data_name_479}',f'{dataset_prefix}/{save_data_name_480}',f'{dataset_prefix}/{save_data_name_481}',f'{dataset_prefix}/{save_data_name_482}',f'{dataset_prefix}/{save_data_name_483}',f'{dataset_prefix}/{save_data_name_484}',f'{dataset_prefix}/{save_data_name_485}',f'{dataset_prefix}/{save_data_name_486}',f'{dataset_prefix}/{save_data_name_487}',f'{dataset_prefix}/{save_data_name_488}',f'{dataset_prefix}/{save_data_name_489}',f'{dataset_prefix}/{save_data_name_490}',f'{dataset_prefix}/{save_data_name_491}',f'{dataset_prefix}/{save_data_name_492}',f'{dataset_prefix}/{save_data_name_493}',f'{dataset_prefix}/{save_data_name_494}',f'{dataset_prefix}/{save_data_name_495}',f'{dataset_prefix}/{save_data_name_496}',f'{dataset_prefix}/{save_data_name_497}',f'{dataset_prefix}/{save_data_name_498}',f'{dataset_prefix}/{save_data_name_499}',f'{dataset_prefix}/{save_data_name_500}',f'{dataset_prefix}/{save_data_name_501}',f'{dataset_prefix}/{save_data_name_502}',f'{dataset_prefix}/{save_data_name_503}',f'{dataset_prefix}/{save_data_name_504}',f'{dataset_prefix}/{save_data_name_505}',f'{dataset_prefix}/{save_data_name_506}',f'{dataset_prefix}/{save_data_name_507}',f'{dataset_prefix}/{save_data_name_508}',f'{dataset_prefix}/{save_data_name_509}',f'{dataset_prefix}/{save_data_name_510}',f'{dataset_prefix}/{save_data_name_511}',f'{dataset_prefix}/{save_data_name_512}',f'{dataset_prefix}/{save_data_name_513}',f'{dataset_prefix}/{save_data_name_514}',f'{dataset_prefix}/{save_data_name_515}',f'{dataset_prefix}/{save_data_name_516}',f'{dataset_prefix}/{save_data_name_517}',f'{dataset_prefix}/{save_data_name_518}',f'{dataset_prefix}/{save_data_name_519}',f'{dataset_prefix}/{save_data_name_520}',f'{dataset_prefix}/{save_data_name_521}',f'{dataset_prefix}/{save_data_name_522}',f'{dataset_prefix}/{save_data_name_523}',f'{dataset_prefix}/{save_data_name_524}',f'{dataset_prefix}/{save_data_name_525}',f'{dataset_prefix}/{save_data_name_526}',f'{dataset_prefix}/{save_data_name_527}',f'{dataset_prefix}/{save_data_name_528}',f'{dataset_prefix}/{save_data_name_529}',f'{dataset_prefix}/{save_data_name_530}',f'{dataset_prefix}/{save_data_name_531}',f'{dataset_prefix}/{save_data_name_532}',f'{dataset_prefix}/{save_data_name_533}',f'{dataset_prefix}/{save_data_name_534}',f'{dataset_prefix}/{save_data_name_535}',f'{dataset_prefix}/{save_data_name_536}',f'{dataset_prefix}/{save_data_name_537}',f'{dataset_prefix}/{save_data_name_538}',f'{dataset_prefix}/{save_data_name_539}',f'{dataset_prefix}/{save_data_name_540}',f'{dataset_prefix}/{save_data_name_541}',f'{dataset_prefix}/{save_data_name_542}',f'{dataset_prefix}/{save_data_name_543}',f'{dataset_prefix}/{save_data_name_544}',f'{dataset_prefix}/{save_data_name_545}',f'{dataset_prefix}/{save_data_name_546}',f'{dataset_prefix}/{save_data_name_547}',f'{dataset_prefix}/{save_data_name_548}',f'{dataset_prefix}/{save_data_name_549}',f'{dataset_prefix}/{save_data_name_550}',f'{dataset_prefix}/{save_data_name_551}',f'{dataset_prefix}/{save_data_name_552}',f'{dataset_prefix}/{save_data_name_553}',f'{dataset_prefix}/{save_data_name_554}',f'{dataset_prefix}/{save_data_name_555}',f'{dataset_prefix}/{save_data_name_556}',f'{dataset_prefix}/{save_data_name_557}',f'{dataset_prefix}/{save_data_name_558}',f'{dataset_prefix}/{save_data_name_559}',f'{dataset_prefix}/{save_data_name_560}',f'{dataset_prefix}/{save_data_name_561}',f'{dataset_prefix}/{save_data_name_562}',f'{dataset_prefix}/{save_data_name_563}',f'{dataset_prefix}/{save_data_name_564}',f'{dataset_prefix}/{save_data_name_565}',f'{dataset_prefix}/{save_data_name_566}',f'{dataset_prefix}/{save_data_name_567}',f'{dataset_prefix}/{save_data_name_568}',f'{dataset_prefix}/{save_data_name_569}',
        ]
    elif num_train_objects == 'mixed_old_and_real_world_noisy_1119':
        dataset_prefix_1 = '/scratch/yufeiw2/dp3_demo'
        dataset_prefix_2 = '/scratch/yufeiw2/dp3_demo_real_world_noise_pcd'
        
        old_list = [i * 3 for i in range(150)]
        all_old_obj_paths = ["{}/{}".format(dataset_prefix_1, globals()["save_data_name_{}".format(i)]) for i in old_list]
        
        all_new_obj_paths = os.listdir(dataset_prefix_2)
        all_new_obj_paths = sorted(all_new_obj_paths)
        all_new_obj_paths = [os.path.join(dataset_prefix_2, x) for x in all_new_obj_paths]
        
        all_obj_paths = all_old_obj_paths + all_new_obj_paths
        
    elif num_train_objects == 'real_world_noisy_pcd_clean_distorted_goal_all':
        dataset_prefix = '/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal'
        all_obj_paths = os.listdir(dataset_prefix)
        all_obj_paths = sorted(all_obj_paths)
        all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
    
    elif num_train_objects == '500_plus_all_real_world':
        non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
        real_world_camera_500_paths = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd")
        real_world_camera_500_paths = sorted(real_world_camera_500_paths)
        real_world_camera_500_paths = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd", x) for x in real_world_camera_500_paths]
        all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths
        # all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
        print(all_obj_paths)
        
    elif num_train_objects == '500_plus_all_real_world_clean_distorted_goal':
        dataset_prefix = "/scratch/yufeiw2/dp3_demo_clean_distorted_goal"
        non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
        real_world_camera_500_paths = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal")
        real_world_camera_500_paths = sorted(real_world_camera_500_paths)
        real_world_camera_500_paths = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal", x) for x in real_world_camera_500_paths]
        all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths
        # all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
        print(all_obj_paths)
        
    else:
        raise ValueError('num_train_objects not supported')
        
    return all_obj_paths