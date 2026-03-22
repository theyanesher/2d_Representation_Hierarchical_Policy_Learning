import torch
# from manipulation.goal_image_utils import generate_heatmap_from_points
import numpy as np

def process_observations(obs_dict, observation_mode):
    if observation_mode == 'plucker_early_fusion':
        for cam_name in ['cam0', 'cam1', 'cam2']:
            im = obs_dict[f'{cam_name}_image']  # (B,T,3,H,W)
            plucker = obs_dict[f'{cam_name}_plucker']  # (B,T,6,H,W)
            fused = torch.cat([im, plucker], dim=2)  # (B,T,9,H,W)
            obs_dict[f'{cam_name}_image'] = fused
            del obs_dict[f'{cam_name}_plucker']
    elif observation_mode in ['pointmap_early_fusion', 'gripper_pointmap_early_fusion']:
        for cam_name in ['cam0', 'cam1', 'cam2']:
            im = obs_dict[f'{cam_name}_image']  # (B,T,3,H,W)
            pointmap = obs_dict[f'{cam_name}_pointmap']  # (B,T,3,H,W)
            fused = torch.cat([im, pointmap], dim=2)  # (B,T,6,H,W)
            obs_dict[f'{cam_name}_image'] = fused
            del obs_dict[f'{cam_name}_pointmap']

def process_obs_shape_meta(observation_mode, shape_meta, obs_shape_meta):
    if observation_mode == 'plucker_early_fusion':
        for cam_name in ['cam0', 'cam1', 'cam2']:
            im_shape = shape_meta['obs'][f'{cam_name}_image']['shape'] # (3,256,256)
            plucker_shape = shape_meta['obs'][f'{cam_name}_plucker']['shape'] # (6,256,256)
            new_imshape = (im_shape[0]+plucker_shape[0], im_shape[1], im_shape[2])
            obs_shape_meta[f'{cam_name}_image'] = {
                'shape': new_imshape,
                'type': 'rgb'
            }
            del obs_shape_meta[f'{cam_name}_plucker']
    elif observation_mode in ['pointmap_early_fusion', 'gripper_pointmap_early_fusion']:
        for cam_name in ['cam0', 'cam1', 'cam2']:
            im_shape = shape_meta['obs'][f'{cam_name}_image']['shape'] # (3,256,256)
            pointmap_shape = shape_meta['obs'][f'{cam_name}_pointmap']['shape'] # (3,256,256)
            new_imshape = (im_shape[0]+pointmap_shape[0], im_shape[1], im_shape[2])
            obs_shape_meta[f'{cam_name}_image'] = {
                'shape': new_imshape,
                'type': 'rgb'
            }
            del obs_shape_meta[f'{cam_name}_pointmap']

def filter_obs_keys(obs_dict, shape_meta, pointmap_gripper_frame=False):
    shape_meta_keys = set(shape_meta['obs'].keys())
    obs_dict_keys = set(obs_dict.keys())
    keys_to_remove = obs_dict_keys - shape_meta_keys
    if pointmap_gripper_frame:
        for key in [f'{cam_name}_pointmap' for cam_name in ['cam0', 'cam1', 'cam2']]:
            gripper_to_world = obs_dict['gripper_to_world']
            world_to_gripper = torch.linalg.inv(gripper_to_world)  # (T, 4, 4)
            R = world_to_gripper[:, :, :3, :3]  # (T, 3, 3)
            t = world_to_gripper[:, :, :3, 3]   # (T, 3)
            B, T, _, H, W = obs_dict[key].shape
            pts = obs_dict[key].reshape(B, T, 3, -1)          # (T, 3, H*W)
            mask = torch.all(pts == 0, axis=1, keepdims=True)  # (T, 1, H*W)
            transformed = (R @ pts + t[:, :, :, None])
            obs_dict[key] = torch.where(mask, 0.0, transformed).reshape(B, T, 3, H, W)

    for key in keys_to_remove:
        del obs_dict[key]
    for key in obs_dict.keys():
        if 'image' in key:
            obs_dict[key] = obs_dict[key].permute(0,1,4,2,3).to(torch.float32) / 255.0
        elif 'depth' in key:
            obs_dict[key] = obs_dict[key][:,:,None].to(torch.float32)
    assert set(obs_dict.keys()) == shape_meta_keys, f"After filtering, obs_dict keys {set(obs_dict.keys())} do not match shape_meta keys {shape_meta_keys}"
    return obs_dict
