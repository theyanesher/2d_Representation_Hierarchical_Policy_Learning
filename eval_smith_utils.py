import torch
import hydra
from omegaconf import OmegaConf
from train_ddp import TrainDP3Workspace
from copy import deepcopy
import os
import json

# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Convert low-level policy actions (unnormalized, delta rotation in gripper frame)
to mimicgen/robomimic env actions (normalized in [-1, 1], delta rotation in world frame).

- Policy: next_rotation = cur_rotation @ delta_rotation_gripper  (gripper/local frame)
- Env:    next_rotation = delta_rotation_world @ cur_rotation    (world frame)

So: delta_rotation_world = cur_rotation @ delta_rotation_gripper @ cur_rotation.T
"""

import numpy as np
import torch

import robosuite.utils.transform_utils as T
# from mimicgen.utils.rotation_transformer import RotationTransformer


# # 6D <-> matrix (numpy or torch)
# _ROT6D_TO_MAT = RotationTransformer("rotation_6d", "matrix")


def policy_action_to_env_action(
    action: np.ndarray,
    cur_eef_quat: np.ndarray,
    max_dpos: float,
    max_drot: float,
) -> np.ndarray:
    """
    Convert a single policy action to env action.

    Args:
        action: (10,) unnormalized policy output:
            action[:3]   = delta position (world)
            action[3:9]  = delta rotation in 6D, **gripper frame** (next_R = cur_R @ delta_R_gripper)
            action[9]    = gripper (will be clipped to [-1, 1])
        cur_eef_quat: (4,) current eef quaternion (wxyz or xyzw as per robosuite)
        max_dpos: scalar, controller output_max[0] for position
        max_drot: scalar, controller output_max[3] for rotation

    Returns:
        env_action: (7,) normalized for robomimic:
            env_action[:3]   = delta position in [-1, 1]
            env_action[3:6]  = delta rotation axis-angle in [-1, 1], **world frame**
            env_action[6]    = gripper in [-1, 1]
    """
    delta_pos = np.array(action[:3], dtype=np.float64)
    delta_rot_6d_gripper = np.array(action[3:9], dtype=np.float64)
    # gripper = float(action[9]) / 0.01 ### this is the panda gripper speed. This is the initial incorrect training that we forget to flip the gripper sign for square d2. 
    gripper = float(action[9]) / -0.01 ### this is the panda gripper speed

    # 1) Convert delta rotation from gripper frame to world frame
    #    Policy: next_R = cur_R @ delta_R_gripper
    #    Env:    next_R = delta_R_world @ cur_R  =>  delta_R_world = cur_R @ delta_R_gripper @ cur_R.T
    cur_R = T.quat2mat(cur_eef_quat)  # (3,3)
    delta_R_gripper = rotation_transfer_6D_to_matrix(delta_rot_6d_gripper)  # (3,3)
    delta_R_world = cur_R @ delta_R_gripper @ cur_R.T
    delta_axisangle_world = T.quat2axisangle(T.mat2quat(delta_R_world))

    # 2) Normalize to [-1, 1] for env
    pos_norm = np.clip(delta_pos / (max_dpos), -1.0, 1.0)
    rot_norm = np.clip(delta_axisangle_world / (max_drot), -1.0, 1.0)
    gripper_norm = np.clip(gripper, -1.0, 1.0)

    return np.concatenate([pos_norm, rot_norm, [gripper_norm]], dtype=np.float32)


def policy_action_batch_to_env_action(
    action_batch: np.ndarray,
    cur_eef_quats: np.ndarray,
    max_dpos: float,
    max_drot: float,
) -> np.ndarray:
    """
    Convert a batch of policy actions to env actions.

    Args:
        action_batch: (B, T, 10) or (B, 10) - we use the last step along T
        cur_eef_quats: (B, 4) current eef quaternions
        max_dpos, max_drot: scalars

    Returns:
        env_actions: (B, 7) if input was (B, 10); else (B, T, 7)
    """

    
    B, T, _ = action_batch.shape
    out = np.zeros((B, T, 7), dtype=np.float32)
    for i in range(B):
        for t in range(T):
            out[i, t, :] = policy_action_to_env_action(
                action_batch[i, t, :],
                cur_eef_quats[i],
                max_dpos,
                max_drot,
            )
   
    return out


# def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
#     """(6,) -> (3,3) rotation matrix."""
#     return _ROT6D_TO_MAT.forward(rot6d).reshape(3, 3)


def rotation_transfer_6D_to_matrix(orient):
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

    return rotate_matrix

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient


def low_level_policy_infer(obj_pcd, agent_pos, goal_gripper_pcd, gripper_pcd, policy, cat_idx=0):
    input_dict = {
        "point_cloud": obj_pcd,
        "agent_pos": agent_pos,
        'gripper_pcd': gripper_pcd,
        'goal_gripper_pcd': goal_gripper_pcd,
    }

    batched_action = policy.predict_action(input_dict, torch.tensor([cat_idx]).to(policy.device))
    # import pdb; pdb.set_trace()

    # return batched_action['action'] # B, T, 10
    return batched_action['action_pred'] # B, T, 10

def load_low_level_policy(exp_dir, checkpoint_name):
    with hydra.initialize(config_path='../3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
        cfg = recomposed_config
        
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir)

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    policy = policy.to('cuda')
    
    return policy


def load_multitask_high_level_model(path):
    ckpt_path = os.path.dirname(path)
    config_path = os.path.join(ckpt_path, "config.json")
    cfg = json.load(open(config_path, "r"))
    cfg = OmegaConf.create(cfg)
    args = cfg
    
    device = torch.device("cuda")
    general_args = args.general
    input_channel = 5 if general_args.add_one_hot_encoding else 3
    if general_args.get("use_rgb", False):
        input_channel += 3
    if general_args.get("use_dino", False):
        input_channel += 1024

    output_dim = 13 
    from test_PointNet2.model_invariant import PointNet2_super_multitask
    
    if "category_embedding_type" not in general_args:
        general_args.category_embedding_type = None
    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None
    
    model = PointNet2_super_multitask(num_classes=output_dim, keep_gripper_in_fps=general_args.keep_gripper_in_fps, input_channel=input_channel,
                                      first_sa_point=general_args.get("first_sa_point", 2048),
                                      fp_to_full=general_args.get("fp_to_full", False),
                                      replace_bn_w_gn=general_args.get("replace_bn_with_gn", False),
                                      replace_bn_w_in=general_args.get("replace_bn_with_in", False),
                                      embedding_dim=embedding_dim,
                                      film_in_sa_and_fp=general_args.get("film_in_sa_and_fp", False),
                                      embedding_as_input=general_args.get("embedding_as_input", False),
                                      replace_bn_w_ln=general_args.get("replace_bn_with_ln", False),
                                      ).to(device)
    
    model.load_state_dict(torch.load(path, map_location=device)['model'])
    print("Successfully load model from: ", path)
    model.eval()
        
    return model, args

def infer_multitask_high_level_model(inputs, goal_prediction_model, cat_embedding=None, high_level_args=None, extra=None):
    # Mirror the training-time pointcloud format selected by
    # `general.add_one_hot_encoding` in train_multitask_ddp_weighted_displacement_gmm.py:
    #   - off: scene+gripper concatenated as (B, N+4, 3)
    #   - on:  scene tagged (1,0), gripper tagged (0,1) -> (B, N+4, 5)
    # The model's first SA conv is built with `input_channel - 3` extra feature
    # channels, so a mismatch here triggers a "expected K channels, got M" error
    # at PointNetSetAbstractionMsg's first Conv2d.
    add_one_hot = False
    if high_level_args is not None:
        general = high_level_args.get("general", high_level_args) if hasattr(high_level_args, "get") else high_level_args
        add_one_hot = bool(general.get("add_one_hot_encoding", 0)) if hasattr(general, "get") else bool(getattr(general, "add_one_hot_encoding", 0))

    if add_one_hot:
        N_scene_points = inputs.shape[1] - 4
        pointcloud_one_hot = torch.zeros(inputs.shape[0], inputs.shape[1], 2).float().to(inputs.device)
        pointcloud_one_hot[:, :N_scene_points, 0] = 1
        pointcloud_one_hot[:, N_scene_points:, 1] = 1
        inputs = torch.cat([inputs, pointcloud_one_hot], dim=2)  # B, N+4, 5

    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1).float().contiguous()
    with torch.no_grad():
        pred_dict = goal_prediction_model(inputs_, cat_embedding, build_grasp=False, articubot_format=True) 
    outputs = pred_dict['pred_offsets']
    pred_points = pred_dict['pred_points'] 
    weights = pred_dict['pred_scores'].squeeze(-1)
    inputs = pred_points
    B, N, _, _ = outputs.shape
    outputs = outputs.view(B, N, -1)
    
    outputs = outputs.view(B, N, 4, 3)
    
    ### sample an displacement according to the weight
    probabilities = weights  # Must sum to 1
    probabilities = torch.nn.functional.softmax(weights, dim=1)

    # Sample one index based on the probabilities
    sampled_index = torch.argmax(probabilities.squeeze(0))

    displacement_mean = outputs[:, sampled_index, :, :] # B, 4, 3
    input_point_pos = inputs[:, sampled_index, :] # B, 3
    prediction = input_point_pos.unsqueeze(1) + displacement_mean # B, 4, 3
        
    return prediction
