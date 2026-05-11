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
from mimicgen.utils.rotation_transformer import RotationTransformer


# 6D <-> matrix (numpy or torch)
_ROT6D_TO_MAT = RotationTransformer("rotation_6d", "matrix")


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
    gripper = float(action[9])

    # 1) Convert delta rotation from gripper frame to world frame
    #    Policy: next_R = cur_R @ delta_R_gripper
    #    Env:    next_R = delta_R_world @ cur_R  =>  delta_R_world = cur_R @ delta_R_gripper @ cur_R.T
    cur_R = T.quat2mat(cur_eef_quat)  # (3,3)
    delta_R_gripper = _rot6d_to_matrix(delta_rot_6d_gripper)  # (3,3)
    delta_R_world = cur_R @ delta_R_gripper @ cur_R.T
    delta_axisangle_world = T.quat2axisangle(T.mat2quat(delta_R_world))

    # 2) Normalize to [-1, 1] for env
    pos_norm = np.clip(delta_pos / (max_dpos + 1e-8), -1.0, 1.0)
    rot_norm = np.clip(delta_axisangle_world / (max_drot + 1e-8), -1.0, 1.0)
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
    if action_batch.ndim == 2:
        action_batch = action_batch[:, np.newaxis, :]  # (B, 1, 10)
        squeeze = True
    else:
        squeeze = False
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
    if squeeze:
        out = out[:, 0, :]  # (B, 7)
    return out


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(6,) -> (3,3) rotation matrix."""
    return _ROT6D_TO_MAT.forward(rot6d).reshape(3, 3)
