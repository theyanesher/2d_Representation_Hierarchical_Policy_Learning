import os
import hydra
import torch
import dill
from omegaconf import OmegaConf, open_dict
import pathlib
import pybullet as p
import numpy as np
from copy import deepcopy
import sys
from termcolor import cprint
import tqdm
import json
import time
from pathlib import Path
import yaml
import pickle as pkl
import argparse
from datetime import datetime
from typing import List, Optional
from collections import deque
from diffusion_policy.diffusion_policy.workspace.train_dit_block_workspace import TrainDiTBlockWorkspace
from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from manipulation.robogen_image_wrapper import RobogenImageWrapper
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from diffusion_policy.diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_image_env, save_numpy_as_mp4
from diffusion_policy.common.action_util import (
    hybrid_relative_to_hybrid_delta_actions,
    hybrid_delta_to_hybrid_relative_actions,
    delta_to_hybrid_delta_actions,
    relative_to_delta_actions,
    )
from diffusion_policy.diffusion_policy.common.obs_util import filter_obs_keys
from diffusion_policy.common.debug_util import save_pointcloud_video, save_pointmap_visualization, save_rgb_video_with_heatmaps

"""pixi run python diffusion_policy/eval_hierarchical_diffpo_single_object.py --low_level_exp_dir outputs/2026.01.24/14.51.48_diffusion_unet_hybrid_image --low_level_ckpt_name epoch_60.ckpt --high_level_ckpt_name path/to/pointnet2.pth --update_goal_freq 5 --folder_name data/rgb_eval
"""

# ---------------------------------------------------------------------------
# Geometry helpers (mirrored from scripts/project_gripper_to_images.py)
# ---------------------------------------------------------------------------

def _project_world_to_pixel(point_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    """Project a 3-D world point to (u, v, z). Returns None if behind camera."""
    p_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    p_cam = extrinsic @ p_h
    z = p_cam[2]
    if z <= 0:
        return None
    p_img = intrinsic @ p_cam[:3]
    return float(p_img[0] / p_img[2]), float(p_img[1] / p_img[2]), float(z)

def _gaussian_heatmap(H: int, W: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    return np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))

def _compute_heatmap_for_cam(point_world, extrinsic, intrinsic, H, W, sigma):
    """Returns (H, W) float32 Gaussian heatmap, or zeros if point is not visible."""
    result = _project_world_to_pixel(point_world, extrinsic, intrinsic)
    if result is None:
        return np.zeros((H, W), dtype=np.float32)
    u, v, _ = result
    if not (0 <= int(round(u)) < W and 0 <= int(round(v)) < H):
        return np.zeros((H, W), dtype=np.float32)
    return _gaussian_heatmap(H, W, cx=u, cy=v, sigma=sigma)

def generate_goal_heatmaps(predicted_goal, parallel_input_dict, sigma=20.0, num_cams=3):
    """Project the predicted goal gripper position onto each camera as a Gaussian heatmap.

    predicted_goal : torch.Tensor (B, 2, 4, 3) on cuda — gripper keypoints in world coords.
                     Uses the last keypoint (index -1) of the first gripper as the heatmap centre.
    parallel_input_dict : obs dict containing cam{i}_extrinsic (B,T,4,4),
                          cam{i}_intrinsic (B,T,3,3), and cam{i}_image (B,T,H,W,3).

    Returns dict {'cam0_heatmap': tensor(B,T,H,W), 'cam1_heatmap': ..., ...} on cuda, float32.
    The heatmap is replicated across all T obs steps (same goal used for the whole window).
    """
    B = predicted_goal.shape[0]
    # Mean of 4 keypoints for the first gripper as the representative world point
    # point_world = predicted_goal[0, 0, :, :].mean(dim=0).cpu().numpy()  # (3,)
    # Use the last keypoint of the first gripper as the goal position
    point_world = predicted_goal[0, 0, -1, :].cpu().numpy()  # (3,)

    img_shape = parallel_input_dict['cam0_image'].shape  # (B, T, H, W, 3)
    H, W, T = int(img_shape[2]), int(img_shape[3]), int(img_shape[1])

    heatmaps = {}
    for cam in range(num_cams):
        E = parallel_input_dict[f'cam{cam}_extrinsic'][0, -1].cpu().numpy()   # (4, 4)
        K = parallel_input_dict[f'cam{cam}_intrinsic'][0, -1].cpu().numpy()   # (3, 3)
        hm_np = _compute_heatmap_for_cam(point_world, E, K, H, W, sigma)      # (H, W)
        # Replicate to (B, T, 1, H, W) — channel dim matches training dataset convention
        hm = torch.from_numpy(hm_np).to(predicted_goal.device)
        heatmaps[f'cam{cam}_heatmap'] = hm[None, None, None].expand(B, T, 1, H, W).contiguous()
    return heatmaps

# ---------------------------------------------------------------------------

def construct_env(cfg, config_file, solution_path, task_name, init_state_file,
                  real_world_camera=False, noise_real_world_pcd=False,
                  randomize_camera=False,
                  hl_num_points=4500, hl_obs_mode='act3d'):
    """Build the low-level image env and the high-level PCD observation wrapper.

    Both wrappers share the same underlying pybullet env.  Only ll_env drives the
    simulation via .step().  hl_pcd_env is used for read-only observation via
    ._get_observation(), called after each ll_env.step() to get the updated state.
    """
    raw_env, _ = build_up_image_env(
                    config_file,
                    solution_path,
                    task_name,
                    init_state_file,
                    render=False,
                    horizon=600,
            )

    object_name = "StorageFurniture".lower()
    raw_env.reset()

    # Low-level: image-based wrapper — no PCDs needed, high-level uses its own wrapper
    image_env = RobogenImageWrapper(raw_env, object_name,
                                    observation_mode=cfg.task.env_runner.observation_mode,
                                    real_world_camera=real_world_camera,
                                    noise_real_world_pcd=noise_real_world_pcd,
                                    extract_pcds=False)
    if randomize_camera > 0:
        image_env.reset_random_cameras(randomize_camera)

    ll_env = MultiStepWrapper(image_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
                              max_episode_steps=600, reward_agg_method='sum')

    # High-level: point-cloud observation wrapper around the same raw_env
    hl_pcd_env = RobogenPointCloudWrapper(raw_env, object_name,
                                          num_points=hl_num_points,
                                          observation_mode=hl_obs_mode,
                                          real_world_camera=real_world_camera,
                                          noise_real_world_pcd=noise_real_world_pcd)

    return ll_env, hl_pcd_env

def prepare_env(experiment_folder, experiment_path, all_experiments):

    expert_opened_angles = []
    init_state_files = []
    config_files = []
    for experiment in all_experiments:
        if "meta" in experiment:
            continue

        experiment_dir = os.path.join(experiment_path, experiment)
        if not os.path.isdir(experiment_dir):
            continue
        if os.path.exists(os.path.join(experiment_dir, "label.json")):
            with open(os.path.join(experiment_dir, "label.json"), 'r') as f:
                label = json.load(f)
            if not label['good_traj']: continue
        else:
            print(f"Warning: No label.json found in {experiment_dir}, skipping...")
            continue

        first_step_states_path = os.path.join(experiment_dir, "states")
        expert_states = os.listdir(first_step_states_path)
        if len(expert_states) == 0:
            continue

        expert_opened_angle_file = os.path.join(experiment_dir, "opened_angle.txt")
        if os.path.exists(expert_opened_angle_file):
            with open(expert_opened_angle_file, "r") as f:
                angles = f.readlines()
                expert_opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                ratio = expert_opened_angle / max_angle

        expert_opened_angles.append(expert_opened_angle)
        init_state_file = os.path.join(first_step_states_path, "state_0.pkl")
        init_state_files.append(init_state_file)
        config_file = os.path.join(experiment_dir, "task_config.yaml")
        config_files.append(config_file)

    return config_files, init_state_files, expert_opened_angles

def high_level_policy_infer(pcd_obs_dict, high_level_policy, output_obj_pcd_only=True):
    """Run the PointNet2 high-level policy on a single PCD observation.

    pcd_obs_dict: dict with tensors already on cuda and shaped (B, T, N, *):
        'point_cloud': (B, T, N, 3)
        'gripper_pcd': (B, T, 4, 3)
    Uses the last timestep (index -1) for inference.
    No camera dimension — obs come from RobogenPointCloudWrapper, not RobogenImageWrapper.
    """
    with torch.no_grad():
        # Take the last timestep; no camera dim unlike the image wrapper path
        pointcloud = pcd_obs_dict['point_cloud'][:, -1, :, :]  # (B, N, 3)
        gripper_pcd = pcd_obs_dict['gripper_pcd'][:, -1, :]     # (B, 4, 3)

        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)    # (B, N+4, 3)

        inputs = inputs.to('cuda')
        inputs_ = inputs.permute(0, 2, 1)
        outputs = high_level_policy(inputs_)
        weights = outputs[:, :, -1]   # B, N+4
        outputs = outputs[:, :, :-1]  # B, N+4, 12
        if output_obj_pcd_only:
            weights = weights[:, :-4]
            outputs = outputs[:, :-4, :]
            inputs = inputs[:, :-4, :]

        B, N, _ = outputs.shape
        outputs = outputs.view(B, N, 4, 3)

        outputs = outputs + inputs[:, :, :3].unsqueeze(2)
        weights = torch.nn.functional.softmax(weights, dim=1)
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        outputs = outputs.unsqueeze(1)
    return outputs  # (B, 1, 4, 3)

def run_eval_non_parallel(cfg, low_level_policy, high_level_policy,
                          save_path, exp_beg_idx=0,
                          exp_end_idx=1000,
                          horizon=150,
                          exp_beg_ratio=None, exp_end_ratio=None,
                          dataset_index=None,
                          output_obj_pcd_only=False,
                          update_goal_freq=1,
                          real_world_camera=False,
                          noise_real_world_pcd=False,
                          randomize_camera=False,
                          action_mode='delta',
                          hl_num_points=4500,
                          hl_obs_mode='act3d',
                          ):

    ### loop through each test object
    for dataset_idx, (experiment_folder, experiment_name) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name)):
        print(dataset_idx)
        if dataset_index is not None:
            dataset_idx = dataset_index

        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        config_files, init_state_files, expert_opened_angles = prepare_env(experiment_folder, experiment_path, all_experiments)

        # Load existing results if resuming into the same save_path
        existing_json = "{}/opened_joint_angles_{}.json".format(save_path, dataset_idx)
        if os.path.exists(existing_json):
            with open(existing_json, "r") as f:
                opened_joint_angles = json.load(f)
        else:
            opened_joint_angles = {}

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]

        ### loop through each test configuration of the object
        for local_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
            exp_idx = local_idx + exp_beg_idx  # global index for naming/logging

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
            all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
            task_name = "grasp the handle of the storage furniture door".strip().replace(" ", "_")
            solution_path = 'articulated'

            # Both wrappers share the same underlying pybullet env
            env, hl_pcd_env = construct_env(cfg, config_file, solution_path, task_name, init_state_file,
                                            real_world_camera, noise_real_world_pcd, randomize_camera,
                                            hl_num_points=hl_num_points, hl_obs_mode=hl_obs_mode)

            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()
            all_rgbs = [rgb]
            all_heatmap_rgbs = []
            last_predicted_goal = None
            is_gripper_frame = cfg.task.dataset.get('pointmap_frame', 'robot_frame') == 'gripper_frame'

            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)

                ### High-level: predict goal from PCD every update_goal_freq steps
                if high_level_policy is not None:
                    if t == 1 or t % update_goal_freq == 0:
                        # raw_env state is already at the current step (advanced by the
                        # previous ll_env.step()), so just re-observe via the PCD wrapper
                        pcd_obs = hl_pcd_env._get_observation()
                        pcd_obs_dict = {
                            'point_cloud': torch.from_numpy(pcd_obs['point_cloud']).unsqueeze(0).unsqueeze(0).to('cuda'),  # (1, 1, N, 3)
                            'gripper_pcd': torch.from_numpy(pcd_obs['gripper_pcd']).unsqueeze(0).unsqueeze(0).to('cuda'),  # (1, 1, 4, 3)
                        }
                        predicted_goal = high_level_policy_infer(
                            pcd_obs_dict, high_level_policy, output_obj_pcd_only=output_obj_pcd_only)
                        last_predicted_goal = predicted_goal
                    else:
                        predicted_goal = last_predicted_goal

                    predicted_goal = predicted_goal.repeat(1, 2, 1, 1)
                    parallel_input_dict['goal_gripper_pcd'] = predicted_goal

                    # Project goal onto each camera as a Gaussian heatmap
                    goal_heatmaps = generate_goal_heatmaps(predicted_goal, parallel_input_dict)
                    parallel_input_dict.update(goal_heatmaps)

                    # --- Heatmap + goal-point visualization (before filter_obs_keys) ---
                    # goal point in world coords: last keypoint of first gripper
                    goal_pt_world = predicted_goal[0, 0, -1, :].cpu().numpy()  # (3,)
                    vis_frames = []
                    for cam in range(2):  # cam0 and cam1 only (in shape_meta)
                        cam_rgb = parallel_input_dict[f'cam{cam}_image'][0, -1].cpu().numpy()  # (H, W, 3) uint8
                        hm = goal_heatmaps[f'cam{cam}_heatmap'][0, -1, 0].cpu().numpy()        # (H, W) float32 [0,1]
                        H_img, W_img = cam_rgb.shape[:2]
                        # Blend heatmap as green overlay on top of RGB
                        hm_color = np.stack([np.zeros_like(hm), hm, np.zeros_like(hm)], axis=-1) * 255.0  # (H,W,3) green
                        blended = np.clip(0.6 * cam_rgb.astype(np.float32) + 0.4 * hm_color, 0, 255).astype(np.uint8)
                        # Draw the goal point as a red filled square
                        E = parallel_input_dict[f'cam{cam}_extrinsic'][0, -1].cpu().numpy()
                        K = parallel_input_dict[f'cam{cam}_intrinsic'][0, -1].cpu().numpy()
                        proj = _project_world_to_pixel(goal_pt_world, E, K)
                        if proj is not None:
                            u, v, _ = proj
                            cy, cx = int(round(v)), int(round(u))
                            r = 8  # circle radius in pixels
                            ys, xs = np.ogrid[:H_img, :W_img]
                            ring = (((ys - cy)**2 + (xs - cx)**2) <= r**2) & \
                                   (((ys - cy)**2 + (xs - cx)**2) >= (r - 2)**2)
                            blended[ring] = [0, 0, 255]  # blue circle outline
                        vis_frames.append(blended)
                    all_heatmap_rgbs.append(np.concatenate(vis_frames, axis=1))  # (H, 2*W, 3)

                ### Low-level: predict action from image observations
                with torch.no_grad():
                    parallel_input_dict = filter_obs_keys(parallel_input_dict, cfg.task.shape_meta, is_gripper_frame)
                    batched_action = low_level_policy.predict_action(parallel_input_dict)

                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']
                if action_mode == 'hybrid_relative':
                    np_batched_action = hybrid_relative_to_hybrid_delta_actions(np_batched_action)
                elif action_mode == 'delta':
                    np_batched_action = delta_to_hybrid_delta_actions(np_batched_action, parallel_input_dict['state'][:, -1, :].cpu().numpy())
                elif action_mode == 'hybrid_delta':
                    np_batched_action = np_batched_action
                elif action_mode == 'relative':
                    np_batched_action = relative_to_delta_actions(np_batched_action)
                    np_batched_action = delta_to_hybrid_delta_actions(np_batched_action, parallel_input_dict['state'][:, -1, :].cpu().numpy())
                else:
                    raise ValueError(f"Unsupported action mode: {action_mode}")

                ### Step the simulation only through the low-level image env
                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                rgb = env.env.render()
                all_rgbs.append(rgb)

            # Release hl_pcd_env before closing the physics client to avoid
            # double-free from both wrappers sharing the same pybullet instance
            hl_pcd_env = None
            env.env._env.close()

            ### save statistics
            opened_joint_angles[config_file] = \
            {
                "final_door_joint_angle": float(info['opened_joint_angle'][-1]),
                "expert_door_joint_angle": expert_opened_angles[local_idx],
                "initial_joint_angle": float(info['initial_joint_angle'][-1]),
                "ik_failure": float(info['ik_failure'][-1]),
                'grasped_handle': float(info['grasped_handle'][-1]),
                "exp_idx": exp_idx,
            }

            with open("{}/opened_joint_angles_{}.json".format(save_path, dataset_idx), "w") as f:
                json.dump(opened_joint_angles, f, indent=4)

            gif_save_exp_name = experiment_folder.split("/")[-2]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)
            gif_save_path = "{}/{}_{}.mp4".format(gif_save_folder, exp_idx,
                    float(info["improved_joint_angle"][-1]))

            save_numpy_as_mp4(np.array(all_rgbs), gif_save_path)

            if all_heatmap_rgbs:
                heatmap_save_path = gif_save_path.replace('.mp4', '_heatmap.mp4')
                save_numpy_as_mp4(np.array(all_heatmap_rgbs), heatmap_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, default=None)
    parser.add_argument('--low_level_ckpt_name', type=str, default=None)
    parser.add_argument("--eval_exp_name", type=str, default=None)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument("--folder_name", type=str, default='data/rgb_eval')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--high_level_ckpt_name", type=str, default=None)
    parser.add_argument("--update_goal_freq", type=int, default=1,
                        help="Re-run the high-level policy every N low-level steps")
    parser.add_argument("--output_obj_pcd_only", type=int, default=1)
    parser.add_argument("--exp_beg_idx", type=int, default=0)
    parser.add_argument("--exp_end_idx", type=int, default=10000,
                        help="End index (exclusive). Default 10000 = evaluate all.")
    parser.add_argument("--resume_save_path", type=str, default=None,
                        help="Path to existing output folder to resume/append results into.")
    args = parser.parse_args()

    ### Load high-level policy (PointNet2 regression model)
    high_level_policy = None
    if args.high_level_ckpt_name is not None:
        load_model_path = args.high_level_ckpt_name
        num_class = 13
        input_channel = 3
        from weighted_displacement_model.model_invariant import PointNet2_super
        high_level_policy = PointNet2_super(num_classes=num_class, input_channel=input_channel).to("cuda")
        high_level_policy.load_state_dict(torch.load(load_model_path))
        high_level_policy.eval()

    ### Load low-level policy (image-based diffusion policy)
    exp_dir = args.low_level_exp_dir
    checkpoint_name = args.low_level_ckpt_name

    cfg = OmegaConf.load(f"{exp_dir}/.hydra/config.yaml")
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir)
    low_level_policy = deepcopy(workspace.model)
    if OmegaConf.select(workspace.cfg, "training.use_ema", default=False):
        low_level_policy = deepcopy(workspace.ema_model)
    low_level_policy.eval()
    low_level_policy.reset()
    low_level_policy = low_level_policy.to('cuda')

    ### Prepare the evaluation environment config
    with open_dict(cfg):
        cfg.task.env_runner.experiment_name = ['' for _ in range(10)]
        folder_name = args.folder_name if args.folder_name is not None else 'data/rgb_eval'
        cfg.task.env_runner.experiment_folder = [
            f'{folder_name}/41510'
        ]
        cfg.task.env_runner.demo_experiment_path = [None for _ in range(10)]

    ### Dump evaluation configuration
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    if args.resume_save_path is not None:
        save_path = args.resume_save_path
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_path = "outputs_eval/{}/{}/{}".format("/".join(Path(exp_dir).parts[-2:]), checkpoint_name, timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    action_mode = cfg.get('action_mode', 'hybrid_delta')
    print("Using action mode: ", action_mode)
    # import pdb; pdb.set_trace()
    randomize_camera = None
    if cfg.task.dataset.data_dir.startswith('data/rgb/') or cfg.task.dataset.data_dir.startswith('/scratch/pbhowal/Articubot_Data_For_DP_and_Groot/Heatmap_Articubot_Dataset/'):
        randomize_camera = 0
    elif cfg.task.dataset.data_dir.startswith('data/rgb_camera_left_right_randomized/'):
        randomize_camera = 1
    elif cfg.task.dataset.data_dir.startswith('data/rgb_camera_randomized/'):
        randomize_camera = 2
    else:
        raise ValueError(f"Unsupported dataset: {cfg.task.dataset.data_dir}")
    print("Using randomize_camera: ", randomize_camera)

    checkpoint_info = {
        "low_level_policy": checkpoint_dir,
        "low_level_policy_checkpoint": checkpoint_name,
        "high_level_policy_checkpoint": args.high_level_ckpt_name,
        "update_goal_freq": args.update_goal_freq,
        "randomize_camera_mode": randomize_camera,
        "action_mode": action_mode,
    }
    checkpoint_info.update(args.__dict__)
    with open("{}/checkpoint_info.json".format(save_path), "w") as f:
        json.dump(checkpoint_info, f, indent=4)

    run_eval_non_parallel(
            cfg, low_level_policy, high_level_policy,
            save_path,
            horizon=35,
            exp_beg_idx=args.exp_beg_idx,
            exp_end_idx=args.exp_end_idx,
            real_world_camera=args.real_world_camera,
            randomize_camera=randomize_camera,
            action_mode=action_mode,
            output_obj_pcd_only=bool(args.output_obj_pcd_only),
            update_goal_freq=args.update_goal_freq,
    )

"""
pixi run python diffusion_policy/eval_hierarchical_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.03.19/18.18.35_diffusion_unet_ddp_hybrid \
  --low_level_ckpt_name epoch_95.ckpt \
  --high_level_ckpt_name outputs/High_Level.../data/rgb_eval8.pth \
  --update_goal_freq 5 \
  --folder_name data/rgb_eval \
  --exp_beg_idx 100 \
  --exp_end_idx 10000 \
  --resume_save_path outputs_eval/2026.03.19/18.18.35_diffusion_unet_ddp_hybrid/epoch_95.ckpt/2026-03-21_12-59

"""