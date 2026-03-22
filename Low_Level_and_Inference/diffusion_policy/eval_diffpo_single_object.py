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


def construct_env(cfg, config_file, solution_path, task_name, init_state_file, 
                  real_world_camera=False, noise_real_world_pcd=False,
                  randomize_camera=False):
    env, _ = build_up_image_env(
                    config_file,
                    solution_path,
                    task_name,
                    init_state_file,
                    render=False, 
                    horizon=600,
            )
            
    object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenImageWrapper(env, object_name, 
                                                observation_mode=cfg.task.env_runner.observation_mode,
                                                real_world_camera=real_world_camera,
                                                noise_real_world_pcd=noise_real_world_pcd,
                                                extract_pcds=args.high_level_ckpt_name is not None
                                            )
        
    if randomize_camera > 0:
        pointcloud_env.reset_random_cameras(randomize_camera)

    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    return env

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

def high_level_policy_infer(parallel_input_dict, high_level_policy, output_obj_pcd_only=True):
    with torch.no_grad():
        # pointcloud = parallel_input_dict['point_cloud'][:, -1, :, :]
        # gripper_pcd = parallel_input_dict['gripper_pcd'][:, -1, :]
        pointcloud = parallel_input_dict['point_cloud'][:, -1, 0, :]
        gripper_pcd = parallel_input_dict['gripper_pcd'][:, -1, 0]
        
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
            
        # if args.add_one_hot_encoding:
        #     # for pointcloud, we add (1, 0)
        #     # for gripper_pcd, we add (0, 1)
        #     pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2).float().to(pointcloud.device)
        #     pointcloud_one_hot[:, :, 0] = 1
        #     pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
        #     gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2).float().to(pointcloud.device)
        #     gripper_pcd_one_hot[:, :, 1] = 1
        #     gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
        #     inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
            
        inputs = inputs.to('cuda')
        inputs_ = inputs.permute(0, 2, 1)
        outputs = high_level_policy(inputs_)
        weights = outputs[:, :, -1] # B, N
        outputs = outputs[:, :, :-1] # B, N, 12
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
    return outputs

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
        
        opened_joint_angles = {}

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        
        ### loop through each test configuration of the object
        for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
            all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
            task_name =  "grasp the handle of the storage furniture door".strip().replace(" ", "_")
            solution_path = 'articulated'
            env = construct_env(cfg, config_file, solution_path, task_name, init_state_file, real_world_camera, noise_real_world_pcd, 
                                randomize_camera)
            
            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()
            all_rgbs = [rgb]
            last_goal = None
            is_gripper_frame = cfg.task.dataset.get('pointmap_frame', 'robot_frame') == 'gripper_frame'

            # for t in tqdm.tqdm(range(1, horizon)):
            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)

                if high_level_policy is not None:
                    predicted_goal = high_level_policy_infer(parallel_input_dict, high_level_policy, output_obj_pcd_only=output_obj_pcd_only)
                    predicted_goal = predicted_goal.repeat(1, 2, 1, 1)
                    parallel_input_dict['goal_gripper_pcd'] = predicted_goal

                with torch.no_grad():
                    parallel_input_dict = filter_obs_keys(parallel_input_dict, cfg.task.shape_meta, is_gripper_frame)
                    # save_pointcloud_video({'obs': parallel_input_dict}, filename='media/eval_pointcloud_360_gripper.mp4', sample_idx=0, timestep=0, downsample_rate=5)
                    # breakpoint()
                    # import h5py
                    # from matplotlib import pyplot as plt
                    # data = h5py.File('data/rgb/41510/2025-10-30-21-05-53.h5')
                    # training_image = data['obs']['rgb'][0,0].astype(np.float32) / 255.0
                    # env_image = parallel_input_dict['cam0_image'][0, -1, :, :, :].permute(1,2,0).cpu().numpy()
                    # fig, axs = plt.subplots(1,2)
                    # axs[0].imshow(training_image)
                    # axs[1].imshow(env_image)
                    # plt.show()
                    # plt.imshow(np.abs(training_image - env_image)); plt.show()

                    batched_action = low_level_policy.predict_action(parallel_input_dict)
                    # save_rgb_video_with_heatmaps({'obs': parallel_input_dict}, filename='media/eval_rgb_heatmap_video.mp4', sample_idx=0, episode_length=horizon)

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

                # all_obs.append(parallel_input_dict); all_actions.append(np_batched_action)
                ### step the environment with the low-level action
                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                # env.env.goal_gripper_pcd = np_predicted_goal.squeeze(0)[0].reshape(4, 3)
                # env.env.heatmap = parallel_input_dict['cam0_heatmap'][0, -1].permute(1,2,0).cpu().numpy() if 'cam0_heatmap' in parallel_input_dict else None
                rgb = env.env.render()
                all_rgbs.append(rgb)
            
            env.env._env.close()

            ### save statistics
            opened_joint_angles[config_file] = \
            {
                "final_door_joint_angle": float(info['opened_joint_angle'][-1]), 
                "expert_door_joint_angle": expert_opened_angles[exp_idx], 
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, default=None)
    parser.add_argument('--low_level_ckpt_name', type=str, default=None)
    parser.add_argument("--eval_exp_name", type=str, default=None)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument("--folder_name", type=str, default='data/rgb_eval')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--high_level_ckpt_name", type=str, default=None)
    args = parser.parse_args()
    
    high_level_policy = None
    if args.high_level_ckpt_name is not None:
        ### load the high-level policy
        load_model_path = args.high_level_ckpt_name    
        num_class = 13 
        # input_channel = 5 if args.add_one_hot_encoding else 3
        input_channel = 3
        from weighted_displacement_model.model_invariant import PointNet2_super
        high_level_policy = PointNet2_super(num_classes=num_class, input_channel=input_channel).to("cuda")
        high_level_policy.load_state_dict(torch.load(load_model_path))
        high_level_policy.eval()

    ### load low-level policy
    exp_dir = args.low_level_exp_dir
    checkpoint_name = args.low_level_ckpt_name

    cfg = OmegaConf.load(f"{exp_dir}/.hydra/config.yaml")
    # workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    # workspace = TrainDiTBlockWorkspace(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir, )
    low_level_policy = deepcopy(workspace.model)
    if OmegaConf.select(workspace.cfg, "training.use_ema", default=False):
        low_level_policy = deepcopy(workspace.ema_model)
    low_level_policy.eval()
    low_level_policy.reset()
    low_level_policy = low_level_policy.to('cuda')

    ### prepare the evaluation environment
    with open_dict(cfg):
        cfg.task.env_runner.experiment_name = ['' for _ in range(10)]
        folder_name = args.folder_name if args.folder_name is not None else 'data/rgb_eval'
        cfg.task.env_runner.experiment_folder = [
            f'{folder_name}/41510'
        ]
        cfg.task.env_runner.demo_experiment_path = [None for _ in range(10)]

    ### dump evaluation configuration
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = "outputs_eval/{}/{}/{}".format("/".join(Path(exp_dir).parts[-2:]),checkpoint_name, timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    action_mode = cfg.get('action_mode', 'hybrid_delta')
    print("Using action mode: ", action_mode)

    randomize_camera = None
    if cfg.task.dataset.data_dir.startswith('data/rgb/'):
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
        "randomize camera mode": randomize_camera,
        "action mode": action_mode,
    }

    checkpoint_info.update(args.__dict__)
    with open("{}/checkpoint_info.json".format(save_path), "w") as f:
        json.dump(checkpoint_info, f, indent=4)

    run_eval_non_parallel(
            cfg, low_level_policy, high_level_policy,
            save_path,
            horizon=35,
            exp_beg_idx=0,
            exp_end_idx=100, # =25,
            real_world_camera=args.real_world_camera,
            randomize_camera=randomize_camera,
            action_mode=action_mode,
    )
