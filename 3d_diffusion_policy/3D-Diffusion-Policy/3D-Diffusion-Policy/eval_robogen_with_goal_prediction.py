import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif, save_env
import pybullet as p
import numpy as np
from copy import deepcopy
import sys
from termcolor import cprint
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from manipulation.gpt_reward_api import get_joint_state
import tqdm
import json
from multiprocessing import set_start_method
from multiprocessing import Pool
import time
import yaml
import pickle as pkl
from manipulation.utils import get_pc, get_pc_in_camera_frame, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D, add_sphere, get_pixel_location
import cv2

def construct_env(cfg, config_file, solution_path, task_name, init_state_file):
    env, _ = build_up_env(
                    config_file,
                    solution_path,
                    task_name,
                    init_state_file,
                    # render=False, 
                    render=False, 
                    randomize=False,
                    obj_id=0,
                    horizon=600,
                    random_object_translation=False
            )
            
    object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                num_points=cfg.task.env_runner.num_point_in_pc,
                                                use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                use_segmask=cfg.task.env_runner.use_segmask,
                                                only_handle_points=cfg.task.env_runner.only_handle_points,
                                                observation_mode=cfg.task.env_runner.observation_mode,
                                                dense_pcd_for_goal=cfg.task.env_runner.dense_pcd_for_goal,
                                                )
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    return env

def parallel_eval(args):
    config_path, init_state, action, cfg, idx = args 
    config_file = config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
    all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()
        task_name = first_step.replace(" ", "_")
    
    env, _ = build_up_env(
            config_path,
            solution_path,
            task_name,
            None,
            render=False, 
            randomize=False,
            obj_id=0,
            horizon=600,
    )
    
    object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                  gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                  num_points=cfg.task.env_runner.num_point_in_pc,
                                                  use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                  use_segmask=cfg.task.env_runner.use_segmask,
                                                  only_handle_points=cfg.task.env_runner.only_handle_points,
                                                  observation_mode=cfg.task.env_runner.observation_mode,
                                                  only_object=cfg.task.env_runner.only_object,
                                                  dense_pcd_for_goal=cfg.task.env_runner.dense_pcd_for_goal,
                                                  )
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    env.reset(reset_state=init_state)
    obs, reward, done, info = env.step(action)
    rgb = env.env.render()
    state = save_env(env.env._env)
        
    pointcloud_env._env.close()
    return obs, rgb, info, state, idx

def parallel_reset(args):
    config_path, init_state_file, cfg, idx = args 
    config_file = config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
    all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()
        task_name = first_step.replace(" ", "_")
    
    env, _ = build_up_env(
            config_path,
            solution_path,
            task_name,
            init_state_file,
            render=False, 
            # render=True, 
            randomize=False,
            obj_id=0,
            horizon=600,
    )
    
    object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                  gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                  num_points=cfg.task.env_runner.num_point_in_pc,
                                                  use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                  use_segmask=cfg.task.env_runner.use_segmask,
                                                  only_handle_points=cfg.task.env_runner.only_handle_points,
                                                  observation_mode=cfg.task.env_runner.observation_mode,
                                                  only_object=cfg.task.env_runner.only_object,
                                                  )
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    obs = env.reset()
    # import pdb; pdb.set_trace()
    state = save_env(env.env._env)
    rgb = env.env.render()
    info = env.env._env._get_info()
    env.env._env.close()
    return obs, state, rgb, info, idx

def parallel_save_gif(args):
    rgbs, save_path = args
    
    save_numpy_as_gif(np.array(rgbs), save_path)

def wrap_obs(list_of_obs):
    parallel_input_dict = {}
    # parallel_input_dict['point_cloud'] = np.concatenate([x['point_cloud'][None, ...] for x in list_of_obs], axis=0)
    # parallel_input_dict['agent_pos'] = np.concatenate([x['agent_pos'][None, ...] for x in list_of_obs], axis=0)
    # parallel_input_dict['feature_map'] = np.concatenate([x['feature_map'][None, ...] for x in list_of_obs], axis=0)
    # parallel_input_dict['gripper_pcd'] = np.concatenate([x['gripper_pcd'][None, ...] for x in list_of_obs], axis=0)
    # parallel_input_dict['pcd_mask'] = np.concatenate([x['pcd_mask'][None, ...] for x in list_of_obs], axis=0)
    # # TODO: add goal key
    # if 'goal_gripper_pcd' in list_of_obs[0]:
    #     parallel_input_dict['goal_gripper_pcd'] = np.concatenate([x['goal_gripper_pcd'][None, ...] for x in list_of_obs], axis=0)
    for key in list_of_obs[0]:
        parallel_input_dict[key] = np.concatenate([x[key][None, ...] for x in list_of_obs], axis=0)
    
    parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
    return parallel_input_dict

def run_eval(cfg, policy, num_worker, save_path, exp_beg_idx=0, exp_end_idx=1000, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None):
    # if type(cfg.task.env_runner.experiment_folder) != list:
    #     cfg.task.env_runner.experiment_folder = [cfg.task.env_runner.experiment_folder]
    # if type(cfg.task.env_runner.experiment_name) != list:
    #     cfg.task.env_runner.experiment_name = [cfg.task.env_runner.experiment_name]
    # if type(cfg.task.env_runner.demo_experiment_path) != list:
    #     cfg.task.env_runner.demo_experiment_path = [cfg.task.env_runner.demo_experiment_path]
    
    # import pdb; pdb.set_trace()
    opened_joint_angles = {}
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
        # experiment_folder = cfg.task.env_runner.experiment_folder
        # experiment_name = cfg.task.env_runner.experiment_name
        # import pdb; pdb.set_trace()
    
        after_reaching_init_state_files = []
        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
        if  demo_experiment_path is not None:
            # all_demo_path = os.path.join(os.environ['PROJECT_DIR'], cfg.task.env_runner.demo_experiment_path, "all_demo_path.txt")
            # with open(all_demo_path, "r") as f:
            #     all_demo_path = f.readlines()
            #     all_demo_path = [x.lstrip().rstrip().split("/")[-1] for x in all_demo_path]
            # all_experiments = all_demo_path
            # demo_experiment_path = demo_experiment_path[demo_experiment_path.find("RoboGen_sim2real/") + len("RoboGen_sim2real/"):]
            # if '/scratch' in demo_experiment_path:
            #     demo_experiment_path = demo_experiment_path.replace("/scratch/yufei", "/project_data/held/yufeiw2/RoboGen_sim2real/data")
            all_subfolder = os.listdir(demo_experiment_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            all_experiments = all_subfolder
            
        all_substeps_path = os.path.join(experiment_folder, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()
        

        expert_opened_angles = []
        for experiment in all_experiments:
            if "meta" in experiment:
                continue
            
            first_step_folder = first_step.replace(" ", "_") + "_primitive"
            first_step_folder = os.path.join(experiment_path, experiment, first_step_folder)
            # if os.path.exists(os.path.join(first_step_folder, "label.json")):
            #     with open(os.path.join(first_step_folder, "label.json"), 'r') as f:
            #         label = json.load(f)
            #     if not label['good_traj']: continue
                
            first_stage_states_path = os.path.join(first_step_folder, "states")
            expert_states = os.listdir(first_stage_states_path)
            if len(expert_states) == 0:
                continue
                
            expert_opened_angle_file = os.path.join(experiment_path, experiment, first_step_folder, "opened_angle.txt")
            if os.path.exists(expert_opened_angle_file):
                with open(expert_opened_angle_file, "r") as f:
                    angles = f.readlines()
                    expert_opened_angle = float(angles[0].lstrip().rstrip())
                    max_angle = float(angles[-1].lstrip().rstrip())
                    ratio = expert_opened_angle / max_angle
                # if ratio < 0.65:
                #     continue
            expert_opened_angles.append(expert_opened_angle)
            
            first_stage_states_path = os.path.join(first_step_folder, "states")
            stage_lengths = os.path.join(first_step_folder, "stage_lengths.json")
            with open(stage_lengths, "r") as f:
                stage_lengths = json.load(f)
            
            if 'stage' in stage_lengths:
                reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
            else:
                reaching_phase = stage_lengths['reach_handle']
                
            after_init_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(reaching_phase))
            after_reaching_init_state_files.append(after_init_state_file)
            init_state_file = os.path.join(first_stage_states_path, "state_0.pkl")
            init_state_files.append(init_state_file)
            # config_file = os.path.join(experiment_path, experiment, "task_config_added_distractors.yaml")
            config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
            config_files.append(config_file)
                    
        after_reaching_init_state_files = after_reaching_init_state_files
        config_files = config_files

        horizon = horizon

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        num_iters = (len(config_files) - 1) // num_worker + 1
        for iter in range(num_iters):
            
            beg_idx = iter * num_worker
            end_idx = min((iter + 1) * num_worker, len(config_files))

            # first do reset of all envs
            args_to_run = [
                [config_files[idx], init_state_files[idx], cfg, idx] for idx in range(beg_idx, end_idx)
            ]
            results = pool.map(parallel_reset, args_to_run)
            # parallel_reset(args_to_run[0])
            results = sorted(results, key=lambda x: x[-1])
            res_obs = [res[0] for res in results]
            batched_states = [res[1] for res in results]
            batched_rgbs = [res[2] for res in results]
            batched_infos = [res[3] for res in results]
            batched_obs = wrap_obs(res_obs)
            with torch.no_grad():
                batched_action = policy.predict_action(batched_obs)
            np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
            np_batched_action = np_batched_action['action']
            
            initial_info = batched_infos
            all_rgbs = [batched_rgbs]
            max_door_joint_angles = np.ones(len(args_to_run)) * -1
            ik_failures = np.zeros(len(args_to_run))
            grasped_handles = np.zeros(len(args_to_run))
            
            for t_idx in tqdm.tqdm(range(1, horizon)):
                args_to_run = [
                    [config_files[idx], batched_states[idx - beg_idx], np_batched_action[idx - beg_idx], cfg, idx] for idx in range(beg_idx, end_idx)
                ]    
                beg = time.time()
                results = pool.map(parallel_eval, args_to_run)
                results = sorted(results, key=lambda x: x[-1])
                res_obs = [res[0] for res in results]
                res_rgb = [res[1] for res in results]
                res_info = [res[2] for res in results]
                res_states = [res[3] for res in results]
                end = time.time()
                door_joint_angles_step = np.array([float(info['initial_joint_angle'][-1]) for info in res_info])
                max_door_joint_angles = np.maximum(max_door_joint_angles, door_joint_angles_step)
                grasped_handle_step = np.array([float(info['grasped_handle'][-1]) for info in res_info])
                grasped_handles = np.logical_or(grasped_handles, grasped_handle_step)
                ik_failure_step = np.array([float(info['ik_failure'][-1]) for info in res_info])
                ik_failures = np.logical_or(ik_failures, ik_failure_step)
                
                # cprint("step time: {}".format(end - beg), "red")
                
                beg = time.time()
                batched_states = res_states
                batched_obs = wrap_obs(res_obs)
                with torch.no_grad():
                    batched_action = policy.predict_action(batched_obs)
                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']
                end = time.time()
                # cprint("predict time: {}".format(end - beg), "red")
                
                all_rgbs.append(res_rgb)

            for idx in range(beg_idx, end_idx):
                opened_joint_angles[config_files[idx]] = \
                    {
                        "max_door_joint_angle": max_door_joint_angles[idx - beg_idx],
                        "final_door_joint_angle": float(res_info[idx - beg_idx]['initial_joint_angle'][-1]), 
                        "expert_door_joint_angle": expert_opened_angles[idx], 
                        "initial_joint_angle": float(initial_info[idx - beg_idx]['initial_joint_angle']),
                        "ik_failure": float(ik_failures[idx - beg_idx]),
                        'grasped_handle': float(grasped_handles[idx - beg_idx]),
                    }
                    
                with open("{}/opened_joint_angles.json".format(save_path), "w") as f:
                    json.dump(opened_joint_angles, f, indent=4)
            
            gif_save_exp_name = experiment_folder.split("/")[-2]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)

            args_to_run = [
                [
                    [per_step_rgbs[idx] for per_step_rgbs in all_rgbs], 
                    "{}/{}_{}.gif".format(gif_save_folder, idx + beg_idx, float(res_info[idx]['initial_joint_angle'][-1]) - float(initial_info[idx]['initial_joint_angle']))
                ] for idx in range(end_idx - beg_idx)
            ]
            pool.map(parallel_save_gif, args_to_run)
            
            
def run_eval_non_parallel(cfg, policy, goal_cfg, goal_policy, 
                          num_worker, save_path, exp_beg_idx=0, exp_end_idx=1000, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None, dataset_index=None, calculate_distance_from_gt=False):
    
    if calculate_distance_from_gt:
        all_obj_distances = []
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
        
        if dataset_index is not None:
            dataset_idx = dataset_index

        

        # if dataset_idx == 0:
        #     continue

        after_reaching_init_state_files = []
        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
        if demo_experiment_path is not None:
            # demo_experiment_path = demo_experiment_path[demo_experiment_path.find("RoboGen_sim2real/") + len("RoboGen_sim2real/"):]
            all_subfolder = os.listdir(demo_experiment_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            all_experiments = all_subfolder
            
        all_substeps_path = os.path.join(experiment_folder, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()
        

        expert_opened_angles = []
        for experiment in all_experiments:
            if "meta" in experiment:
                continue
            
            first_step_folder = first_step.replace(" ", "_") + "_primitive"
            first_step_folder = os.path.join(experiment_path, experiment, first_step_folder)
            if os.path.exists(os.path.join(first_step_folder, "label.json")):
                with open(os.path.join(first_step_folder, "label.json"), 'r') as f:
                    label = json.load(f)
                if not label['good_traj']: continue
                
            first_stage_states_path = os.path.join(first_step_folder, "states")
            expert_states = os.listdir(first_stage_states_path)
            if len(expert_states) == 0:
                continue
                
            expert_opened_angle_file = os.path.join(experiment_path, experiment, first_step_folder, "opened_angle.txt")
            if os.path.exists(expert_opened_angle_file):
                with open(expert_opened_angle_file, "r") as f:
                    angles = f.readlines()
                    expert_opened_angle = float(angles[0].lstrip().rstrip())
                    max_angle = float(angles[-1].lstrip().rstrip())
                    ratio = expert_opened_angle / max_angle
                # if ratio < 0.65:
                #     continue
            expert_opened_angles.append(expert_opened_angle)
            
            first_stage_states_path = os.path.join(first_step_folder, "states")
            stage_lengths = os.path.join(first_step_folder, "stage_lengths.json")
            with open(stage_lengths, "r") as f:
                stage_lengths = json.load(f)
            
            if 'stage' in stage_lengths:
                reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
            else:
                reaching_phase = stage_lengths['reach_handle']
                
            after_init_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(reaching_phase))
            after_reaching_init_state_files.append(after_init_state_file)
            init_state_file = os.path.join(first_stage_states_path, "state_0.pkl")
            init_state_files.append(init_state_file)
            config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
            config_files.append(config_file)
                    
        after_reaching_init_state_files = after_reaching_init_state_files
        config_files = config_files

        opened_joint_angles = {}

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        # import pdb; pdb.set_trace()
        all_distances = []
        all_grasp_distances = []

        for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
            all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
            with open(all_substeps_path, "r") as f:
                substeps = f.readlines()
                first_step = substeps[0].lstrip().rstrip()
                task_name = first_step.replace(" ", "_")
            
            env = construct_env(cfg, config_file, solution_path, task_name, init_state_file)
            
            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()

            initial_info = info
            all_rgbs = [rgb]
            closed=False
            for t in range(1, horizon):
                parallel_input_dict = obs
                # import pdb; pdb.set_trace()
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                
                
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                
                with torch.no_grad():
                    predicted_goal = goal_policy.predict_action(parallel_input_dict)
                np_predicted_goal = dict_apply(predicted_goal, lambda x: x.detach().to('cpu').numpy())
                np_predicted_goal = np_predicted_goal['action']
                
                parallel_input_dict['goal_gripper_pcd'] = predicted_goal['action'][:, :2, :].view(1, 2, 4, 3)
                if cfg.task.env_runner.dense_pcd_for_goal:
                    parallel_input_dict['point_cloud'] = parallel_input_dict['dense_point_cloud']
                with torch.no_grad():
                    batched_action = policy.predict_action(parallel_input_dict)
                    
                    
                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']

                # if np_batched_action[0, 1, 9] < -0.004:
                #     closed = True

                # if closed:
                #     np_batched_action[:, :, 9] = -0.04
                
                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                if calculate_distance_from_gt:
                    predicted_goal = np_predicted_goal.squeeze(0)[1].reshape(4, 3)
                    gt_goal = env.env.goal_gripper_pcd
                    distance = np.linalg.norm(predicted_goal - gt_goal, axis=1).mean()
                    all_distances.append(distance)
                    grasp_distance = np.linalg.norm(predicted_goal[-1] - gt_goal[-1])
                    all_grasp_distances.append(grasp_distance)
                    rgb = env.env.render()
                    for point in predicted_goal:
                        pixel_x, pixel_y, _ = get_pixel_location(env.env._env.projection_matrix, env.env._env.view_matrix, point, env.env._env.camera_width, env.env._env.camera_height)
                        color = (255, 0, 0)
                        thickness = 2
                        radius = 3
                        image = cv2.circle(rgb, (pixel_x, pixel_y), radius, color, thickness)
                        # save image
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(f'data/debug/{dataset_idx}_{exp_idx}.png', image)

                    break
                env.env.goal_gripper_pcd = np_predicted_goal.squeeze(0)[0].reshape(4, 3)
                rgb = env.env.render()
                all_rgbs.append(rgb)
            
            env.env._env.close()

            if calculate_distance_from_gt:
                continue
            
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
            gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
                    float(info["improved_joint_angle"][-1]))
            
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)

        if calculate_distance_from_gt:
            print("average distance: {}".format(np.mean(all_distances)))
            print("average grasp distance: {}".format(np.mean(all_grasp_distances)))
            all_obj_distances.append(np.mean(all_distances))

    if calculate_distance_from_gt:
        print("average distance over all objects: {}".format(np.mean(all_obj_distances)))
        
if __name__ == "__main__":
    
    num_worker = 30
    # pool = Pool(processes=num_worker)
    pool=None
    
    # load the low-level reaching policy
    ### with goal gripper, with self attention, fixed order bug in attention
    # checkpoint_name = 'epoch-300.ckpt'
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0701-ddp-obj-45448-hor-8-train-ep-260-gripper-goal-w-gripper-displacement-to-closest-objpoint-self-attention-correct-order/2024.07.01/18.35.59_train_dp3_robogen_open_door"
    
    # 10 objects
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.03/19.08.43_train_dp3_robogen_open_door"
    # checkpoint_name = 'latest.ckpt'

    # 50 objects
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door"
    # checkpoint_name = 'latest.ckpt'

    # 200 objects
    exp_dir = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08141110-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-normalize_action-30/2024.08.14/11.10.47_train_dp3_robogen_open_door'
    checkpoint_name = 'latest.ckpt'

    # # # low 50 objects dense pcd around goal
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08191250-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-dense_gripper-combine-pcd_noise-new/2024.08.19/12.50.15_train_dp3_robogen_open_door"
    # checkpoint_name = 'latest.ckpt' 

    # # low 200 objects trained with predicted goal
    exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08240109-act3d_goal_mlp_displacement_gripper_to_object-ns-2-h-8-demonum-75-all-pt_goal/2024.08.24/01.09.48_train_dp3_robogen_open_door"
    checkpoint_name = 'latest.ckpt' 
    checkpoint_name = 'epoch-30.ckpt' 

    exp_dir = "/home/mino/Software/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door"
    checkpoint_name = 'latest.ckpt'

    # 200 objects
    # exp_dir = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08141110-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-normalize_action-30/2024.08.14/11.10.47_train_dp3_robogen_open_door'
    # checkpoint_name = 'latest.ckpt'


    # # low 50 objects dense pcd around goal
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08191250-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-dense_gripper-combine-pcd_noise-new/2024.08.19/12.50.15_train_dp3_robogen_open_door"
    # checkpoint_name = 'latest.ckpt' 

    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
    cfg = recomposed_config
    cfg.use_pretrained_high_level_policy_as_low_level_input = False # because that is only for training

    
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir)

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    policy = policy.to('cuda')

    
    cfg.task.env_runner.experiment_name = ['0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first' for _ in range(10)]
    cfg.task.env_runner.experiment_folder = [
        'data/diverse_objects/open_the_door_40147/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_44817/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_44962/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45132/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45219/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45243/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45332/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45378/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45384/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45463/task_open_the_door_of_the_storagefurniture_by_its_handle'
        ]
    # import pdb; pdb.set_trace()
    cfg.task.env_runner.demo_experiment_path = [None for _ in range(10)]

    # current best model
    goal_checkpoint_name = 'epoch-30.ckpt'
    goal_exp_dir = '/project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0807-200-obj-pred-goal-gripper-PointNet2-backbone-UNet-diffusion-ep-75-epsilon/2024.08.07/14.03.40_train_dp3_robogen_open_door'


    # goal_checkpoint_name = 'latest.ckpt'
    # goal_exp_dir = "/project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0812-200-obj-pred-goal-gripper-mlp-self-attn-backbone-UNet-diffusion-ep-75/2024.08.12/09.56.36_train_dp3_robogen_open_door"

    # goal_checkpoint_name = 'latest.ckpt'
    # goal_exp_dir = "/project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0817-200-obj-pred-weighted-displacement-pointnet2-augmentation-pcd/2024.08.18/18.16.24_train_dp3_robogen_open_door"

    goal_checkpoint_name = 'epoch-30.ckpt'
    goal_exp_dir = "/home/mino/Software/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0807-200-obj-pred-goal-gripper-PointNet2-backbone-UNet-diffusion-ep-75-epsilon/2024.08.07/14.03.40_train_dp3_robogen_open_door"

    # goal_checkpoint_name = 'epoch-12.ckpt'
    # goal_exp_dir = "/project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0823-200-obj-pred-goal-gripper-pointnet-weighted-diffusion-epsilon-augmentation-pcd/2024.08.23/19.45.27_train_dp3_robogen_open_door"

    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(goal_exp_dir)),
        )
    goal_cfg = recomposed_config
    # goal_cfg.policy.act3d_encoder_cfg.use_attn_for_point_features = "large_self_attention"
    print("use_attn_for_point_features: {}".format(goal_cfg.policy.act3d_encoder_cfg.use_attn_for_point_features))
    
    goal_workspace = TrainDP3Workspace(goal_cfg)
    goal_checkpoint_dir = "{}/checkpoints/{}".format(goal_exp_dir, goal_checkpoint_name)
    goal_workspace.load_checkpoint(path=goal_checkpoint_dir)

    goal_policy = deepcopy(goal_workspace.model)
    if goal_workspace.cfg.training.use_ema:
        goal_policy = deepcopy(goal_workspace.ema_model)
    goal_policy.eval()
    goal_policy.reset()
    goal_policy = goal_policy.to('cuda')
    
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    checkpoint_name_start_idx = checkpoint_dir.find("3D-Diffusion-Policy/data/")  + len("3D-Diffusion-Policy/data/")
    
    for run_idx in range(1):
        save_path = "data/eval_low_50_pcd_200_high_level_weighted_diffusion_1/{}/{}".format(checkpoint_dir[checkpoint_name_start_idx:].replace("/", "_"), run_idx)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
        cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"
        run_eval_non_parallel(
                cfg, policy, goal_cfg, goal_policy, 
                num_worker, save_path, 
                pool=pool, 
                horizon=35,
                exp_beg_idx=0,
                exp_end_idx=25,
                # dataset_index=9，
                # exp_beg_ratio=0.9,
                # exp_end_ratio=1,
                # calculate_distance_from_gt=True,
        )
