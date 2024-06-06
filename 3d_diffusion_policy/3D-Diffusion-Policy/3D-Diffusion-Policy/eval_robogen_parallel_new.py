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
    parallel_input_dict['point_cloud'] = np.concatenate([x['point_cloud'][None, ...] for x in list_of_obs], axis=0)
    parallel_input_dict['agent_pos'] = np.concatenate([x['agent_pos'][None, ...] for x in list_of_obs], axis=0)
    parallel_input_dict['feature_map'] = np.concatenate([x['feature_map'][None, ...] for x in list_of_obs], axis=0)
    parallel_input_dict['gripper_pcd'] = np.concatenate([x['gripper_pcd'][None, ...] for x in list_of_obs], axis=0)
    parallel_input_dict['pcd_mask'] = np.concatenate([x['pcd_mask'][None, ...] for x in list_of_obs], axis=0)
    
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
        
        assert demo_experiment_path is not None
            # all_demo_path = os.path.join(os.environ['PROJECT_DIR'], cfg.task.env_runner.demo_experiment_path, "all_demo_path.txt")
            # with open(all_demo_path, "r") as f:
            #     all_demo_path = f.readlines()
            #     all_demo_path = [x.lstrip().rstrip().split("/")[-1] for x in all_demo_path]
            # all_experiments = all_demo_path
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
                    [float(res_info[idx - beg_idx]['initial_joint_angle'][-1]), 
                    expert_opened_angles[idx], 
                    float(initial_info[idx - beg_idx]['initial_joint_angle'])]
                    
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
            
            
def run_eval_non_parallel(cfg, policy, num_worker, save_path, exp_beg_idx=0, exp_end_idx=1000, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None):
    
    cfg.task.env_runner.experiment_folder = [cfg.task.env_runner.experiment_folder]
    cfg.task.env_runner.experiment_name = [cfg.task.env_runner.experiment_name]
    cfg.task.env_runner.demo_experiment_path = [cfg.task.env_runner.demo_experiment_path]
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
    
        after_reaching_init_state_files = []
        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
        if demo_experiment_path is not None:
            demo_experiment_path = demo_experiment_path[demo_experiment_path.find("RoboGen_sim2real/") + len("RoboGen_sim2real/"):]
            all_subfolder = os.listdir(os.path.join(os.environ['PROJECT_DIR'], demo_experiment_path))
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
                if ratio < 0.65:
                    continue
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
        for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
            all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
            with open(all_substeps_path, "r") as f:
                substeps = f.readlines()
                first_step = substeps[0].lstrip().rstrip()
                task_name = first_step.replace(" ", "_")
            
            env, _ = build_up_env(
                    config_file,
                    solution_path,
                    task_name,
                    init_state_file,
                    # render=False, 
                    render=True, 
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
                                                        )
                
            env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                                max_episode_steps=600, reward_agg_method='sum')
            
            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()

            initial_info = info
            all_rgbs = [rgb]
            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                parallel_input_dict['point_cloud'] = parallel_input_dict['point_cloud'].unsqueeze(0)
                parallel_input_dict['agent_pos'] = parallel_input_dict['agent_pos'].unsqueeze(0)
                parallel_input_dict['feature_map'] = parallel_input_dict['feature_map'].unsqueeze(0)
                parallel_input_dict['gripper_pcd'] = parallel_input_dict['gripper_pcd'].unsqueeze(0)
                parallel_input_dict['pcd_mask'] = parallel_input_dict['pcd_mask'].unsqueeze(0)
                

                with torch.no_grad():
                    batched_action = policy.predict_action(parallel_input_dict)
                    
                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']
                
                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                rgb = env.env.render()
                all_rgbs.append(rgb)
            
            env.env._env.close()
            
            opened_joint_angles[config_files[exp_idx]] = \
                [float(info['opened_joint_angle'][-1]), 
                expert_opened_angles[exp_idx], 
                float(initial_info['initial_joint_angle'])]
                
            with open("{}/opened_joint_angles.json".format(save_path), "w") as f:
                json.dump(opened_joint_angles, f, indent=4)
            
            gif_save_exp_name = experiment_folder.split("/")[-2]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)
            gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
                    float(info["improved_joint_angle"][-1]))
            
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)
        
if __name__ == "__main__":
    
    num_worker = 50
    pool = Pool(processes=num_worker)
    checkpoint_name = "latest.ckpt"
    exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0602-act3d-obj-46462-train-ratio-0.2/2024.06.02/19.34.59_train_dp3_robogen_open_door"

    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
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
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    checkpoint_name_start_idx = checkpoint_dir.find("3D-Diffusion-Policy/data/")  + len("3D-Diffusion-Policy/data/")
    save_path = "data/debug/{}".format(checkpoint_dir[checkpoint_name_start_idx:].replace("/", "_"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # exp_beg_idx = 0
    # exp_end_idx = num_worker + num_worker // 2
    exp_beg_ratio = 0
    exp_end_ratio = 0.2
        
    run_eval_non_parallel(cfg, policy, num_worker, save_path, 
            #  exp_beg_idx=exp_beg_idx, 
            #  exp_end_idx=exp_end_idx, 
             pool=pool, 
             horizon=135,
             exp_beg_ratio=exp_beg_ratio,
             exp_end_ratio=exp_end_ratio,
        )
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    # ps.print_stats(50)
    # print(s.getvalue())
    # ps = pstats.Stats(pr, stream=s).sort_stats('time')
    # ps.print_stats(50)
    # print(s.getvalue())