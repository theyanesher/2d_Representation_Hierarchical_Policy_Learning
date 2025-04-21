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
            
def get_all_configs(cfg, exp_beg_idx=0, exp_end_idx=1000, exp_beg_ratio=None, exp_end_ratio=None, post_fix=''):
    
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
    
        after_reaching_init_state_files = []
        init_state_files = []
        grasping_state_files = []
        final_state_files = []
        config_files = []
        all_stage_lengths = []
        first_step_folders = []
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
            first_step_folders.append(first_step_folder)
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

            expert_opened_angles.append(expert_opened_angle)
            
            first_stage_states_path = os.path.join(first_step_folder, "states")
            stage_lengths = os.path.join(first_step_folder, "stage_lengths.json")
            with open(stage_lengths, "r") as f:
                stage_lengths = json.load(f)
            
            if 'stage' in stage_lengths:
                reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
            else:
                reaching_phase = stage_lengths['reach_handle']
                
            all_stage_lengths.append(stage_lengths)
                
            after_init_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(reaching_phase))
            after_reaching_init_state_files.append(after_init_state_file)
            init_state_file = os.path.join(first_stage_states_path, "state_0.pkl")
            init_state_files.append(init_state_file)
            
            open_begin_t_idx = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact'] + stage_lengths['close_gripper']
            grasping_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(open_begin_t_idx))
            grasping_state_files.append(grasping_state_file)
            
            total_length = len(os.listdir(first_stage_states_path))
            final_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(total_length - 1))
            final_state_files.append(final_state_file)
            
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
        all_stage_lengths = all_stage_lengths[exp_beg_idx:exp_end_idx]
        final_state_files = final_state_files[exp_beg_idx:exp_end_idx]
        grasping_state_files = grasping_state_files[exp_beg_idx:exp_end_idx]
        first_step_folders = first_step_folders[exp_beg_idx:exp_end_idx]

        
    return config_files, init_state_files, final_state_files, grasping_state_files, first_step_folders
        
def save_robot_eef_poses(config_files, init_state_files, final_state_files, grasping_state_files, first_step_folders):
    for exp_idx, (config_file, init_state_file, grasping_state_file, final_state_file, first_step_folder) in enumerate(zip(
            config_files, init_state_files, grasping_state_files, final_state_files, first_step_folders)):
            
        print("procesing config file: {}".format(config_file))
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
                render=False, 
                randomize=False,
                obj_id=0,
                horizon=600,
                mobile=False,
        )
        
        object_name = "StorageFurniture".lower()
        
        mobile_state_folder = os.path.join(first_step_folder, "mobile_states")
        if not os.path.exists(mobile_state_folder):
            os.makedirs(mobile_state_folder)
        
        for state_file in [init_state_file, grasping_state_file, final_state_file]:
            with open(state_file, 'rb') as f:
                state = pkl.load(f)
            env.reset(reset_state=state)
            eef_pos, eef_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
            t = state_file.split("/")[-1].split(".")[0].split("_")[-1]
            eef_pose_save_file = os.path.join(mobile_state_folder, f"eef_pose_{t}.pkl")
            print("saving eef pose to {}".format(eef_pose_save_file))
            with open(eef_pose_save_file, 'wb') as f:
                pkl.dump((eef_pos, eef_orient), f)
                
        env.close()
        
def save_mobile_base_initial_joint_angles(config_files, init_state_files, final_state_files, grasping_state_files, first_step_folders):
    for exp_idx, (config_file, init_state_file, grasping_state_file, final_state_file, first_step_folder) in enumerate(zip(
        config_files, init_state_files, grasping_state_files, final_state_files, first_step_folders)):
            
        print("procesing config file: {}".format(config_file))
        
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
                render=False, 
                randomize=False,
                obj_id=0,
                horizon=600,
                mobile=True,
        )
        
        env.reset()
        object_name = "StorageFurniture".lower()
        
        
        mobile_state_folder = os.path.join(first_step_folder, "mobile_states")
        # print("trying to make a folder at {}".format(mobile_state_folder))
        if not os.path.exists(mobile_state_folder):
            os.makedirs(mobile_state_folder)
        
        for s_idx, state_file in enumerate([init_state_file, grasping_state_file, final_state_file]):
            t = state_file.split("/")[-1].split(".")[0].split("_")[-1]
            eef_pose_save_file = os.path.join(mobile_state_folder, f"eef_pose_{t}.pkl")
            with open(eef_pose_save_file, 'rb') as f:
                eef_pos, eef_orient = pkl.load(f)
            ik_indices = [i for i in range(len(env.robot.right_arm_joint_indices))]
            ik_joint_angles = env.robot.ik(env.robot.right_end_effector, eef_pos, eef_orient, ik_indices=ik_indices, max_iterations=10000, residualThreshold=1e-4)
            p.addUserDebugPoints([eef_pos], [[1, 0, 0]], 10)
            # import pdb; pdb.set_trace()
            env.robot.set_joint_angles(env.robot.right_arm_joint_indices, ik_joint_angles)
            # import pdb; pdb.set_trace()
            if s_idx == 1:
                close_joint_angle = 0.0
                for _ in range(40):
                    env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [close_joint_angle, close_joint_angle], set_instantly=False)
                    p.stepSimulation()
            
            # get the parent folder of config_file
            config_file_parent_folder = os.path.dirname(config_file)
            mobile_config_file = os.path.join(config_file_parent_folder, "mobile_config.yaml")
            os.system(f"cp {config_file} {mobile_config_file}")
            with open(mobile_config_file, 'r') as f:
                mobile_config = yaml.safe_load(f)
            
            for obj in mobile_config:
                if "initial_joint_angles" in obj:
                    obj["initial_joint_angles"] = str(tuple(list(ik_joint_angles)))
                    break
                
            with open(mobile_config_file, 'w') as f:
                yaml.dump(mobile_config, f)
                
            save_env(env, os.path.join(mobile_state_folder, "state_{}.pkl".format(t)))
                
        env.close()
            
            
if __name__ == "__main__":
    ### chialiang's best low-level model trained on 10 objects
    checkpoint_name = 'latest.ckpt'
    exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.03/19.08.43_train_dp3_robogen_open_door"
    
    with hydra.initialize(config_path='../../3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
    cfg = recomposed_config
    
    # all training objects
    cfg.task.env_runner.demo_experiment_path = [
        "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/data/dp3_demo/0527-act3d-always-close",
    ]
    cfg.task.env_runner.experiment_folder = [
        "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle"
    ]
    cfg.task.env_runner.experiment_name = ["0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"]
    
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    checkpoint_name_start_idx = checkpoint_dir.find("3D-Diffusion-Policy/data/")  + len("3D-Diffusion-Policy/data/")
    
    exp_beg_ratio = 0.9
    exp_end_ratio = 1
        
    config_files, init_state_files, final_state_files, grasping_state_files, first_step_folders = get_all_configs(cfg, 
            exp_beg_ratio=exp_beg_ratio,
            exp_end_ratio=exp_end_ratio,
    )
    
    # save_robot_eef_poses(config_files, init_state_files, final_state_files, grasping_state_files, first_step_folders)
    save_mobile_base_initial_joint_angles(config_files, init_state_files, final_state_files, grasping_state_files, first_step_folders)
    