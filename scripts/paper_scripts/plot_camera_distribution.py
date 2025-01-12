import numpy as np
import pybullet
import time
import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
# from train import TrainDP3Workspace
from train_ddp import TrainDP3Workspace
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
import time 
import yaml
import pickle as pkl
import cv2
import argparse

def construct_env(config_file, solution_path, task_name, init_state_file, obj_translation=None, real_world_camera=False, noise_real_world_pcd=False,
                  randomize_camera=False):
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
                    random_object_translation=obj_translation,
            )
            
    object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, 
                                                num_points=4500,
                                                observation_mode='act3d_goal_displacement_gripper_to_object',
                                                real_world_camera=real_world_camera,
                                                noise_real_world_pcd=noise_real_world_pcd,
                                                )
         
    return pointcloud_env

### build the env
experiment_name = '0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
experiment_folder = 'data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle'
demo_experiment_path = None

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
    expert_opened_angles.append(expert_opened_angle)
    
    first_stage_states_path = os.path.join(first_step_folder, "states")
    stage_lengths = os.path.join(first_step_folder, "stage_lengths.json")
    with open(stage_lengths, "r") as f:
        stage_lengths = json.load(f)
    
    init_state_file = os.path.join(first_stage_states_path, "state_0.pkl")
    init_state_files.append(init_state_file)
    config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
    config_files.append(config_file)
            
config_files = config_files

opened_joint_angles = {}

exp_end_idx = len(config_files)
exp_beg_idx = 0

config_files = config_files[exp_beg_idx:exp_end_idx]
init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]

for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
        
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
    all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()
        task_name = first_step.replace(" ", "_")
                
    env = construct_env(config_file, solution_path, task_name, init_state_file)
    break

### get random camera positions and orientations
view_matrices = []
for r_cam_idx in range(40):
    print("setting random camera", r_cam_idx)
    env.reset_random_cameras()
    view_matrices.append(env._env.view_matrix)
    
### get real world camera positions and orientations
view_matrices_real_world = []
for r_cam_idx in range(40):
    print("setting real world random camera", r_cam_idx)
    env.randomize_real_world_camera()
    view_matrices_real_world.append(env._env.view_matrix)
    
### plot randomized camera positions
for view_matrix in view_matrices:
    view_matrix = np.array(view_matrix).reshape(4, 4, order='F')
    view_matrix = np.linalg.inv(view_matrix)
    cam_pos = view_matrix[:3, 3]
    # print(cam_pos)
    p.addUserDebugPoints([cam_pos], [[1, 0, 0]], physicsClientId=env._env.id, pointSize=10)
    
### plot real world randomzied camera positions
for view_matrix in view_matrices_real_world:
    view_matrix = np.array(view_matrix).reshape(4, 4, order='F')
    view_matrix = np.linalg.inv(view_matrix)
    cam_pos = view_matrix[:3, 3]
    # print(cam_pos)
    p.addUserDebugPoints([cam_pos], [[0, 1, 0]], physicsClientId=env._env.id, pointSize=10)


# camera_target = [0, 0, 0.6]
# camera_eye = [-0.5, 1, 2]
# view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1])
# project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=640/576 ,nearVal=env.depth_near, farVal=env.depth_far, physicsClientId=env._env.id)

for _ in range(100000000):
    p.stepSimulation(physicsClientId=env._env.id)
    
