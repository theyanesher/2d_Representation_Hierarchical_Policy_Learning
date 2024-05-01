import os
from manipulation.utils import build_up_env, load_env, save_env, save_numpy_as_gif
import json
import yaml
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from manipulation.motion_planning_utils import motion_planning
import pickle
import scipy
from manipulation.gpt_reward_api import get_link_pc, get_handle_pos
from manipulation.gpt_primitive_api import open_gripper, close_gripper, reach_till_contact, open_door, get_link_handle, get_link_pose, get_pc_num_within_gripper
from multiprocessing import set_start_method
from multiprocessing import Pool
import copy

def perturb_demo(args):
    step_name = "grasp_the_door_handle_primitive"
    exp_path, task_name, ts = args
    demo_path = os.path.join(exp_path, ts)
    config_path = os.path.join(demo_path, "config.yaml")
    if not os.path.exists(config_path):
        meta_info_path = os.path.join(demo_path, step_name, "meta_info.json")
        with open(meta_info_path, "r") as f:
            meta_info = json.load(f)
        config_path = meta_info["config_path"]
        config = yaml.safe_load(open(config_path, "r"))
        solution_path = [x['solution_path'] for x in config if 'solution_path' in x][0]
        
    env, _ = build_up_env(config_path, solution_path, task_name, None, render=False)
    env.reset()

    exp_name = exp_path.split("/")[-1]
    new_exp_name = exp_name + "_perturbed_open_per_angle_direct_grasp_no_opening"
    new_exp_path = os.path.join("/".join(exp_path.split("/")[:-1]), new_exp_name)
    new_meta_info = copy.deepcopy(meta_info)
    new_meta_info["original_demo_path"] = demo_path

    step_name = "grasp_the_door_handle_primitive"
    states_file_path = os.path.join(demo_path, step_name, "states")
    all_states = os.listdir(states_file_path)
    all_states = sorted(all_states, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    subsampled_states = all_states[::10]
    
    extracted_pkl = os.path.join(demo_path, step_name, "extracted.pkl")
    if not os.path.exists(extracted_pkl):
        return None

    with open(extracted_pkl, "rb") as f:
        extracted = pickle.load(f)
        pc_list, pos_ori_list, rgb_list = extracted
        pc_list = pc_list[::10]
    
    handle_positions = []
    eef_positions = []
    num_handle_point_in_gripper_list = []
    for state in subsampled_states:
        state_path = os.path.join(states_file_path, state)
        state = pickle.load(open(state_path, "rb"))
        env.reset(reset_state=state)
        link_pc = get_link_pc(env, 'storagefurniture', 'link_0')
        object_name = 'storagefurniture'
        all_handle_pos, handle_joint_id = get_handle_pos(env, object_name, return_median=False)
        handle_pc, handle_joint_id, handle_median = get_link_handle(all_handle_pos, handle_joint_id, link_pc)
        handle_positions.append(handle_median)
        eef_pos, eef_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
        eef_positions.append(eef_pos)
        num_handle_point_in_gripper = get_pc_num_within_gripper(eef_pos, eef_orient, handle_pc)
        num_handle_point_in_gripper_list.append(num_handle_point_in_gripper)
    
    return pc_list, handle_positions, eef_positions, num_handle_point_in_gripper_list
    
if __name__ == "__main__":
    # build the env according to the stored config
    task_name = "grasp_the_door_handle"
    exp_name = "vary_robot_init_joint_near_handle_perturbed_open_per_angle_direct_grasp_no_opening"
    exp_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/{}".format(exp_name)
    all_timesteps = os.listdir(exp_path)
    all_timesteps = sorted(all_timesteps)
    all_timesteps = all_timesteps
    
    set_start_method('spawn', force=True)
    num_worker = 80
    pool = Pool(processes=num_worker)
    
    all_args = [
        [exp_path, task_name, ts] for ts in all_timesteps
    ]

    # perturb_demo([exp_path, task_name, all_timesteps[0]])
    
    res = pool.map(perturb_demo, all_args)
    all_pc = [x[0] for x in res if x is not None]
    all_handle_positions = [x[1] for x in res if x is not None]
    all_eef_positions = [x[2] for x in res if x is not None]
    all_num_handle_point_in_gripper_list = [x[3] for x in res if x is not None]
    # flatten the list
    all_pc = [item for sublist in all_pc for item in sublist]
    all_handle_positions = [item for sublist in all_handle_positions for item in sublist]
    all_eef_positions = [item for sublist in all_eef_positions for item in sublist]
    all_num_handle_point_in_gripper_list = [item for sublist in all_num_handle_point_in_gripper_list for item in sublist]
    
    with open("data/test_dp3_encoder.pkl", "wb") as f:
        pickle.dump([all_pc, all_handle_positions, all_eef_positions, all_num_handle_point_in_gripper_list], f)    
    