import numpy as np
import pybullet as p
import cv2
from manipulation.lift_utils import build_up_env, set_eef_to_pose, save_numpy_as_gif, mp_to_target, grasp_the_object, mp_to_target_with_object, lift_up_the_object
from manipulation.gpt_reward_api import get_handle_pos
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
import time
from manipulation.utils import save_env, load_env, take_round_images_around_object
import os
from manipulation.utils import get_pc, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
import torch
import pytorch3d.ops as torch3d_ops
import open3d as o3d
from copy import deepcopy
import zarr
import tqdm
import time
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from manipulation.motion_planning_utils import motion_planning
from termcolor import cprint
import shutil
from manipulation.gpt_reward_api import get_joint_state

def sort_states_file_by_file_number(state_path):
    # all the file are named as state_0.pkl, state_1.pkl, ...
    ret_files = []
    for file in os.listdir(state_path):
        if file.startswith("state_") and file.endswith(".pkl"):
            ret_files.append(file)

    ret_files = sorted(ret_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return ret_files

def extract_pc_states_for_all_trajectories(task_config_path, solution_path, object_name):
    
    experiment_folder = os.path.join(solution_path, "experiment")
    all_experiments = os.listdir(experiment_folder)
    all_experiments = sorted(all_experiments)
    
    ret_pc = []
    ret_pos_ori = []
    for experiment in all_experiments:
        expert_states = []
        experiment_path = os.path.join(experiment_folder, experiment)
        cprint("Extracting data from experiment: " + experiment, "blue")

        all_substeps_path = os.path.join(solution_path, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()
            second_step = substeps[1].lstrip().rstrip()

        all_substeps_type = os.path.join(solution_path, "substep_types.txt")
        with open(all_substeps_type, "r") as f:
            all_substeps_type = f.readlines()
            first_step_type = all_substeps_type[0].lstrip().rstrip()
        first_step_folder = first_step.replace(" ", "_") + "_" + first_step_type
        first_stage_states_path = os.path.join(experiment_path, first_step_folder, "states")
        
        second_step_folder = second_step.replace(" ", "_")

        states_folder = os.path.join(experiment_path, second_step_folder, "states")
        simulator, _ = build_up_env(
            task_config=task_config_path,
            solution_path=solution_path,
            task_name=second_step_folder,
            restore_state_file=None,
            render=False,
            randomize=False,
            obj_id=0,
            )
        load_env(simulator, load_path=os.path.join(states_folder, "state_249.pkl"))
        joint_angle = get_joint_state(simulator, "StorageFurniture", "joint_0")
        cprint(f"joint angle: {joint_angle}", "green")
        
        ## ===========================================================
        ## TODO: Maybe add some filters about the scores to filter out
        ## ===========================================================
        # checkpoints_path = os.path.join(experiment_path, second_step_folder, "checkpoints")

        # if not os.path.exists(checkpoints_path):
        #     print("No checkpoints found, skipping this trajectory")
        #     continue

        # files_in_checkpoints = os.listdir(checkpoints_path)
        # for file in files_in_checkpoints:
        #     if file.endswith("score.txt"):
        #         score_path = os.path.join(checkpoints_path, file)
        #         with open(score_path, "r") as f:
        #             score = f.readline()
        #             if float(score) < 100:
        #                 print("Score is too low, skipping this trajectory")
        #                 continue

        # ## ===========================================================

        # second_stage_states_path = os.path.join(experiment_path, second_step_folder, "states")

        # first_stage_states = sort_states_file_by_file_number(first_stage_states_path)
        # second_stage_states = sort_states_file_by_file_number(second_stage_states_path)

        # expert_states.extend([os.path.join(first_stage_states_path, x) for x in first_stage_states])
        # expert_states.extend([os.path.join(second_stage_states_path, x) for x in second_stage_states])


        # ## NOTE: Maybe generate a lot of demos on this trajectory ???
        # ret_pc = []
        # ret_pos_ori = []
        # rpy_list = [[[0, 0, -45], [0, 0, -135]], [[0, 0, -150], [0, 0, -45]], [[0, 0, -135], [0, 0, -30]], [[0, 0, -30], [0, 0, -150]]]
        # # rpy_list = [[[0, 0, -45], [0, 0, -135]]]

        # print("building env config with score: ", score)

        # for rpy in rpy_list:
        #     simulator, _ = build_up_env(
        #     task_config=task_config_path,
        #     solution_path=solution_path,
        #     task_name=second_step_folder,
        #     restore_state_file=None,
        #     render=False,
        #     randomize=False,
        #     obj_id=0,
        #     )
        #     simulator = RobogenPointCloudWrapper(simulator, object_name, rpy_mean_list=rpy, seed=0)
        #     pc_list = []
        #     pos_ori_list = []
        #     for state in tqdm.tqdm(expert_states):
        #         load_env(simulator._env, load_path=state)
        #         observation = simulator._get_observation()
        #         point_cloud = observation['point_cloud'].tolist()
        #         pos_ori = observation['agent_pos'].tolist()

        #         pc_list.append(point_cloud)
        #         pos_ori_list.append(pos_ori)
                
        #     simulator._env.close()
        #     ret_pc.append(pc_list)
        #     ret_pos_ori.append(pos_ori_list)

    return ret_pc, ret_pos_ori

def extract_pc_states_for_one_trajectory(task_config_path, solution_path, object_name):
    expert_states = []
    experiment_folder = os.path.join(solution_path, "experiment")
    all_experiments = os.listdir(experiment_folder)
    all_experiments = sorted(all_experiments)
    last_experiment = all_experiments[-2]
    print("last experiment: ", last_experiment)
    experiment_path = os.path.join(experiment_folder, last_experiment)

    all_substeps_path = os.path.join(solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()
        second_step = substeps[1].lstrip().rstrip()

    all_substeps_type = os.path.join(solution_path, "substep_types.txt")
    with open(all_substeps_type, "r") as f:
        all_substeps_type = f.readlines()
        first_step_type = all_substeps_type[0].lstrip().rstrip()
    first_step_folder = first_step.replace(" ", "_") + "_" + first_step_type
    first_stage_states_path = os.path.join(experiment_path, first_step_folder, "states")
    
    second_step_folder = second_step.replace(" ", "_")

    
    ## ===========================================================
    ## TODO: Maybe add some filters about the scores to filter out
    ## ===========================================================
    checkpoints_path = os.path.join(experiment_path, second_step_folder, "checkpoints")

    if not os.path.exists(checkpoints_path):
        print("No checkpoints found, skipping this trajectory")
        return [], []

    files_in_checkpoints = os.listdir(checkpoints_path)
    for file in files_in_checkpoints:
        if file.endswith("score.txt"):
            score_path = os.path.join(checkpoints_path, file)
            with open(score_path, "r") as f:
                score = f.readline()
                if float(score) < 100:
                    print("Score is too low, skipping this trajectory")
                    return [], []

    ## ===========================================================

    second_stage_states_path = os.path.join(experiment_path, second_step_folder, "states")

    first_stage_states = sort_states_file_by_file_number(first_stage_states_path)
    second_stage_states = sort_states_file_by_file_number(second_stage_states_path)

    expert_states.extend([os.path.join(first_stage_states_path, x) for x in first_stage_states])
    expert_states.extend([os.path.join(second_stage_states_path, x) for x in second_stage_states])


    ## NOTE: Maybe generate a lot of demos on this trajectory ???
    ret_pc = []
    ret_pos_ori = []
    # rpy_list = [[[0, 0, -45], [0, 0, -135]], [[0, 0, -150], [0, 0, -45]], [[0, 0, -135], [0, 0, -30]]]
    rpy_list = [[[0, 0, -45], [0, 0, -135]]]

    print("building env config with score: ", score)

    for rpy in rpy_list:
        simulator, _ = build_up_env(
        task_config=task_config_path,
        solution_path=solution_path,
        task_name=second_step_folder,
        restore_state_file=None,
        render=True,
        randomize=False,
        obj_id=0,
        )
        simulator = RobogenPointCloudWrapper(simulator, object_name, rpy_mean_list=rpy)
        pc_list = []
        pos_ori_list = []
        for state in tqdm.tqdm(expert_states):
            load_env(simulator._env, load_path=state)
            observation = simulator._get_observation()
            point_cloud = observation['point_cloud'].tolist()
            pos_ori = observation['agent_pos'].tolist()

            pc_list.append(point_cloud)
            pos_ori_list.append(pos_ori)
            
        simulator._env.close()
        ret_pc.append(pc_list)
        ret_pos_ori.append(pos_ori_list)

    return ret_pc, ret_pos_ori
    
def extract_demos_from_a_directory(dirtory_path, object_category):
    task_paths = os.listdir(dirtory_path)
    task_paths = sorted(task_paths)

    all_pc_list = []
    all_state_list = []
    all_action_list = []
    last_state_indices = []
    total_count = 0
    for task_path in task_paths:
        files_and_folders = os.listdir(os.path.join(dirtory_path, task_path))
        solution_path, task_config_path = None, None
        for file_or_folder in files_and_folders:
            if file_or_folder.startswith("task"):
                solution_path = os.path.join(dirtory_path, task_path, file_or_folder)
            if file_or_folder.endswith(".yaml"):
                task_config_path = os.path.join(dirtory_path, task_path, file_or_folder)
        if solution_path is None or task_config_path is None:
            print("No solution path or task config path found for task: ", task_path)
            continue

        # ret_pc, ret_pos_ori = extract_pc_states_for_one_trajectory(task_config_path, solution_path, object_category)
        ret_pc, ret_pos_ori = extract_pc_states_for_all_trajectories(task_config_path, solution_path, object_category)
        # for pc, pos_ori in zip(ret_pc, ret_pos_ori):
        for pc, pos_ori in zip(ret_pc, ret_pos_ori):
            all_pc_list = all_pc_list + pc
            all_state_list = all_state_list + pos_ori

            actions = []
            for i in range(len(pos_ori) - 1):
                cur_pos = pos_ori[i][:3]
                target_pos = pos_ori[i+1][:3]

                delta_pos = np.array(target_pos) - np.array(cur_pos)

                cur_ori = pos_ori[i][3:9]
                target_ori = pos_ori[i+1][3:9]

                cur_ori = rotation_transfer_6D_to_matrix(cur_ori)
                target_ori = rotation_transfer_6D_to_matrix(target_ori)

                delta_ori = cur_ori.T @ target_ori
               
                delta_ori = rotation_transfer_matrix_to_6D(delta_ori)

                cur_angle = pos_ori[i][9]
                target_angle = pos_ori[i+1][9]

                delta_angle = target_angle - cur_angle
                action = delta_pos.tolist() + delta_ori.tolist() + [delta_angle]
                actions.append(action)

            actions.append([0,0,0,1,0,0,0,1,0,0])
            
            all_action_list = all_action_list + actions
            total_count += len(pc)
            last_state_indices.append(deepcopy(total_count))
        # return all_pc_list, all_state_list, all_action_list, last_state_indices
    
    return all_pc_list, all_state_list, all_action_list, last_state_indices
        
def save_data(pc_list, state_list, action_list, last_state_indices, save_dir):

    if os.path.exists(save_dir):
        input("The save directory already exists, press Enter to remove it and continue ...")
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    state_arrays = np.stack(state_list, axis=0)
    point_cloud_arrays = np.stack(pc_list, axis=0)
    action_arrays = np.stack(action_list, axis=0)
    episode_ends_arrays = np.array(last_state_indices)


    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    del state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta
    

if __name__ == "__main__":
    pc_list, state_list, action_list, last_state_indices = extract_demos_from_a_directory("data/storagefurniture_48700", "StorageFurniture")
    # save_data(pc_list, state_list, action_list, last_state_indices, "data/extracted/sac_storagefurniture_48700_1.zarr")

    print("last state indices: ", last_state_indices)

    # load the data
    # zarr_root = zarr.open("data/extracted/sac_storagefurniture_48700_one_trajectory.zarr")
    # zarr_data = zarr_root['data']
    # zarr_meta = zarr_root['meta']
    # action_arrays = zarr_data['action'][:]
    # action_list = action_arrays.tolist()



    # import pdb; pdb.set_trace()
    # # target_pos_ori = target_pos_ori[0]
    # env, _ = build_up_env(
    #     "/home/ziyu/Desktop/workspace/RoboGen-sim2real/data/storagefurniture_48700/storagefurniture_48700_sac/open_the_door_of_the_storagefurniture_by_its_handle_The_robotic_arm_will_open_the_door_of_the_storage_furniture_by_its_handle.yaml",
    #     "data/storagefurniture_48700/storagefurniture_48700_sac/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "open_the_storage_furniture_door",
    #     None, 
    #     render=True, 
    #     randomize=False,
    #     obj_id=0,
    # )
    # object_name = "StorageFurniture"
    # env.reset()

    # env = RobogenPointCloudWrapper(env, object_name)
    # rgbs = []

    # np.random.seed(time.time_ns() % 2**32)

    # # import pdb; pdb.set_trace()
    # for i in range(400):
    #     env.step(action_list[i])
    #     control_rgbs = env._env.get_control_rgbs()
    #     rgbs.extend(control_rgbs)

    #     pos, ori = env._env.robot.get_pos_orient(env._env.robot.right_end_effector)
        

    # save_numpy_as_gif(np.array(rgbs), "data/extracted/sac_storagefurniture_with_eff_48700.gif")


