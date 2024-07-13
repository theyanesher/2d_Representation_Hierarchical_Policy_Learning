import numpy as np
from manipulation.utils import build_up_env
from manipulation.utils import load_env, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
import os
from copy import deepcopy
import zarr
import tqdm
import time
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from termcolor import cprint
import shutil
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from multiprocessing import set_start_method
import json
from scipy.spatial.transform import Rotation as R
import pickle
from manipulation.utils import save_numpy_as_gif
from multiprocessing import Pool
from add_distractors_around_target import add_distractors_around_target
from collections import defaultdict

def parallel_render(args):
    task_config_path, solution_path, first_step, rpy, gripper_num_points, add_contact, \
        state, object_name, num_point_in_pc, use_segmask, only_handle_points, observation_mode, idx = args
    
    # cprint("Extracting data from state idx " + str(idx), "blue")
    simulator, _ = build_up_env(
                task_config=task_config_path,
                solution_path=solution_path,
                task_name=first_step.replace(" ", "_"),
                restore_state_file=None,
                render=False,
                randomize=False,
                obj_id=0,
    )
    
    simulator = RobogenPointCloudWrapper(simulator, 
        object_name, rpy_mean_list=rpy, seed=0,
        gripper_num_points=gripper_num_points, add_contact=add_contact, num_points=num_point_in_pc,
        use_segmask=use_segmask, only_handle_points=only_handle_points, 
        observation_mode=observation_mode,
        record_all_observation=True)
    
    load_env(simulator._env, load_path=state)
    observation = simulator._get_observation()
    rgb = simulator._env.render()
    
    point_cloud = observation['point_cloud'].tolist()
    traj_pos_ori = observation['agent_pos'].tolist()
    feature_map = observation['feature_map'].tolist()
    gripper_pcd = observation['gripper_pcd'].tolist()
    pcd_mask = observation['pcd_mask'].tolist()
    
    simulator._env.close()
        
    return point_cloud, traj_pos_ori, rgb, feature_map, gripper_pcd, pcd_mask, idx

def sort_states_file_by_file_number(state_path):
    # all the file are named as state_0.pkl, state_1.pkl, ...
    ret_files = []
    for file in os.listdir(state_path):
        if file.startswith("state_") and file.endswith(".pkl"):
            ret_files.append(file)

    ret_files = sorted(ret_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return ret_files

def get_first_step_folder(solution_path):
    all_substeps_path = os.path.join(solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()

    all_substeps_type = os.path.join(solution_path, "substep_types.txt")
    with open(all_substeps_type, "r") as f:
        all_substeps_type = f.readlines()
        first_step_type = all_substeps_type[0].lstrip().rstrip()
    first_step_folder = first_step.replace(" ", "_") + "_"  + first_step_type
    
    return first_step, first_step_folder

def get_all_expert_angles(all_experiments, solution_path):
    first_step, first_step_folder = get_first_step_folder(solution_path)
    ratios = []
    opened_angles = []
    for experiment_path in all_experiments:
        opened_angle_file = os.path.join(experiment_path, first_step_folder, "opened_angle.txt")
        if os.path.exists(opened_angle_file): # for some perturbed trajectories, we did not really continue openeing the handle. 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                ratio = opened_angle / max_angle
                ratios.append(ratio)
                opened_angles.append(opened_angle)
    
    return np.array(ratios), np.array(opened_angles)

def extract_pc_states_for_all_trajectories(pool_args):
    task_config_path, solution_path, object_name, exp_name, parallel, gripper_num_points, add_contact, \
        beg_idx, end_idx, save_path, add_distractors, angle_threshold, args = pool_args
    
    
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
        
    all_experiments = os.listdir(experiment_folder)
    all_experiments = sorted(all_experiments)
    all_experiments = all_experiments
    all_experiments = all_experiments[beg_idx:end_idx]
    
    # all_traj_pc = []
    # all_traj_pos_ori = []
    # all_traj_feature_maps = []
    # all_traj_gripper_pcds = []
    # all_traj_rgbs = []
    # all_traj_pcd_masks = []
    # all_traj_dp3_pc = []
    
    all_traj_stage_lengths = []
    all_traj_store_label_paths = []
    obs_keys = ['point_cloud', 'agent_pos', 'feature_map', 'gripper_pcd', 'pcd_mask', 'dp3_point_cloud', 'goal_gripper_pcd', 'displacement_gripper_to_object']
    all_traj_obs_dict_of_list = defaultdict(list)
    
    first_step, first_step_folder = get_first_step_folder(solution_path)
    for experiment in tqdm.tqdm(all_experiments):
        if "meta" in experiment:
            continue
        
        cprint("Extracting data from experiment: " + experiment, "blue")
        
        expert_states = []
        experiment_path = os.path.join(experiment_folder, experiment)
        task_config_path = os.path.join(experiment_path, "task_config.yaml")
        first_stage_states_path = os.path.join(experiment_path, first_step_folder, "states")
        
        first_stage_states = sort_states_file_by_file_number(first_stage_states_path)      
        expert_states.extend([os.path.join(first_stage_states_path, x) for x in first_stage_states])
        if len(expert_states) == 0:
            print("No states found for experiment continue")
            continue
        
        # label_path = os.path.join(experiment_path, first_step_folder, "label.json")
        # if os.path.exists(label_path):
        #     label = json.load(open(label_path, "r"))
        #     if not label["good_traj"]:
        #         print("Not good traj continue")
        #         continue
        
        stage_lengths = os.path.join(experiment_path, first_step_folder, "stage_lengths.json")
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)
    
        opened_angle_file = os.path.join(experiment_path, first_step_folder, "opened_angle.txt")
        if os.path.exists(opened_angle_file): 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                ratio = opened_angle / max_angle
            if opened_angle < angle_threshold:
                print("not open enough, continue")
                continue

        # already saved the data
        if os.path.exists(os.path.join(save_path, experiment)):
            print("Already saved the data, continue")
            continue

        if add_distractors:
            new_yaml_path = os.path.join(experiment_path, "task_config_added_distractors.yaml")
            if not os.path.exists(new_yaml_path):
                add_distractors_around_target(task_config_path, solution_path, first_step.replace(" ", "_"), save_path=new_yaml_path)
            task_config_path = new_yaml_path
        
        bad_experiment = False
        rpy_list = [[[0, 0, -45], [0, 0, -135]]] # TODO: make this an argument to pass in
        beg = time.time()
        for rpy in rpy_list:
            if not parallel:
                simulator, _ = build_up_env(
                    task_config=task_config_path,
                    solution_path=solution_path,
                    task_name=first_step.replace(" ", "_"),
                    restore_state_file=None,
                    render=False,
                    randomize=False,
                    obj_id=0,
                )
                simulator = RobogenPointCloudWrapper(simulator, 
                    object_name, rpy_mean_list=rpy, seed=0,
                    gripper_num_points=gripper_num_points, add_contact=add_contact, num_points=args.pointcloud_num,
                    use_segmask=args.use_segmask, only_handle_points=args.only_handle_points,
                    observation_mode=args.observation_mode, record_all_observation=True)
                
                door_joint_angles = []
                traj_list = defaultdict(list)

                reach_till_contact_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"]
                open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
                reach_till_contact_idx = reach_till_contact_idx // args.combine_action_steps
                open_time_idx = open_time_idx // args.combine_action_steps

                expert_states = expert_states[::args.combine_action_steps]
                for t_idx, state in enumerate(tqdm.tqdm(expert_states)):
                    load_env(simulator._env, load_path=state)
                    
                    info = simulator._env._get_info()
                    joint_angle = info['opened_joint_angle']
                    door_joint_angles.append(joint_angle)
                    
                    # only object is the opposite to add_distractors
                    only_object = not add_distractors
                    observation = simulator._get_observation(only_object=only_object)  
                    rgb = simulator._env.render()

                    traj_list['rgb'].append(rgb)
                    for key in obs_keys:
                        traj_list[key].append(observation[key].tolist())
                                    
                door_joint_angles = np.array(door_joint_angles)
                door_joint_angle_diffs = np.diff(door_joint_angles)
                open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
                door_joint_angle_diffs = door_joint_angle_diffs[open_time_idx:]
                
                # if np.any(door_joint_angle_diffs > angle_threshold / 100 * 3):
                #     print("Not good traj due to joint angle diff too large when opening the door")
                #     cprint(experiment, "red")
                #     bad_experiment=True
                
                simulator._env.close()
            else:
                # parallel version
                if args.after_opening:
                    expert_states = expert_states[stage_lengths['open_gripper']:]
                results = pool.map(parallel_render, 
                    [(task_config_path, solution_path, first_step, rpy, gripper_num_points, add_contact,
                    expert_states[i], object_name, args.pointcloud_num, args.use_segmask, args.only_handle_points, \
                        args.observation_mode,
                        i) for i in range(len(expert_states))])
                results = sorted(results, key=lambda x: x[-1])
                pc_list = [x[0] for x in results]        
                pos_ori_list = [x[1] for x in results]
                rgb_list = [x[2] for x in results]
                feature_map_list = [x[3] for x in results]
                gripper_pcd_list = [x[4] for x in results]
                pcd_mask_list = [x[5] for x in results]
                dp3_pc_list = [x[6] for x in results]
    
            end = time.time()
            cprint(f"Finished extracting data from trajectory index: {len(expert_states) // args.combine_action_steps} time cost {end - beg}" , "green")

        if not bad_experiment:
            all_traj_stage_lengths.append(stage_lengths)
            all_traj_store_label_paths.append(os.path.join(experiment_path, first_step_folder))
            for key in obs_keys + ['rgb']:
                all_traj_obs_dict_of_list[key].append(traj_list[key])
        else:
            label_path = os.path.join(experiment_path, first_step_folder, "label.json")
            with open(label_path, "w") as f:
                json.dump({"good_traj": False, "failure reason": "simulation error during door opening"}, f)        
    
    return all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths
    
def extract_demos_from_a_directory(dirtory_path, object_category, exp_name=None, parallel=True, 
                                    gripper_num_points=0, add_contact=False, save_path=None, add_distractors=False):
    action_dist_save_path = os.path.join(save_path, "action_dist")
    if not os.path.exists(action_dist_save_path):
        os.makedirs(action_dist_save_path)
    demo_rgb_save_path = os.path.join(save_path, "demo_rgbs")
    if not os.path.exists(demo_rgb_save_path):
        os.makedirs(demo_rgb_save_path)

    all_pc_list = []
    all_state_list = []
    all_action_list = []
    all_feature_map_list = []
    all_gripper_pcd_list = []
    all_pcd_mask_list = []
    last_state_indices = []
    all_demo_paths = []
    total_count = 0
    for task_path in [args.exp_folder]:
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
        
        if exp_name is None:
            experiment_folder = os.path.join(solution_path, "experiment")
        else:
            experiment_folder = os.path.join(solution_path, "experiment", exp_name)
        all_experiments = os.listdir(experiment_folder)
        num_experiment = len(all_experiments)
        
        all_expert_opened_ratios, all_expert_opened_angles = get_all_expert_angles([os.path.join(experiment_folder, exp) for exp in all_experiments], solution_path)
        angle_threshold = np.quantile(all_expert_opened_angles, 0.1)
        
        
        num_experiment = min(num_experiment, args.num_experiment)        
        batch_size = 5
        num_batch = (num_experiment - 1) // batch_size + 1
        
        for batch_idx in range(num_batch):
            beg_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_experiment)
            
            if not args.parallel:
                all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths = extract_pc_states_for_all_trajectories(
                    [
                        task_config_path, solution_path, object_category, exp_name, 
                        False,
                        gripper_num_points, add_contact, 
                        beg_idx, end_idx,
                        save_path, add_distractors,
                        angle_threshold, args
                    ])
            else:
                pool_args = [
                    [task_config_path, solution_path, object_category, exp_name, False, gripper_num_points, add_contact, \
                    i, i+1, save_path, add_distractors, angle_threshold, args] for i in range(beg_idx, end_idx)
                ]
                beg = time.time()
                res = pool.map(extract_pc_states_for_all_trajectories, pool_args)
                end = time.time()
                cprint("Finished extracting data from all trajectories using time {}".format(end - beg), "green")
                all_traj_pc = [x[0][0] for x in res if len(x[0]) > 0]
                all_traj_pos_ori = [x[1][0] for x in res if len(x[1]) > 0]
                all_traj_rgbs = [x[2][0] for x in res if len(x[2]) > 0]
                all_traj_feature_maps = [x[3][0] for x in res if len(x[3]) > 0]
                all_traj_gripper_pcds = [x[4][0] for x in res if len(x[4]) > 0]
                all_traj_pcd_masks = [x[5][0] for x in res if len(x[5]) > 0]
                all_traj_stage_lengths = [x[6][0] for x in res if len(x[6]) > 0]
                all_traj_store_label_paths = [x[7][0] for x in res if len(x[7]) > 0]
                all_traj_dp3_pc = [x[8][0] for x in res if len(x[8]) > 0]

            all_traj_pc = all_traj_obs_dict_of_list['point_cloud']
            all_traj_pos_ori = all_traj_obs_dict_of_list['agent_pos']
            all_traj_rgbs = all_traj_obs_dict_of_list['rgb']
            all_traj_feature_maps = all_traj_obs_dict_of_list['feature_map']
            all_traj_gripper_pcds = all_traj_obs_dict_of_list['gripper_pcd']
            all_traj_pcd_masks = all_traj_obs_dict_of_list['pcd_mask']
            all_traj_dp3_pc = all_traj_obs_dict_of_list['dp3_point_cloud']
            all_traj_goal_gripper_pcd = all_traj_obs_dict_of_list['goal_gripper_pcd']
            all_traj_displacement_gripper_to_object = all_traj_obs_dict_of_list['displacement_gripper_to_object']
            for traj_idx in tqdm.tqdm(range(len(all_traj_pc)), total=len(all_traj_pc)):

                traj_pc, traj_pos_ori, traj_feature_maps, traj_gripper_pcd, traj_pcd_masks = all_traj_pc[traj_idx], all_traj_pos_ori[traj_idx], all_traj_feature_maps[traj_idx], all_traj_gripper_pcds[traj_idx], all_traj_pcd_masks[traj_idx]
                traj_stage_length, traj_store_label_path  = all_traj_stage_lengths[traj_idx], all_traj_store_label_paths[traj_idx]
                traj_dp3_pc = all_traj_dp3_pc[traj_idx]
                traj_goal_gripper_pcd, traj_displacement_gripper_to_object = all_traj_goal_gripper_pcd[traj_idx], all_traj_displacement_gripper_to_object[traj_idx]
                
                good_traj = True
                failure_reason = "null"

                traj_actions = []
                quaternion_diffs = []
                
                opening_start_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact'] + traj_stage_length['close_gripper']
                after_contact_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact']
                opening_start_idx = opening_start_idx // args.combine_action_steps
                after_contact_idx = after_contact_idx // args.combine_action_steps
            
                filtered_pcs = []
                filtered_pos_oris = []
                filtered_feature_maps = []
                filtered_gripper_pcds = []
                filtered_pcd_masks = []
                filtered_rgbs = []
                filtered_dp3_pcs = []
                filtered_goal_gripper_pcds = []
                filtered_displacement_gripper_to_objects = []
                
                base_pos = traj_pos_ori[0][:3]
                base_ori_6d = traj_pos_ori[0][3:9]
                base_finger_angle = traj_pos_ori[0][9]
                base_rgb = all_traj_rgbs[traj_idx][0]
                base_feature_map = traj_feature_maps[0]
                base_gripper_pcd = traj_gripper_pcd[0]
                base_pc = traj_pc[0]
                base_pcd_mask = traj_pcd_masks[0]
                base_pos_ori = traj_pos_ori[0]
                base_dp3_pc = traj_dp3_pc[0]
                base_goal_gripper_pcd = traj_goal_gripper_pcd[0]
                base_displacement_gripper_to_object = traj_displacement_gripper_to_object[0]
                
                for i in range(len(traj_pos_ori) - 1):
                    target_pos_ori = traj_pos_ori[i+1]
                    cur_pos = traj_pos_ori[i][:3]
                    target_pos = traj_pos_ori[i+1][:3]

                    single_step_delta_pos = np.array(target_pos) - np.array(cur_pos)
                    
                    # if single step translation is too large, ignore this trajectory
                    if np.linalg.norm(single_step_delta_pos) > 0.02 * args.combine_action_steps:
                        good_traj = False
                        failure_reason = "delta movement too large"
                        print("not good traj due to delta movement too large")
                        break
                    
                    delta_pos = np.array(target_pos) - np.array(base_pos)

                    cur_ori_6d = traj_pos_ori[i][3:9]
                    
                    target_ori_6d = traj_pos_ori[i+1][3:9]
                    cur_ori_matrix = rotation_transfer_6D_to_matrix(cur_ori_6d)
                    base_ori_matrix = rotation_transfer_6D_to_matrix(base_ori_6d)
                    target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)

                    delta_ori_matrix = base_ori_matrix.T @ target_ori_matrix
                    delta_ori_6d = rotation_transfer_matrix_to_6D(delta_ori_matrix)
                    
                    cur_ori_quat =  R.from_matrix(cur_ori_matrix).as_quat()
                    base_ori_quat =  R.from_matrix(base_ori_matrix).as_quat()
                    target_ori_quat = R.from_matrix(target_ori_matrix).as_quat()
                    quat_diff = np.arccos(2 * np.dot(base_ori_quat, target_ori_quat)**2 - 1)
                    one_step_quaternion_diff = np.arccos(2 * np.dot(cur_ori_quat, target_ori_quat)**2 - 1)
                    quaternion_diffs.append(quat_diff)
                    
                    # if single step rotation is too large, ignore this trajectory
                    if np.abs(one_step_quaternion_diff) > 0.085 * args.combine_action_steps:
                        good_traj = False
                        failure_reason = "delta quaternion too large"
                        print("not good due to delta quaternion too large")
                        break
                                            
                    target_finger_angle = traj_pos_ori[i+1][9]
                    delta_finger_angle = target_finger_angle - base_finger_angle
                            
                    filter_action = False
                    if i >= after_contact_idx and i < opening_start_idx:
                        if np.abs(delta_finger_angle) < args.min_finger_angle_diff and args.filter_close_zero_action:
                            # print("filter close zero action")
                            filter_action = True
                            
                    if filter_action:
                        continue
                    else:
                        action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
                        traj_actions.append(action)
                        filtered_pcs.append(base_pc)
                        filtered_pcd_masks.append(base_pcd_mask)
                        filtered_gripper_pcds.append(base_gripper_pcd)
                        filtered_feature_maps.append(base_feature_map)
                        filtered_pos_oris.append(base_pos_ori)
                        filtered_rgbs.append(base_rgb)
                        filtered_dp3_pcs.append(base_dp3_pc)
                        filtered_goal_gripper_pcds.append(base_goal_gripper_pcd)
                        filtered_displacement_gripper_to_objects.append(base_displacement_gripper_to_object)
                        
                        base_pc = traj_pc[i+1]
                        base_pcd_mask = traj_pcd_masks[i+1]
                        base_gripper_pcd = traj_gripper_pcd[i+1]
                        base_feature_map = traj_feature_maps[i+1]
                        base_pos_ori = traj_pos_ori[i+1]
                        base_rgb = all_traj_rgbs[traj_idx][i+1]
                        base_pos = target_pos
                        base_ori_6d = target_ori_6d
                        base_finger_angle = target_finger_angle
                        base_dp3_pc = traj_dp3_pc[i+1]
                        base_displacement_gripper_to_object = traj_displacement_gripper_to_object[i+1]
                        base_goal_gripper_pcd = traj_goal_gripper_pcd[i+1]
                        
            
                # plot the delta translation action distribution
                if traj_idx % 5 == 0:        
                    try:
                        save_numpy_as_gif(np.array(filtered_rgbs), os.path.join(demo_rgb_save_path, "demo_" + str(traj_idx) + ".gif"))
                        plt.close("all")

                        delta_translations = np.array(traj_actions)[:, :3]
                        delta_translations_lengths = np.linalg.norm(delta_translations, axis=1)
                        delta_joint_angles = np.array(traj_actions)[:, -1]
                        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                        axes = axes.reshape(-1)
                        vals = [delta_translations_lengths, quaternion_diffs, delta_joint_angles]
                        titles = ["delta_translation_lengths", "quaternion_diffs", "delta_joint_angles"]
                        for idx, val in enumerate(vals):
                            axes[idx].plot(range(len(val)), val, "-*")
                            keys = ["reach_handle", "reach_to_contact", "close_gripper", "open_door"]
                            
                            base = 0
                            for key in keys:
                                base += traj_stage_length[key]
                                axes[idx].axvline(x=base, color='r', linestyle='--')
                                axes[idx].text(base, 0, key, rotation=90)
                            axes[idx].set_title(titles[idx])
                                
                        suffix = "good" if good_traj else "bad"
                        save_fig_path = os.path.join(action_dist_save_path, "delta_distribution_{}_{}.png".format(traj_idx, suffix))
                        plt.savefig(save_fig_path)
                        plt.close("all")
                    except:
                        pass

                path = os.path.join(traj_store_label_path, "label.json")
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        json.dump({"good_traj": good_traj, "failure reason": failure_reason}, f)

                if good_traj:
                    for i in range(len(traj_actions)):
                        if i >= after_contact_idx:
                            # print("set gripper to be always close after contact")
                            traj_actions[i][-1] = args.close_gripper_action
                            
                    experiment_path = os.path.dirname(traj_store_label_path)
                    all_demo_paths.append(experiment_path)
                    data_save_path = os.path.join(save_path, experiment_path.split("/")[-1])
                    beg = time.time()
                    save_data(filtered_pcs, filtered_pos_oris, filtered_feature_maps, filtered_gripper_pcds, 
                            filtered_pcd_masks, traj_actions, filtered_dp3_pcs, 
                            filtered_goal_gripper_pcds, filtered_displacement_gripper_to_objects, 
                            data_save_path)
                    end = time.time()
                    cprint("Finished saving data to {} using time {}".format(data_save_path, end-beg), "green")
                    del filtered_pcs, filtered_pos_oris, filtered_feature_maps, filtered_gripper_pcds, filtered_pcd_masks, traj_actions, filtered_dp3_pcs, filtered_displacement_gripper_to_objects, filtered_goal_gripper_pcds
            
            save_example_pointcloud([traj_pc[0] for traj_pc in all_traj_pc], save_path)
            save_example_pointcloud([traj_dp3_pc[0] for traj_dp3_pc in all_traj_dp3_pc], save_path, name='example_dp3_pointcloud')
            del all_traj_pc, all_traj_pos_ori, all_traj_rgbs, all_traj_feature_maps, all_traj_gripper_pcds, all_traj_pcd_masks, all_traj_stage_lengths, all_traj_store_label_paths
                
    with open(os.path.join(save_path, "all_demo_path.txt"), "w") as f:
        f.write("\n".join(all_demo_paths))
            
# def save_data(pc_list, state_list, feature_map_list, gripper_pcd_list, pcd_mask_list, action_list, dp3_pc_list, 
#                 goal_gripper_pcd_list, displacement_gripper_to_object_list,
#               save_dir):
#     zarr_root = zarr.group(save_dir)
#     try:
#         zarr_data = zarr_root.create_group('data')
#         zarr_meta = zarr_root.create_group('meta')

#         state_arrays = np.array(state_list)
#         point_cloud_arrays = np.array(pc_list)
#         action_arrays = np.array(action_list)
#         dp3_point_cloud_arrays = np.array(dp3_pc_list)
#         if 'act3d' in args.observation_mode:
#             feature_map_arrays = np.array(feature_map_list)
#             gripper_pcd_arrays = np.array(gripper_pcd_list)
#             pcd_mask_list = np.array(pcd_mask_list)
#             goal_gripper_pcd_arrays = np.array(goal_gripper_pcd_list)

#         compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
#         state_chunk_size = (100, state_arrays.shape[1])
#         point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
#         dp3_point_cloud_chunk_size = (100, dp3_point_cloud_arrays.shape[1], dp3_point_cloud_arrays.shape[2])
#         action_chunk_size = (100, action_arrays.shape[1])
#         zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
#         zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
#         zarr_data.create_dataset('dp3_point_cloud', data=dp3_point_cloud_arrays, chunks=dp3_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
#         zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
#         if 'act3d' in args.observation_mode:
#             feature_map_chunk_size = (100, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3], feature_map_arrays.shape[4]) # there can be mutiple cameras
#             gripper_pcd_chunk_size = (100, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
#             pcd_mask_chunk_size = (100, pcd_mask_list.shape[1])
#             zarr_data.create_dataset('feature_map', data=feature_map_arrays, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
#             zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
#             zarr_data.create_dataset('pcd_mask', data=pcd_mask_list, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)

#         if 'act3d' in args.observation_mode:
#             del feature_map_arrays, gripper_pcd_arrays
#         del state_arrays, point_cloud_arrays, action_arrays
#         del zarr_root, zarr_data, zarr_meta
#         del dp3_point_cloud_arrays
#     except Exception as e:
#         print(e)
#         print("Failed to save data to: ", save_dir)

def save_data(pc_list, state_list, feature_map_list, gripper_pcd_list, pcd_mask_list, action_list, 
              dp3_pc_list,
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              save_dir):

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)
    feature_map_arrays = np.array(feature_map_list)
    gripper_pcd_arrays = np.array(gripper_pcd_list)
    pcd_mask_list = np.array(pcd_mask_list)
    
    chunk_size = 1
    state_chunk_size = (chunk_size, state_arrays.shape[1])
    point_cloud_chunk_size = (chunk_size, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (chunk_size, action_arrays.shape[1])
    feature_map_chunk_size = (chunk_size, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3], feature_map_arrays.shape[4]) # there can be mutiple cameras
    gripper_pcd_chunk_size = (chunk_size, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    pcd_mask_chunk_size = (chunk_size, pcd_mask_list.shape[1])
    if goal_gripper_pcd is not None:
        goal_gripper_pcd = np.array(goal_gripper_pcd)
        goal_gripper_pcd_chunk_size = (chunk_size, goal_gripper_pcd.shape[1], goal_gripper_pcd.shape[2])
    if displacement_gripper_to_object is not None:
        displacement_gripper_to_object = np.array(displacement_gripper_to_object)
        displacement_gripper_to_object_chunk_size = (chunk_size, displacement_gripper_to_object.shape[1], displacement_gripper_to_object.shape[2])
    if dp3_pc_list is not None:
        dp3_pc_list = np.array(dp3_pc_list)
        dp3_point_cloud_chunk_size = (chunk_size, dp3_pc_list.shape[1], dp3_pc_list.shape[2])
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    traj_len = len(state_list)
    for t_idx in range(traj_len):
        step_save_dir = os.path.join(save_dir, str(t_idx))
        if not os.path.exists(step_save_dir):
            os.makedirs(step_save_dir)
            
        zarr_root = zarr.group(step_save_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        zarr_data.create_dataset('state', data=state_arrays[t_idx][None, :], chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays[t_idx][None, :], chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays[t_idx][None, :], chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('feature_map', data=feature_map_arrays[t_idx][None, :], chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays[t_idx][None, :], chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('pcd_mask', data=pcd_mask_list[t_idx][None, :], chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        if goal_gripper_pcd is not None:
            zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd[t_idx][None, :], chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        if displacement_gripper_to_object is not None:
            zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_object[t_idx][None, :], chunks=displacement_gripper_to_object_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        if dp3_pc_list is not None:
            zarr_data.create_dataset('dp3_point_cloud', data=dp3_pc_list[t_idx][None, :], chunks=dp3_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    del state_arrays, point_cloud_arrays, feature_map_arrays, gripper_pcd_arrays, action_arrays
    del zarr_root, zarr_data, zarr_meta
    if goal_gripper_pcd is not None:
        del goal_gripper_pcd
    if displacement_gripper_to_object is not None:
        del displacement_gripper_to_object
    if dp3_pc_list is not None:
        del dp3_pc_list

def save_example_pointcloud(pc_list, save_dir, name='example_pointcloud'):
    if len(pc_list) > 10:
        idxes = np.random.choice(len(pc_list), 10)
    else:
        idxes = range(len(pc_list))
    save_dir = os.path.join(save_dir, "example_pointcloud")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, idx in enumerate(idxes):
        point_cloud = np.array(pc_list[idx])
        ax = plt.axes(projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        ax.view_init(azim=-90, elev=10)
        plt.savefig(os.path.join(save_dir, "example_pc_" + str(i) + ".png"))
        plt.show()
        plt.close()


def main(folder_name, object_name, save_path, exp_name=None, parallel=True,
         gripper_num_points=0, add_contact=False, add_distractors=False):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    meta_info = {
        "folder_name": folder_name,
        "exp_name": exp_name,
    }
    meta_info.update(args.__dict__)
    with open(os.path.join(save_path, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=4)
    
    extract_demos_from_a_directory(folder_name, object_name,exp_name=exp_name, parallel=parallel, 
        gripper_num_points=gripper_num_points, add_contact=add_contact, save_path=save_path, add_distractors=add_distractors)
    

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--gripper_num_points", type=int, default=0)
    args.add_argument("--add_contact", type=int, default=0)
    args.add_argument("--after_opening", type=int, default=0)
    args.add_argument("--fixed_finger_movement", type=int, default=1)
    args.add_argument("--pointcloud_num", type=int, default=4500)
    args.add_argument("--use_segmask", type=int, default=0)
    args.add_argument("--only_handle_points", type=int, default=0)
    args.add_argument("--observation_mode", type=str, default='segmask')
    args.add_argument("--filter_close_zero_action", type=int, default=1)
    args.add_argument("--min_finger_angle_diff", type=float, default=0.001)
    args.add_argument("--close_gripper_action", type=float, default=-0.006)
    args.add_argument("--combine_action_steps", type=int, default=2)

    
    args.add_argument("--object_name", type=str, required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--exp_folder", type=str, default=None)
    args.add_argument("--folder_name", type=str, required=True)
    args.add_argument("--generate", type=int, default=1)
    args.add_argument("--parallel", type=int, default=1)
    args.add_argument("--exp_name", type=str, default=None)
    args.add_argument("--num_experiment", type=int, default=10000)
    args.add_argument("--num_worker", type=int, default=20)
    args.add_argument("--add_distractors", type=int, default=0)
    args = args.parse_args()
    
   

    if args.generate:
        if args.parallel:
            set_start_method('spawn', force=True)
            num_worker = args.num_worker
            pool = Pool(processes=num_worker)
        main(args.folder_name, args.object_name, args.save_path, exp_name=args.exp_name, 
             parallel=args.parallel, 
             gripper_num_points=args.gripper_num_points, add_contact=args.add_contact, add_distractors=args.add_distractors)
    else:
        # # load the data
        zarr_root = zarr.open("data/dp3_demo/0512-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first-joint-angle-action")
        zarr_data = zarr_root['data']
        zarr_meta = zarr_root['meta']
        action_arrays = zarr_data['action'][:]
        last_state_indices = zarr_meta['episode_ends'][:]

        action_list = action_arrays.tolist()

        accumulated_angle_diff_list = []

        for j in range(len(last_state_indices)):

            # target_pos_ori = target_pos_ori[0]
            env, _ = build_up_env(
                "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024-05-11-00-24-52/task_config.yaml",
                "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle",
                "grasp_the_door_handle",
                "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024-05-11-00-24-52/grasp_the_door_handle_primitive/states/state_0.pkl", 
                render=True, 
                randomize=False,
                obj_id=0,
            )
            object_name = "StorageFurniture"
            env.reset()
            
            env = RobogenPointCloudWrapper(env, 
                object_name, seed=0,
                gripper_num_points=args.gripper_num_points, add_contact=args.add_contact, 
                num_points=args.pointcloud_num)
            
            rgbs = []

            np.random.seed(time.time_ns() % 2**32)
            robot = env._env.robot

            current_joint_angle = robot.get_joint_angles(robot.all_joint_indices)
            accumulated_angle_diff = 0
            if j == 0:
                offset = 0
            else:
                offset = last_state_indices[j]
                
            for i in range(min(400, len(action_list))):
                env.step(action_list[i+offset])
                rgbs.append(env.render())
                # control_rgbs = env._env.get_control_rgbs()
                # rgbs.extend(control_rgbs)

                # pos, ori = env._env.robot.get_pos_orient(env._env.robot.right_end_effector)
                
                # new_current_joint_angle = robot.get_joint_angles(robot.all_joint_indices)
                # diff = np.array(new_current_joint_angle) - np.array(current_joint_angle)
                # accumulated_angle_diff += np.linalg.norm(diff)
                # current_joint_angle = new_current_joint_angle

            # cprint("accumulated_angle_diff: " + str(accumulated_angle_diff), "green")
            # accumulated_angle_diff_list.append(accumulated_angle_diff)

            env._env.close()

            save_numpy_as_gif(np.array(rgbs), "data/debug.gif")
