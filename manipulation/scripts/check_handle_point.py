# TODO: after the reach till contact stage, the gripper action should always be close. 
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

def parallel_render(args):
    task_config_path, solution_path, first_step, rpy, in_gripper_frame, gripper_num_points, add_contact, \
        state, object_name, num_point_in_pc, use_joint_angle, idx = args
    
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
        object_name, rpy_mean_list=rpy, seed=0, in_gripper_frame=in_gripper_frame, 
        gripper_num_points=gripper_num_points, add_contact=add_contact, num_points=num_point_in_pc,
        use_joint_angle=use_joint_angle)
    load_env(simulator._env, load_path=state)
    observation = simulator._get_observation()
    rgb = simulator._env.render()
    
    point_cloud = observation['point_cloud'].tolist()
    pos_ori = observation['agent_pos'].tolist()
    info = simulator._env._get_info()
    handle_median = info['handle_pos']
    simulator._env.close()
        
    return point_cloud, pos_ori, rgb, handle_median, idx

def sort_states_file_by_file_number(state_path):
    # all the file are named as state_0.pkl, state_1.pkl, ...
    ret_files = []
    for file in os.listdir(state_path):
        if file.startswith("state_") and file.endswith(".pkl"):
            ret_files.append(file)

    ret_files = sorted(ret_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return ret_files

def extract_pc_states_for_all_trajectories(task_config_path, solution_path, object_name, exp_name=None, 
                                           in_gripper_frame=False, parallel=True,
                                           gripper_num_points=0, add_contact=False):
    
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
    all_experiments = os.listdir(experiment_folder)
    all_experiments = sorted(all_experiments)
    all_experiments = all_experiments
    
    # non_perturbed_experiments = [x for x in all_experiments if 'perturb' not in x]
    # perturbed_reaching_experiments = [x for x in all_experiments if 'perturb' in x and 'reaching' in x]
    # perturned_open_expierments = [x for x in all_experiments if 'perturb' in x and 'open' in x]

    # all_experiments = non_perturbed_experiments
    # if args.include_reaching_perturbation:
    #     all_experiments += perturbed_reaching_experiments
    # if args.include_open_perturbation:
    #     all_experiments += perturned_open_expierments

    all_experiments = all_experiments[:args.num_experiment]
    
    ret_pc = []
    ret_pos_ori = []
    stages = []
    store_experiment_label_paths = []
    all_traj_rgbs = []
    all_traj_handle_poses = []
    for experiment in tqdm.tqdm(all_experiments):
        if "meta" in experiment:
            continue
        
        expert_states = []
        experiment_path = os.path.join(experiment_folder, experiment)
        cprint("Extracting data from experiment: " + experiment, "blue")
        task_config_path = os.path.join(experiment_path, "task_config.yaml")

        all_substeps_path = os.path.join(solution_path, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()
            second_step = substeps[1].lstrip().rstrip()

        all_substeps_type = os.path.join(solution_path, "substep_types.txt")
        with open(all_substeps_type, "r") as f:
            all_substeps_type = f.readlines()
            first_step_type = all_substeps_type[0].lstrip().rstrip()
        first_step_folder = first_step.replace(" ", "_") + "_"  + first_step_type
        first_stage_states_path = os.path.join(experiment_path, first_step_folder, "states")
        stage_lengths = os.path.join(experiment_path, first_step_folder, "stage_lengths.json")
        
        first_stage_states = sort_states_file_by_file_number(first_stage_states_path)      
        expert_states.extend([os.path.join(first_stage_states_path, x) for x in first_stage_states])
        if len(expert_states) == 0:
            print("No states found for experiment continue")
            continue
        
        store_experiment_label_paths.append(os.path.join(experiment_path, first_step_folder))
        label_path = os.path.join(experiment_path, first_step_folder, "label.json")
        if os.path.exists(label_path):
            label = json.load(open(label_path, "r"))
            if not label["good_traj"]:
                print("Not good traj continue")
                continue
            
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)
        stages.append(stage_lengths)
        
        if 'reach_handle' in stage_lengths.keys():
            reaching_phase = stage_lengths['reach_handle']
        else:
            reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
      
        
    
        opened_angle_file = os.path.join(experiment_path, first_step_folder, "opened_angle.txt")
        if os.path.exists(opened_angle_file): # for some perturbed trajectories, we did not really continue openeing the handle. 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                ratio = opened_angle / max_angle
            if not (ratio > 0.5):
                print("not open enough, continue")
                continue

        if args.use_extracted and os.path.exists(os.path.join(experiment_path, first_step_folder, "extracted.pkl")):
            with open(os.path.join(experiment_path, first_step_folder, "extracted.pkl"), "rb") as f:
                data = pickle.load(f)
                if len(data) == 2:
                    pc_list, pos_ori_list = data
                    rgb_list = None
                else:
                    pc_list, pos_ori_list, rgb_list = data
                    
                assert len(pc_list[0]) == args.pointcloud_num
                
            if args.after_reaching:
                pc_list = pc_list[reaching_phase:]
                pos_ori_list = pos_ori_list[reaching_phase:]
                if rgb_list is not None:
                    rgb_list = rgb_list[reaching_phase:]
            if args.after_opening:
                pc_list = pc_list[stage_lengths['open_gripper']:]
                pos_ori_list = pos_ori_list[stage_lengths['open_gripper']:]
                if rgb_list is not None:
                    rgb_list = rgb_list[stage_lengths['open_gripper']:]
        else:
            rpy_list = [[[0, 0, -45], [0, 0, -135]]]
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
                        object_name, rpy_mean_list=rpy, seed=0, in_gripper_frame=in_gripper_frame, 
                        gripper_num_points=gripper_num_points, add_contact=add_contact)
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
                else:
                    # parallel version
                    if args.after_reaching:
                        expert_states = expert_states[reaching_phase:]
                    if args.after_opening:
                        expert_states = expert_states[stage_lengths['open_gripper']:]
                    results = pool.map(parallel_render, 
                        [(task_config_path, solution_path, first_step, rpy, in_gripper_frame, gripper_num_points, add_contact,
                        expert_states[i], object_name, args.pointcloud_num, args.use_joint_angle, i) for i in range(len(expert_states))])
                    results = sorted(results, key=lambda x: x[-1])
                    # print([result[2] for result in results])
                    pc_list = [x[0] for x in results]        
                    pos_ori_list = [x[1] for x in results]
                    rgb_list = [x[2] for x in results]
                    handle_median_pos_list = [x[3] for x in results]
    
            end = time.time()
            cprint(f"Finished extracting data from trajectory index: {str(len(ret_pc))} time cost {end - beg}" , "green")

        ret_pc.append(pc_list)
        ret_pos_ori.append(pos_ori_list)
        all_traj_rgbs.append(rgb_list)
        all_traj_handle_poses.append(handle_median_pos_list)
        
        # if not args.after_reaching and not args.after_opening:
        #     with open(os.path.join(experiment_path, first_step_folder, "extracted_{}.pkl".format(args.use_joint_angle)), "wb") as f:
        #         pickle.dump((pc_list, pos_ori_list, rgb_list), f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return ret_pc, ret_pos_ori, all_traj_rgbs, all_traj_handle_poses, stages, store_experiment_label_paths
    
def extract_demos_from_a_directory(dirtory_path, object_category, exp_name=None, in_gripper_frame=False, parallel=True, 
                                    gripper_num_points=0, add_contact=False, save_path=None):
    task_paths = os.listdir(dirtory_path)
    task_paths = sorted(task_paths)
    
    action_dist_save_path = os.path.join(save_path, "action_dist")
    if not os.path.exists(action_dist_save_path):
        os.makedirs(action_dist_save_path)
    demo_rgb_save_path = os.path.join(save_path, "demo_rgbs")
    if not os.path.exists(demo_rgb_save_path):
        os.makedirs(demo_rgb_save_path)

    all_pc_list = []
    all_state_list = []
    all_action_list = []
    last_state_indices = []
    total_count = 0
    for task_path in task_paths[args.task_beg_idx:args.task_end_idx]:
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

        # ret_pc, ret_pos_ori = extract_pc_states_for_one_trajectory(task_config_path, solution_path, object_category, in_gripper_frame=in_gripper_frame)
        ret_pc, ret_pos_ori, all_traj_rgbs, all_traj_handle_pos, stages, store_label_paths = extract_pc_states_for_all_trajectories(
            task_config_path, solution_path, object_category, exp_name=exp_name, 
            in_gripper_frame=in_gripper_frame, parallel=parallel,
            gripper_num_points=gripper_num_points, add_contact=add_contact)
        
        with open(os.path.join(save_path, "all_demo_path.txt"), "w") as f:
            f.write("\n".join(store_label_paths))
        
        for traj_idx, (pc, pos_ori, handle_pos, stage_length, store_label_path) in \
                tqdm.tqdm(enumerate(zip(ret_pc, ret_pos_ori, all_traj_handle_pos, stages, store_label_paths))):

            good_traj = True

            traj_actions = []
            quaternion_diffs = []
            if not args.use_joint_angle:
                base_pos = pos_ori[0][:3]
                base_ori_6d = pos_ori[0][3:9]
                base_finger_angle = pos_ori[0][9]
            
            open_door_start_idx = 0
            # NOTE: for open_door_per_angle_new.py the keys order are different
            if 'stage' not in stage_length.keys():
                keys = ["reach_handle", "open_gripper", "reach_to_contact", "close_gripper"]
            else:
                keys = ['open_gripper', "grasp_handle", 'close_gripper'] if "open_gripper" in stage_length['stage'] else ['grasp_handle', 'close_gripper']

            for key in keys:
                open_door_start_idx += stage_length.get(key, 0)
            
            after_contact_step_idx = stage_length['reach_handle'] + stage_length['reach_to_contact']
        
            filtered_pcs = []
            filtered_pos_oris = []
            filtered_rgbs = []
            base_rgb = all_traj_rgbs[traj_idx][0]
            base_pc = pc[0]
            base_pos_ori = pos_ori[0]
            num_handle_points_in_pc = []
            for i in range(len(pos_ori) - 1):
                target_pos_ori = pos_ori[i+1]
                
                cur_pc = pc[i]
                cur_handle_pos = handle_pos[i]
                distance = np.linalg.norm(np.array(cur_pc).reshape(-1, 3) - np.array(cur_handle_pos).reshape(1, 3), axis=1)
                num_pc_near_handle = np.sum(distance < 0.02)
                print("num_pc_near_handle: ", num_pc_near_handle)
                num_handle_points_in_pc.append(num_pc_near_handle)

                if not args.use_joint_angle:                
                    cur_pos = pos_ori[i][:3]
                    target_pos = pos_ori[i+1][:3]

                    single_step_delta_pos = np.array(target_pos) - np.array(cur_pos)
                    
                    # if single step translation is too large, ignore this trajectory
                    if np.linalg.norm(single_step_delta_pos) > 0.02:
                        good_traj = False
                        print("not good traj due to delta movement too large")
                        break
                    
                    delta_pos = np.array(target_pos) - np.array(base_pos)

                    cur_ori_6d = pos_ori[i][3:9]
                    
                    # change the delta_pos into gripper frame
                    if in_gripper_frame:
                        cur_mat = rotation_transfer_6D_to_matrix(cur_ori_6d)
                        delta_pos = cur_mat.T @ delta_pos                    

                    target_ori_6d = pos_ori[i+1][3:9]
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
                    if np.abs(one_step_quaternion_diff) > 0.085:
                        good_traj = False
                        print("not good due to delta quaternion too large")
                        break
                    if i > open_door_start_idx and np.abs(one_step_quaternion_diff) > 0.02: # open door has strange behavior
                        good_traj = False
                        print("not good due to delta quaternion too large during opening door")
                        break
                    
                    # cur_finger_angle = pos_ori[i][9]
                    target_finger_angle = pos_ori[i+1][9]

                    # delta_finger_angle = target_finger_angle - cur_finger_angle
                    delta_finger_angle = target_finger_angle - base_finger_angle
                    if args.fixed_finger_movement:
                        # NOTE: the finger dimension only controls the open or close, and it will open/close by a fixed amount
                        if i > after_contact_step_idx:
                            delta_finger_angle = traj_actions[after_contact_step_idx - 1][-1]
                
                filter_action = False
                if args.filter_small_action: 
                    if args.after_reaching or args.after_opening:
                        if np.linalg.norm(delta_pos) < args.min_translation and np.linalg.norm(quat_diff) < args.min_rotation and np.abs(delta_finger_angle) < args.min_finger_angle_diff:
                            filter_action = True
                    else:
                        if np.linalg.norm(delta_pos) < args.min_translation and np.linalg.norm(quat_diff) < args.min_rotation and np.abs(delta_finger_angle) < args.min_finger_angle_diff:
                            if args.filter_after_reaching and i > stage_length["reach_handle"]:
                                filter_action = True
                            if not args.filter_after_reaching:
                                filter_action = True

                if filter_action:
                    continue
                else:
                    if not args.use_joint_angle:
                        action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
                    else:
                        # pos_ori is the normalized joint angle. Action is the delta change in normalized joint angle
                        action = np.array(target_pos_ori) - np.array(base_pos_ori) 
                        action = action[:-1] # action only controls one finger; the other one is symmetric
                        if i > after_contact_step_idx:
                            action[-1] = -0.08
                        #     action[-1] = traj_actions[after_contact_step_idx - 1][-1]
                        #     print("after contact finger movement: ", action[-1])
                        # else:
                        #     print("before contact finger movement: ", action[-1])
                            
                        action = action.tolist()
                            
                    traj_actions.append(action)
                    filtered_pcs.append(base_pc)
                    filtered_pos_oris.append(base_pos_ori)
                    filtered_rgbs.append(base_rgb)
                    base_pc = pc[i+1]
                    base_pos_ori = pos_ori[i+1]
                    base_rgb = all_traj_rgbs[traj_idx][i+1]
                    if not args.use_joint_angle:
                        base_pos = target_pos
                        base_ori_6d = target_ori_6d
                        base_finger_angle = target_finger_angle
                    
           
                    
            # plot the delta translation action distribution
            if traj_idx % 1 == 0:        
                try:
                    experiment_name = store_label_path.split("/")[-2]
                    plt.close("all")
                    plt.plot(num_handle_points_in_pc, "-*")
                    with open(os.path.join(action_dist_save_path, "{}_num_handle_point.pkl".format(experiment_name)), "wb") as f:
                        pickle.dump(num_handle_points_in_pc, f)
                        
                    save_fig_path = os.path.join(action_dist_save_path, "handle_point_in_pc_{}.png".format(experiment_name))
                    plt.savefig(save_fig_path)
                    plt.close("all")
                except:
                    pass

            path = os.path.join(store_label_path, "label.json")
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump({"good_traj": good_traj}, f)

            if good_traj:
                all_pc_list = all_pc_list + filtered_pcs
                # change the state into gripper frame
                if in_gripper_frame:
                    temp_pos_ori = []
                    for pos_ori_i in pos_ori:
                        temp_pos_ori.append([0,0,0,1,0,0,0,1,0] + pos_ori_i[9:])
                    all_state_list = all_state_list + temp_pos_ori
                else:
                    all_state_list = all_state_list + filtered_pos_oris

                # traj_actions.append([0,0,0,1,0,0,0,1,0,0])            
                all_action_list = all_action_list + traj_actions
                total_count += len(filtered_pcs)
                last_state_indices.append(deepcopy(total_count))
    
    return all_pc_list, all_state_list, all_action_list, last_state_indices
        

def main(folder_name, object_name, save_path, exp_name=None, in_gripper_frame=True, parallel=True,
         gripper_num_points=0, add_contact=False):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    meta_info = {
        "folder_name": folder_name,
        "in_gripper_frame": in_gripper_frame,
        "exp_name": exp_name,
    }
    meta_info.update(args.__dict__)
    with open(os.path.join(save_path, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=4)
    
    pc_list, state_list, action_list, last_state_indices = extract_demos_from_a_directory(
        folder_name, object_name,exp_name=exp_name, in_gripper_frame=in_gripper_frame, parallel=parallel, 
        gripper_num_points=gripper_num_points, add_contact=add_contact, save_path=save_path)
    


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--in_gripper_frame", type=int, default=0)
    args.add_argument("--gripper_num_points", type=int, default=0)
    args.add_argument("--add_contact", type=int, default=0)
    args.add_argument("--after_reaching", type=int, default=0)
    args.add_argument("--after_opening", type=int, default=0)
    args.add_argument("--filter_small_action", type=float, default=0)
    args.add_argument("--filter_after_reaching", type=float, default=0)
    args.add_argument("--min_translation", type=float, default=0.0045)
    args.add_argument("--min_rotation", type=float, default=0.008)
    args.add_argument("--min_finger_angle_diff", type=float, default=0.001)
    args.add_argument("--include_reaching_perturbation", type=int, default=0)
    args.add_argument("--include_open_perturbation", type=int, default=0)
    args.add_argument("--fixed_finger_movement", type=int, default=1)
    args.add_argument("--pointcloud_num", type=int, default=4500)
    args.add_argument("--use_joint_angle", type=int, default=0)

    
    args.add_argument("--object_name", type=str, required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--exp_name", type=str, default=None)
    args.add_argument("--folder_name", type=str, required=True)
    args.add_argument("--generate", type=int, default=1)
    args.add_argument("--parallel", type=int, default=1)
    args.add_argument("--task_beg_idx", type=int, default=0)
    args.add_argument("--task_end_idx", type=int, default=1)
    args.add_argument("--num_experiment", type=int, default=10000)
    args.add_argument("--num_worker", type=int, default=80)
    args.add_argument("--use_extracted", type=int, default=1)
    args = args.parse_args()
    
   

    if args.generate:
        set_start_method('spawn', force=True)
        num_worker = args.num_worker
        pool = Pool(processes=num_worker)
        main(args.folder_name, args.object_name, args.save_path, exp_name=args.exp_name, 
             in_gripper_frame=args.in_gripper_frame, parallel=args.parallel, 
             gripper_num_points=args.gripper_num_points, add_contact=args.add_contact)
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
                object_name, seed=0, in_gripper_frame=args.in_gripper_frame, 
                gripper_num_points=args.gripper_num_points, add_contact=args.add_contact, 
                num_points=args.pointcloud_num,
                use_joint_angle=args.use_joint_angle)
            
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
