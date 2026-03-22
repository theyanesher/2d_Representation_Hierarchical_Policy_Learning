import numpy as np
from manipulation.utils import build_up_env, save_numpy_as_mp4
from manipulation.utils import load_env, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
import os
import pickle
import yaml
import tqdm
import time
from manipulation.robogen_image_wrapper import RobogenImageWrapper
from termcolor import cprint
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import json
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from common.hdf5_data_recorder import HDF5DataRecorder
from manipulation.goal_image_utils import generate_heatmap_from_points
from common.data_utils import process_pointmap, process_plucker, process_depth
from diffusion_policy.diffusion_policy.common.action_util import hybrid_delta_to_delta_actions

def sort_states_file_by_file_number(state_path):
    # all the file are named as state_0.pkl, state_1.pkl, ...
    ret_files = []
    for file in os.listdir(state_path):
        if file.startswith("state_") and file.endswith(".pkl"):
            ret_files.append(file)

    ret_files = sorted(ret_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return ret_files

def get_all_expert_angles(all_experiments):
    ratios = []
    opened_angles = []
    # import pdb; pdb.set_trace();
    for experiment_path in all_experiments:
        # import pdb; pdb.set_trace();
        opened_angle_file = os.path.join(experiment_path, "opened_angle.txt")
        if os.path.exists(opened_angle_file): # for some perturbed trajectories, we did not really continue openeing the handle. 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                min_angle = float(angles[1].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                opened_angle -= min_angle
                ratio = opened_angle / (max_angle - min_angle)
                ratios.append(ratio)
                opened_angles.append(opened_angle)
            
    return np.array(ratios), np.array(opened_angles)

def get_obs_keys(args):
    num_cams = 3
    obs_keys = ['state', 'gripper_to_world'] #  , 'goal_gripper_pcd'
    for i in range(num_cams):
        obs_keys += [f'cam{i}_image', f'cam{i}_depth', f'cam{i}_segmask', f'cam{i}_extrinsic', f'cam{i}_intrinsic']
        if 'pointmap' in args.observation_mode or 'all' in args.observation_mode:
            obs_keys += [f'cam{i}_pointmap']
        if 'plucker' in args.observation_mode or 'all' in args.observation_mode:
            obs_keys += [f'cam{i}_plucker']
    return obs_keys

def extract_pc_states_for_all_trajectories(pool_args):
    task_config_path, solution_path, env_name, exp_name, experiments, save_path, angle_threshold, args = pool_args
       
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
    if not os.path.exists(experiment_folder):
        print("Experiment folder does not exist: ", experiment_folder)
        if exp_name is not None:
            experiment_folder = solution_path
        else:
            experiment_folder = os.path.join(solution_path, exp_name)
        
    all_traj_stage_lengths = []
    all_traj_store_label_paths = []

    # Replace the old obs_keys definition:
    obs_keys = get_obs_keys(args)
    # if args.extract_pcds:
    #     obs_keys += ['point_cloud', 'gripper_pcd']
    all_traj_obs_dict_of_list = defaultdict(list)
    
    for experiment in experiments:
        if "meta" in experiment:
            continue
        
        expert_states = []
        experiment_path = os.path.join(experiment_folder, experiment)
        task_config_path = os.path.join(experiment_path, "task_config.yaml")
        states_path = os.path.join(experiment_path, "states")
        
        states = sort_states_file_by_file_number(states_path)      
        expert_states.extend([os.path.join(states_path, x) for x in states])
        if len(expert_states) == 0:
            print("No states found for experiment continue")
            continue
        
        stage_lengths = os.path.join(experiment_path, "stage_lengths.json")
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)

        reach_till_contact_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"]
        open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]

        if len(expert_states) - open_time_idx < 5: # if the opening time is too short, skip this trajectory
            print("Opening time is too short, continue")
            continue
        
        opened_angle_file = os.path.join(experiment_path, "opened_angle.txt")
        if os.path.exists(opened_angle_file): 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                min_angle = float(angles[1].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                opened_angle -= min_angle
                ratio = opened_angle / (max_angle - min_angle)
            if opened_angle < angle_threshold or ratio < args.min_opened_ratio:
                print("not open enough, continue")
                continue

        # already saved the data
        # if os.path.exists(os.path.join(save_path, experiment)):
        #     print("Already saved the data, continue")
        #     continue

        bad_experiment = False
        camera_detected = True
        beg = time.time()
            
        config = yaml.safe_load(open(task_config_path, "r"))
        for config_dict in config:
            if 'name' in config_dict:
                object_name = config_dict['name'].lower()
        # import pdb; pdb.set_trace()
        simulator, _ = build_up_env(
            task_config=task_config_path,
            env_name=env_name,
            restore_state_file=None,
            render=False,
            randomize=False,
            obj_id=0,
        )
        simulator = RobogenImageWrapper(simulator, 
            object_name,
            observation_mode=args.observation_mode, 
            noise_real_world_pcd=args.noise_real_world_pcd,
            real_world_camera=args.real_world_camera,
            extract_pcds=args.extract_pcds
        )

        traj_list = defaultdict(list)
        
        reach_till_contact_idx = reach_till_contact_idx // args.combine_action_steps
        open_time_idx = open_time_idx // args.combine_action_steps

        expert_states = expert_states[::args.combine_action_steps]
        load_env(simulator._env, load_path=expert_states[0])
        # random set the camera of the environment
        if args.randomize_camera > 0:
            simulator.reset_random_cameras(args.randomize_camera)

        if simulator.check_handle_observed_in_pc() < 5:
            print("Handle not observed in the point cloud, continue")
            camera_detected = False
            simulator._env.close()

        if camera_detected:
            for t_idx, state in enumerate(tqdm.tqdm(expert_states)):
                load_env(simulator._env, load_path=state)
                
                # only object is the opposite to add_distractors
                only_object = False #  True
                observation = simulator._get_observation(only_object=only_object)  
                rgb = simulator._env.render()

                traj_list['rgb'].append(rgb)
                # import pdb; pdb.set_trace();
                for key in obs_keys:
                    # import pdb; pdb.set_trace();
                    traj_list[key].append(observation[key])
            # import pdb; pdb.set_trace()
            if args.extract_pcds:
                # import pdb; pdb.set_trace()
                goal_gripper_pcd_at_grasping = traj_list['state'][open_time_idx]
                goal_gripper_pcd_at_end = traj_list['state'][-1]
                for t in range(open_time_idx):
                    traj_list['goal_gripper_pcd'].append(goal_gripper_pcd_at_grasping)
                for t in range(open_time_idx, len(traj_list['state'])):
                    traj_list['goal_gripper_pcd'].append(goal_gripper_pcd_at_end)
            simulator._env.close()
    
        end = time.time()
        cprint(f"Finished extracting data from trajectory with length: {len(expert_states) // args.combine_action_steps} time cost {end - beg}", "green")
        # import pdb; pdb.set_trace()
        if not bad_experiment and camera_detected:
            all_traj_stage_lengths.append(stage_lengths)
            all_traj_store_label_paths.append(experiment_path)
            for key in obs_keys + ['goal_gripper_pcd']:
                all_traj_obs_dict_of_list[key].append(traj_list[key])
            # import pdb; pdb.set_trace();
        else:
            label_path = os.path.join(experiment_path, "label.json")
            print("handle not in camera!!")
            try:
                with open(label_path, "w") as f:
                    json.dump({"good_traj": False, "failure reason": "simulation error during door opening"}, f)        
            except:
                pass

    first_state_val = pickle.load(open(expert_states[0], "rb"))
    last_state_val = pickle.load(open(expert_states[-1], "rb"))
    sim_states = {
            'initial_state': first_state_val,
            'final_state': last_state_val
        }
    # import pdb; pdb.set_trace()
    
    return all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths, sim_states
    
def extract_demos_from_a_directory(dirtory_path, exp_name=None, env_name=None, extract_name=None, save_path=None):
    
    demo_rgb_save_path = os.path.join(save_path, "demo_rgbs")
    if not os.path.exists(demo_rgb_save_path):
        os.makedirs(demo_rgb_save_path)

    all_demo_paths = []
    task_path = extract_name
    solution_path = os.path.join(dirtory_path, task_path)
    files_and_folders = os.listdir(solution_path)
    task_config_path = None
    for file_or_folder in files_and_folders:
        if file_or_folder.endswith(".yaml"):
            task_config_path = os.path.join(dirtory_path, task_path, file_or_folder)
    # import pdb; pdb.set_trace()      
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)

    if not os.path.exists(experiment_folder):
        print("Experiment folder does not exist: ", experiment_folder)
        if exp_name is not None:
            experiment_folder = solution_path
        else:
            experiment_folder = os.path.join(solution_path, exp_name)
    # import pdb; pdb.set_trace()
    all_experiments = [x for x in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, x)) and not x.startswith(".")]
    # import pdb; pdb.set_trace()
    all_experiments = sorted(all_experiments)
    if args.traj_idx is not None:
        all_experiments = [all_experiments[args.traj_idx]]
    print("all experiments to render: ", all_experiments)
    # filter out successful experiments
    success_experiments = []
    for exp in all_experiments:
        # import pdb; pdb.set_trace()
        state_path = os.path.join(experiment_folder, exp, "states")
        if os.path.exists(state_path):
            if len(os.listdir(state_path)) > 1 and os.path.exists(os.path.join(experiment_folder, exp, "all.gif")): # "all.mp4"
                success_experiments.append(exp)
    
    all_experiments = success_experiments
    num_experiment = len(all_experiments)
    # import pdb; pdb.set_trace();
    if args.traj_idx is not None and args.angle_threshold is not None:
        angle_threshold = args.angle_threshold
        print(f"Using forced angle threshold: {angle_threshold}")
    else:
        # Original fallback logic
        # import pdb; pdb.set_trace()
        _, all_expert_opened_angles = get_all_expert_angles([os.path.join(experiment_folder, exp) for exp in all_experiments])
        angle_threshold = np.quantile(all_expert_opened_angles, 0.1)

    num_experiment = min(num_experiment, args.num_experiment)        
    batch_size = 1
    num_batch = (num_experiment - 1) // batch_size + 1

    obs_keys = get_obs_keys(args)
    for batch_idx in range(num_batch):
        beg_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_experiment)
        # import pdb; pdb.set_trace();
        all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths, sim_states = extract_pc_states_for_all_trajectories(
            [
                task_config_path, solution_path, env_name, exp_name, 
                all_experiments[beg_idx:end_idx], 
                save_path, 
                angle_threshold, args
            ])
        # import pdb; pdb.set_trace();
        all_traj_pos_ori = all_traj_obs_dict_of_list['state']

        print("all traj_pos_ori length: ", len(all_traj_pos_ori))

        for traj_idx in tqdm.tqdm(range(len(all_traj_pos_ori)), total=len(all_traj_pos_ori)):
            # import pdb; pdb.set_trace();
            traj_pos_ori = all_traj_pos_ori[traj_idx]
            traj_stage_length, traj_store_label_path  = all_traj_stage_lengths[traj_idx], all_traj_store_label_paths[traj_idx]
            
            print(f"starting to save traj {traj_idx} with length {len(traj_pos_ori)}")

            good_traj = True
            failure_reason = "null"

            traj_actions = []
            quaternion_diffs = []
            opening_start_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact'] + traj_stage_length['close_gripper']
            after_contact_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact']
            opening_start_idx = opening_start_idx // args.combine_action_steps
            after_contact_idx = after_contact_idx // args.combine_action_steps
            filtered_data = defaultdict(list)
            filtered_rgbs = []

            base_pos = traj_pos_ori[0][:3]
            base_ori_6d = traj_pos_ori[0][3:9]
            base_finger_angle = traj_pos_ori[0][9]
            base_pos_ori = traj_pos_ori[0]
            base_rgb = all_traj_obs_dict_of_list['cam0_image'][traj_idx][0]
            
            for i in range(len(traj_pos_ori) - 1):
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
                        filter_action = True
                        
                if filter_action:
                    continue
                else:
                    action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
                    traj_actions.append(action)
                    # import pdb; pdb.set_trace();
                    for key in obs_keys + ["goal_gripper_pcd"]:
                        filtered_data[key].append(all_traj_obs_dict_of_list[key][traj_idx][i])

                    filtered_rgbs.append(base_rgb)
                    
                    base_pos_ori = traj_pos_ori[i+1]
                    base_rgb = all_traj_obs_dict_of_list['cam0_image'][traj_idx][i+1]
                    base_pos = target_pos
                    base_ori_6d = target_ori_6d
                    base_finger_angle = target_finger_angle
                    # base_dp3_pc = traj_dp3_pc[i+1]
        
            # plot the delta translation action distribution
            if traj_idx % 5 == 0:        
                try:
                    save_numpy_as_mp4(np.array(filtered_rgbs), os.path.join(demo_rgb_save_path, "demo_" + str(traj_idx) + ".mp4"))
                    plt.close("all")
                except:
                    pass

            path = os.path.join(traj_store_label_path, "label.json")
            # import pdb; pdb.set_trace();
            if not os.path.exists(path):
                try:
                    with open(path, 'w') as f:
                        json.dump({"good_traj": good_traj, "failure reason": failure_reason}, f)
                except:
                    pass

            print(f"finished parsing traj, right before saving data, good_traj is {good_traj}")
            if good_traj:
                for i in range(len(traj_actions)):
                    if i >= after_contact_idx:
                        traj_actions[i][-1] = args.close_gripper_action
                        
                experiment_path = traj_store_label_path
                all_demo_paths.append(experiment_path)
                data_save_path = os.path.join(save_path, experiment_path.split("/")[-1])
                beg = time.time()

                experiment_path = traj_store_label_path
    
                # Parse files into dictionaries
                try:
                    with open(os.path.join(experiment_path, "task_config.yaml"), "r") as f:
                        task_config = {}
                        for dl in yaml.safe_load(f):
                            task_config |= dl
                        sim_states['task_config'] = task_config
                    
                    label_path = os.path.join(experiment_path, "label.json")
                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            sim_states['label'] = json.load(f)                
                    # For the txt file, maybe just keep it as a float or a small list
                    angle_path = os.path.join(experiment_path, "opened_angle.txt")
                    if os.path.exists(angle_path):
                        with open(angle_path, "r") as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                            sim_states['opened_angles'] = [float(x) for x in lines]

                except Exception as e:
                    print(f"Error parsing structured data: {e}")

                # from matplotlib import pyplot as plt
                # goal_gripper_pcd = np.asarray(filtered_goal_gripper_pcds)[0,0]
                # gripper_pcd = np.asarray(filtered_gripper_pcds)[0,0]
                # pcd = np.asarray(filtered_point_clouds)[0,0]
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], c='gray', s=1, label='Point Cloud')
                # ax.scatter(gripper_pcd[:,0], gripper_pcd[:,1], gripper_pcd[:,2], c='blue', s=5, label='Gripper PCD')
                # ax.scatter(goal_gripper_pcd[:,0], goal_gripper_pcd[:,1], goal_gripper_pcd[:,2], c='red', s=5, label='Goal Gripper PCD')
                # ax.legend()
                # plt.show()

                heatmaps = None
                # if args.extract_pcds:
                #     filtered_extrinsics_np = np.array(filtered_extrinsics)
                #     filtered_intrinsics_np = np.array(filtered_intrinsics)
                #     filtered_goal_gripper_pcds_np = np.array(filtered_goal_gripper_pcds)
                #     T, C, _, _ = filtered_extrinsics_np.shape
                #     heatmaps = []
                #     for c_idx in range(C):
                #         heatmap_c = []
                #         for t_idx in range(T):
                #             heatmap_t_c = generate_heatmap_from_points(
                #                 filtered_goal_gripper_pcds_np[t_idx],
                #                 filtered_intrinsics_np[t_idx, c_idx],
                #                 filtered_extrinsics_np[t_idx, c_idx],
                #                 (256,256)
                #             )
                #             heatmap_c.append(heatmap_t_c)
                #         heatmap_c = np.concatenate(heatmap_c, axis=0)
                #         # heatmap_c = generate_heatmap_from_points(
                #         #     filtered_goal_gripper_pcds_np,
                #         #     filtered_intrinsics_np[:, c_idx],
                #         #     filtered_extrinsics_np[:, c_idx],
                #         #     (256,256)
                #         # )
                #         heatmaps.append(heatmap_c)
                #     heatmaps = np.stack(heatmaps, axis=1)
                #     heatmaps = heatmaps.transpose(0,1,3,4,2)

                save_data_to_hdf5_recorder(filtered_data, traj_actions, sim_states,
                                           save_dir=data_save_path)
                end = time.time()
                cprint("Finished saving data to {}.h5 using time {}".format(data_save_path, end-beg), "green")
            del filtered_data, filtered_rgbs
            
    with open(os.path.join(save_path, "all_demo_path.txt"), "w") as f:
        f.write("\n".join(all_demo_paths))


def save_data_to_hdf5_recorder(filtered_data, traj_actions, sim_states, save_dir): 
    """
    Saves filtered trajectory data into an HDF5 file formatted for imitation learning.
    """
    save_path = save_dir + ".h5"
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    h5_recorder = HDF5DataRecorder(save_path, mode='w', 
                                   virtual_keys=['cam0_extrinsic', 'cam0_intrinsic',
                                                 'cam1_extrinsic', 'cam1_intrinsic',
                                                 'cam2_intrinsic', 'cam0_plucker',
                                                 'cam1_plucker'], verify_static=True)
    timesteps = len(traj_actions)

    for t_idx in range(timesteps):
        # import pdb; pdb.set_trace();
        obs = {}
        obs['state'] = np.array(filtered_data['state'][t_idx], dtype=np.float32)
        obs['gripper_to_world'] = np.array(filtered_data['gripper_to_world'][t_idx], dtype=np.float32)
        for key, value_list in filtered_data.items():
            # import pdb; pdb.set_trace();
            if key.startswith('cam') or key in ['point_cloud', 'gripper_pcd', 'goal_gripper_pcd', 'heatmap']:
                val = np.array(value_list[t_idx])
                if 'image' in key or 'heatmap' in key:
                    obs[key] = val.astype(np.uint8)
                elif 'segmask' in key or 'segm' in key:
                    obs[key] = val
                elif 'extrinsic' in key or 'intrinsic' in key:
                    obs[key] = val
                elif 'pointmap' in key:
                    obs[key] = process_pointmap(val, compress=True)
                elif 'depth' in key:
                    obs[key] = process_depth(val, compress=True)
                elif 'plucker' in key:
                    obs[key] = process_plucker(val, compress=True)
                else:
                    obs[key] = val
        data = {
            'obs': obs,
            'action': {
                'delta': hybrid_delta_to_delta_actions(
                    np.array(traj_actions[t_idx], dtype=np.float32)[None, ...],
                    np.array(filtered_data['state'][t_idx], dtype=np.float32))[0],
                'hybrid': np.array(traj_actions[t_idx], dtype=np.float32),
            }
        }
        h5_recorder.append_observation(data)
    h5_recorder.save_static_dict(sim_states, "sim_states")
    h5_recorder.finalize()
    h5_recorder.close()

def main(folder_name, save_path, exp_name=None, env_name=None, extract_name=None):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    meta_info = {
        "folder_name": folder_name,
        "exp_name": exp_name,
    }
    meta_info.update(args.__dict__)
    with open(os.path.join(save_path, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=4)
    
    extract_demos_from_a_directory(folder_name, exp_name=exp_name, env_name=env_name, extract_name=extract_name, save_path=save_path)
    
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--observation_mode", type=str, default=['image_pointmap']) # , 'act3d_goal'
    args.add_argument("--filter_close_zero_action", type=int, default=1)
    args.add_argument("--min_finger_angle_diff", type=float, default=0.001)
    args.add_argument("--close_gripper_action", type=float, default=-0.006)
    args.add_argument("--combine_action_steps", type=int, default=2)
    args.add_argument("--min_opened_ratio", type=float, default=0.2)
    args.add_argument("--randomize_camera", type=int, default=0)
    args.add_argument("--noise_real_world_pcd", type=int, default=0)
    args.add_argument("--real_world_camera", type=int, default=0)
    args.add_argument("--extract_pcds", action='store_true')
    args.add_argument("--save_path", type=str, default='data/rgb_data_mino/')
    args.add_argument("--folder_name", type=str, required=True)
    args.add_argument("--exp_name", type=str, default=None)
    args.add_argument("--env_name", type=str, default="articulated")
    args.add_argument("--extract_name", type=str, default=None)
    args.add_argument("--num_experiment", type=int, default=10000)
    args.add_argument("--traj_idx", type=int, default=None)
    args.add_argument("--angle_threshold", type=float)
    args = args.parse_args()

    save_path = args.save_path
    if args.randomize_camera == 1:
        save_path = save_path.replace('rgb', 'rgb_camera_left_right_randomized')
    elif args.randomize_camera == 2:
        save_path = save_path.replace('rgb', 'rgb_camera_randomized')
    elif args.randomize_camera == 3:
        save_path = save_path.replace('rgb', 'rgb_camera_conservative')
    save_path = save_path + args.extract_name + "/"

    main(args.folder_name, save_path, exp_name=args.exp_name, env_name=args.env_name, extract_name=args.extract_name)
    os._exit(0)  # bypass C-extension (open3d/pybullet) double-free during interpreter shutdown
