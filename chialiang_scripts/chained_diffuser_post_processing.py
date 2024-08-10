from manipulation.utils import get_pc, get_pc_in_camera_frame, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D, add_sphere, get_pixel_location, get_matrix_from_pos_rot
import argparse, os, json, sys, glob, time, datetime
from scipy.spatial.transform import Rotation as R
import zarr
import numpy as np
import open3d as o3d
from termcolor import cprint
import fpsample
import shutil
import subprocess
from tqdm import tqdm

USED_KEYS = [
    'state',
    'point_cloud',
    'action',
    'gripper_pcd',
    'goal_gripper_pcd',
    'displacement_gripper_to_object',
    'feature_map',
    'pcd_mask',
]

def waypoint_10d_to_8d(waypoint):

    assert len(waypoint) == 10, f'{len(waypoint)} != 10'

    pos = waypoint[:3]
    rot_6d = waypoint[3:9]
    target_joint_angle = waypoint[9]

    rotate_matrix = rotation_transfer_6D_to_matrix(rot_6d)
    quat = R.from_matrix(rotate_matrix).as_quat()
    
    return np.concatenate([pos, quat, [target_joint_angle]], axis=0)

def waypoint_plus_delta(current_wpt, delta_wpt):

    assert len(current_wpt) == len(delta_wpt), f'{len(current_wpt)} != {len(delta_wpt)}'

    ret = None
    if len(current_wpt) == 10:
        
        # pos info
        pos = np.asarray(current_wpt[:3]) + np.asarray(delta_wpt[:3])

        # rot info
        current_rotate_matrix = rotation_transfer_6D_to_matrix(current_wpt[3:9])
        delta_rotate_matrix = rotation_transfer_6D_to_matrix(delta_wpt[3:9])
        after_rotate_matrix = current_rotate_matrix @ delta_rotate_matrix
        
        quat = R.from_matrix(after_rotate_matrix).as_quat()

        # gripper info
        target_joint_angle = current_wpt[9] + delta_wpt[9]

        ret = np.concatenate([pos, quat, [target_joint_angle]], axis=0)
    
    else:
        raise NotImplementedError

    return ret  # 8D waypoint


def post_process(input_dir, output_dir, prefix=''):

    # group_names = os.listdir(input_dir)
    group_paths = glob.glob(f'{input_dir}/*')
    group_paths.sort()

    # group_paths = ['/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48876/2024-07-07-10-48-32']
    for group_path in group_paths:

        if '2024' not in group_path:

            if os.path.isdir(group_path):

                fname = group_path.split('/')[-1]
                if not os.path.exists(f'{output_dir}/{fname}'):
                
                    print(f'from {group_path}, {output_dir}/{fname}')
                    shutil.copytree(group_path, f'{output_dir}/{fname}')
            
            else :

                fname = group_path.split('/')[-1]
                if not os.path.exists(f'{output_dir}/{fname}'):

                    print(f'from {group_path}, {output_dir}/{fname}')
                    shutil.copy(group_path, f'{output_dir}/{fname}')
            
            continue

        group_name = group_path.split('/')[-1]
        
        output_zarr_dir = os.path.join(output_dir, group_name)
        if os.path.exists(output_zarr_dir): 
            
            if len(os.listdir(output_dir)) == len(os.listdir(input_dir)):
                cprint(f'{output_zarr_dir} has been written', 'green')
                continue

        wpt_groups = os.listdir(group_path)
        wpt_groups = sorted(wpt_groups, key = lambda x: int(x))

        # find goal id
        grasping_end_step = 0
        current_goal_gripper_pcd = None
        absolute_wpts = []
        for wpt_group in tqdm(wpt_groups):

            zarr_path = f'{group_path}/{wpt_group}'
            group = zarr.open(zarr_path, mode='r')
            data_group = group['data']

            # essential info 
            original_goal_gripper_pcd = np.asarray(data_group['goal_gripper_pcd'])
            original_state = np.asarray(data_group['state']).reshape(-1)
            original_action = np.asarray(data_group['action']).reshape(-1)

            if current_goal_gripper_pcd is None:
                current_goal_gripper_pcd = original_goal_gripper_pcd

            # find goal switching point
            if (grasping_end_step == 0) and (np.linalg.norm(current_goal_gripper_pcd - original_goal_gripper_pcd) > 0.001):
                grasping_end_step = int(wpt_group)
            
            # add waypoints that connects from init to end
            if int(wpt_group) == 0:
                original_state_8d = waypoint_10d_to_8d(original_state)
                absolute_wpts.append(original_state_8d) # first waypoint is the initial state

            wpt_plus_delta = waypoint_plus_delta(original_state, original_action) # 8D waypoint in world corrdinate frame
            absolute_wpts.append(wpt_plus_delta) 
        
        absolute_wpts = np.array(absolute_wpts) # (T + 1) x 8
        assert grasping_end_step != 0, f'invalid grasping_end_step {grasping_end_step}'


        if not os.path.exists(os.path.join(output_dir, group_name)):
            os.makedirs(os.path.join(output_dir, group_name))

        ###########################
        # get before grasping obs #
        ###########################

        zarr_path = f'{group_path}/0'
        group = zarr.open(zarr_path, mode='r')

        # one trajectory
        data_group = group['data']
        data = {}

        for key in USED_KEYS:
            data[key] = np.array(data_group[key])

        first_pcd = data['point_cloud']
        first_gripper_pcd = data['gripper_pcd']
        first_goal_gripper_pcd = data['goal_gripper_pcd']
        first_displacement_gripper_to_object = data['displacement_gripper_to_object']
        first_feature_map = data['feature_map']
        first_pcd_mask = data['pcd_mask']
        first_trajectory = absolute_wpts[:(grasping_end_step + 1)][None, :]
        first_init_pose = first_trajectory[0, 0][None, :]
        first_end_pose = first_trajectory[0, -1][None, :]

        cprint(first_pcd.shape, 'green')
        cprint(first_gripper_pcd.shape, 'green')
        cprint(first_goal_gripper_pcd.shape, 'green')
        cprint(first_displacement_gripper_to_object.shape, 'green')
        cprint(first_feature_map.shape, 'green')
        cprint(first_pcd_mask.shape, 'green')
        cprint(first_trajectory.shape, 'green')
        cprint(first_init_pose.shape, 'green')
        cprint(first_end_pose.shape, 'green')

        output_zarr_dir = os.path.join(output_dir, group_name, '0')
        save_data(output_zarr_dir, 
                    first_pcd,
                    first_gripper_pcd,
                    first_goal_gripper_pcd,
                    first_displacement_gripper_to_object,
                    first_feature_map,
                    first_pcd_mask,
                    first_trajectory,
                    first_init_pose,
                    first_end_pose,
                )

        ##########################
        # get after grasping obs #
        ##########################

        zarr_path = f'{group_path}/{grasping_end_step}'
        group = zarr.open(zarr_path, mode='r')

        # one trajectory
        data_group = group['data']
        data = {}

        for key in USED_KEYS:
            data[key] = np.array(data_group[key])

        second_pcd = data['point_cloud']
        second_gripper_pcd = data['gripper_pcd']
        second_goal_gripper_pcd = data['goal_gripper_pcd']
        second_displacement_gripper_to_object = data['displacement_gripper_to_object']
        second_feature_map = data['feature_map']
        second_pcd_mask = data['pcd_mask']
        second_trajectory = absolute_wpts[grasping_end_step:][None, :]
        second_init_pose = second_trajectory[0, 0][None, :]
        second_end_pose = second_trajectory[0, -1][None, :]

        cprint(second_pcd.shape, 'green')
        cprint(second_gripper_pcd.shape, 'green')
        cprint(second_goal_gripper_pcd.shape, 'green')
        cprint(second_displacement_gripper_to_object.shape, 'green')
        cprint(second_feature_map.shape, 'green')
        cprint(second_pcd_mask.shape, 'green')
        cprint(second_trajectory.shape, 'green')
        cprint(second_init_pose.shape, 'green')
        cprint(second_end_pose.shape, 'green')

        output_zarr_dir = os.path.join(output_dir, group_name, '1')
        save_data(output_zarr_dir, 
                    second_pcd,
                    second_gripper_pcd,
                    second_goal_gripper_pcd,
                    second_displacement_gripper_to_object,
                    second_feature_map,
                    second_pcd_mask,
                    second_trajectory,
                    second_init_pose,
                    second_end_pose,
                )
        
def save_data(output_zarr_dir, 
                point_cloud_array, 
                gripper_pcds_array, 
                goal_gripper_pcds_array, 
                displacement_gripper_to_objects_array,
                feature_map_array,
                pcd_mask_array,
                trajectory_array,
                init_pose_array,
                end_pose_array,
            ):

    if os.path.exists(output_zarr_dir):
        cprint('{} exists'.format(output_zarr_dir), 'yellow')
        cmd = "rm -r " + output_zarr_dir
        cprint  (f'{output_zarr_dir} deleted', 'red')
        exit(0)
        os.system(cmd)
    
    os.makedirs(output_zarr_dir)

    zarr_root = zarr.group(output_zarr_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    chunk_size = 1
    point_cloud_chunk_size = (chunk_size, point_cloud_array.shape[1], point_cloud_array.shape[2])
    gripper_pcd_chunk_size = (chunk_size, gripper_pcds_array.shape[1], gripper_pcds_array.shape[2])
    goal_gripper_pcd_chunk_size = (chunk_size, goal_gripper_pcds_array.shape[1], goal_gripper_pcds_array.shape[2])
    displacement_gripper_to_objects_chunk_size = (chunk_size, displacement_gripper_to_objects_array.shape[1], displacement_gripper_to_objects_array.shape[2])
    feature_map_chunk_size = (chunk_size, feature_map_array.shape[2], feature_map_array.shape[2], feature_map_array.shape[3], feature_map_array.shape[4])
    pcd_mask_chunk_size = (chunk_size, pcd_mask_array.shape[1])
    trajectory_chunk_size = (chunk_size, trajectory_array.shape[1], trajectory_array.shape[2])
    init_pose_chunk_size = (chunk_size, init_pose_array.shape[1])
    end_pose_chunk_size = (chunk_size, end_pose_array.shape[1])
    
    # ====== #
    # saving #
    # ====== #
    zarr_data.create_dataset('point_cloud', data=point_cloud_array, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('gripper_pcd', data=gripper_pcds_array, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcds_array, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_objects_array, chunks=displacement_gripper_to_objects_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('feature_map', data=feature_map_array, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('pcd_mask', data=pcd_mask_array, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('trajectory', data=trajectory_array, chunks=trajectory_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('init_pose', data=init_pose_array, chunks=init_pose_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('target_pose', data=end_pose_array, chunks=end_pose_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    cprint(f'{output_zarr_dir} has been written', 'green')

def fix_goal(demo_path, zarr_path):

    all_subfolder = os.listdir(zarr_path)
    for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
        if string in all_subfolder:
            cprint(f'{string} has been removed', 'green')
            all_subfolder.remove(string)
            
    all_subfolder = sorted(all_subfolder)

    keys = ['state', 'action', 'point_cloud', ]
    keys += ['gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']
    combine_action_steps = 2

    for traj in tqdm(all_subfolder, desc='processing'):

        zarr_path_traj = os.path.join(zarr_path, traj)
        exp_name = traj

        if not os.path.exists(os.path.join(demo_path, exp_name)):
            cprint('{} not exists'.format(os.path.join(demo_path, exp_name)), 'yellow')
            cmd = "rm -r " + zarr_path_traj
            # os.system(cmd)
            cprint(f'{zarr_path_traj} deleted', 'red')
            continue

        all_subfolder = os.listdir(os.path.join(demo_path, exp_name))
        all_subfolder = [d for d in all_subfolder if os.path.isdir(os.path.join(demo_path, exp_name, d))]
        d = all_subfolder[0]
        stage_lengths_json_file = os.path.join(demo_path, exp_name, d, 'stage_lengths.json')
        with open(stage_lengths_json_file, 'r') as f:
            stage_lengths = json.load(f)
        new_opening_start_idx = (stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]) // combine_action_steps    
        
        all_steps = os.listdir(zarr_path_traj)
        all_steps = sorted(all_steps, key=lambda x: int(x))
        
        last_step = all_steps[-1]
        zarr_path_step = os.path.join(zarr_path_traj, last_step)
        group = zarr.open(zarr_path_step, 'r')
        src_store = group.store
        src_root = zarr.group(src_store)
        goal_gripper_pcd_arr = src_root['data']["gripper_pcd"][:]

        # for every step after opening, change the goal gripper pcd
        for step in all_steps[new_opening_start_idx:]:
            zarr_path_step = os.path.join(zarr_path_traj, step)
            group = zarr.open(zarr_path_step, 'r+')
            src_store = group.store

            # numpy backend
            src_root = zarr.group(src_store)
            
            if np.linalg.norm(src_root['data']['goal_gripper_pcd'] - goal_gripper_pcd_arr) > 0.001:
                cprint(f'{zarr_path_step} has been modified', 'green')
                src_root['data']['goal_gripper_pcd'] = goal_gripper_pcd_arr


def main(arg):

    exp_dirs = [

        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-00/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-05/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-49/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-45/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
    
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45526/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45661/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45694/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45780/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45910/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45961/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46408/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46417/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46440/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46490/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46762/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46825/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46893/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47235/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47281/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47315/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47529/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47669/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47944/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48063/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48177/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48356/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48623/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48876/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49025/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49062/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49132/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49133/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_40417/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_41085/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_41452/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45162/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45176/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45194/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45203/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45248/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45271/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45290/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45305/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',

    ]

    src_dirs = [ 

        # '/scratch/yufei/dp3_demo/0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action',
        # '/scratch/yufei/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point',
        # '/scratch/yufei/dp3_demo/0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action',
        # '/scratch/yufei/dp3_demo/0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '/scratch/yufei/dp3_demo/0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '/scratch/yufei/dp3_demo/0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '/scratch/yufei/dp3_demo/0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '/scratch/yufei/dp3_demo/0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '/scratch/yufei/dp3_demo/0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '/scratch/yufei/dp3_demo/0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '/scratch/yufei/dp3_demo/0628-act3d-obj-48700-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45526',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45661',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45694',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45780',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45910',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45961',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46408',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46417',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46440',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46490',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46762',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46825',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46893',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47235',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47281',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47315',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47529',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47669',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47944',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48063',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48177',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48356',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48623',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48876',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49025',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49062',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49132',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49133',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-40417',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41085',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41452',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45162',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45176',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45194',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45203',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45248',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45271',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45290',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45305',
    ]

    dst_dirs = [

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-41510-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45448-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46462-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46732-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46801-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46874-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46922-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46966-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47570-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47578-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48700-chained-diffuser',
        
        # # # g1
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45526-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45661-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45694-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45780-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45910-chained-diffuser',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45961-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46408-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46417-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46440-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46490-chained-diffuser',

        # # g2
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46762-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46825-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46893-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47235-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47281-chained-diffuser',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47315-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47529-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47669-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47944-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48063-chained-diffuser',
        
        # # g3
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48177-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48356-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48623-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48876-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49025-chained-diffuser',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49062-chained-diffuser',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49132-chained-diffuser',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49133-chained-diffuser',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-40417-chained-diffuser',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41085-chained-diffuser',

        # # g4
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41452-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45162-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45176-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45194-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45203-chained-diffuser',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45248-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45271-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45290-chained-diffuser',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45305-chained-diffuser',
    ]

    # for (exp_dir, src_dir) in zip(exp_dirs, src_dirs):
    #     fix_goal(exp_dir, src_dir)

    for (src_dir, dst_dir) in zip(src_dirs, dst_dirs):
        post_process(src_dir, dst_dir)

    print('process completed')

    # 0705-obj-41510
    # 0705-obj-45526
    # 0705-obj-46462
    # 0705-obj-46732
    # 0705-obj-46801
    # 0705-obj-46874
    # 0705-obj-46922
    # 0705-obj-46966
    # 0705-obj-47570
    # 0705-obj-47578
    # 0705-obj-48700




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data_post_processing')
    # parser.add_argument('--input_dir', '-id', type=str, default='/scratch/yufei/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point')
    # parser.add_argument('--input_dir', '-id', type=str, default='/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-48700-goal')
    # parser.add_argument('--input_dir', '-id', type=str, default='dp3_demo/0701-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_part')
    parser.add_argument('--input_dir', '-id', type=str, default='/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-46966-goal')
    args = parser.parse_args()
    main(args)
