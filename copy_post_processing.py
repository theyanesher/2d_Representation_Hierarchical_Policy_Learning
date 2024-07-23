import argparse, os, json, sys, glob, time, datetime
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
]

def rsync_files(input_dir, output_dir, prefix=''):

    # group_names = os.listdir(input_dir)
    group_paths = glob.glob(f'{input_dir}/*')
    group_paths.sort()

    for group_path in group_paths:

        if '2024' not in group_path:

            fname = group_path.split('/')[-1]
            if os.path.isdir(fname):
                subprocess.run(['cp', '-r', f'{group_path}', f'{output_dir}/{fname}/'])
            else:
                subprocess.run(['cp', f'{group_path}', f'{output_dir}/{fname}/'])

        group_name = group_path.split('/')[-1]
        wpt_groups = os.listdir(group_path)

        wpt_groups = sorted(wpt_groups, key = lambda x: int(x))

        for wpt_group in tqdm(wpt_groups):

            group_name = group_path.split('/')[-1]

            # cprint('===============', 'green')

            zarr_path = f'{group_path}/{wpt_group}'

            output_wpt_group = f'{output_dir}/{group_name}/{wpt_group}'

            os.makedirs(output_wpt_group, exist_ok=True)
            # cprint(f'{output_wpt_group} created', 'green')

            # zgroup
            src = f'{zarr_path}/.zgroup'
            subprocess.run(['cp', f'{src}', f'{output_wpt_group}/'])
            # cprint(f'from {src} to {output_wpt_group}', 'green')

            # meta
            src = f'{zarr_path}/meta'
            subprocess.run(['cp', '-r', f'{src}', f'{output_wpt_group}/'])
            # cprint(f'from {src} to {output_wpt_group}', 'green')

            # data
            output_wpt_group_data = f'{output_wpt_group}/data'
            os.makedirs(output_wpt_group_data, exist_ok=True)
            # cprint(f'{output_wpt_group_data} created', 'green')
            
            for key in USED_KEYS:
                src = f'{zarr_path}/data/{key}'
                dst = output_wpt_group_data
                subprocess.run(['cp', '-r', f'{src}', f'{dst}/'])
                # cprint(f'{dst} created', 'green')

            # exit(0)

def copy_per_step(input_dir, output_dir, prefix=''):

    # group_names = os.listdir(input_dir)
    group_paths = glob.glob(f'{input_dir}/*')
    group_paths.sort()

    for group_path in group_paths:

        if '2024' not in group_path:

            if os.path.isdir(group_path):

                if not os.path.exists(f'{output_dir}/{fname}'):
                    fname = group_path.split('/')[-1]
                
                    print(f'from {group_path}, {output_dir}/{fname}')
                    shutil.copytree(group_path, f'{output_dir}/{fname}')
            
            else :

                if not os.path.exists(f'{output_dir}/{fname}'):

                    fname = group_path.split('/')[-1]
                    print(f'from {group_path}, {output_dir}/{fname}')
                    shutil.copy(group_path, f'{output_dir}/{fname}')


        group_name = group_path.split('/')[-1]
        wpt_groups = os.listdir(group_path)

        wpt_groups = sorted(wpt_groups, key = lambda x: int(x))

        new_states = []
        new_pcds = []
        new_actions = []

        new_gripper_pcds = []
        new_goal_gripper_pcds = []
        new_displacement_gripper_to_objects = []

        for wpt_group in tqdm(wpt_groups):

            zarr_path = f'{group_path}/{wpt_group}'

            output_zarr_dir = os.path.join(output_dir, group_name, wpt_group)
            if os.path.exists(output_zarr_dir): 
                cprint(f'{output_zarr_dir} has been overwritten', 'green')
                continue
                os.system('rm -rf {}'.format(output_zarr_dir))
                cprint(f'{output_zarr_dir} has been overwritten', 'red')

            group = zarr.open(zarr_path, mode='r')

            # one trajectory
            data_group = group['data']
            data = {}

            for key in USED_KEYS:
                data[key] = np.array(data_group[key])

            original_state = data['state']
            original_pcd = data['point_cloud']
            original_action = data['action']
            original_gripper_pcd = data['gripper_pcd']
            original_goal_gripper_pcd = data['goal_gripper_pcd']
            original_displacement_gripper_to_object = data['displacement_gripper_to_object']

            ###############################################################################
            # save img, state, action arrays into data, and episode ends arrays into meta #
            ###############################################################################
            state_arrays = original_state
            point_cloud_arrays = original_pcd
            action_arrays = original_action
            gripper_pcd_arrays = original_gripper_pcd
            goal_gripper_pcd_arrays = original_goal_gripper_pcd
            original_displacement_gripper_to_object_arrays = original_displacement_gripper_to_object

            # ===================== #
            # config saving options #
            # ===================== #
            
            zarr_root = zarr.group(output_zarr_dir)
            zarr_data = zarr_root.create_group('data')
            zarr_meta = zarr_root.create_group('meta')

            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            state_chunk_size = (1, state_arrays.shape[0])
            # feature_map_chunk_size = (1, feature_maps.shape[0], feature_maps.shape[1], feature_maps.shape[2])
            point_cloud_chunk_size = (1, point_cloud_arrays.shape[0], point_cloud_arrays.shape[1])
            # pcd_mask_chunk_size = (1, pcd_masks.shape[0])
            action_chunk_size = (1, action_arrays.shape[0])
            gripper_pcd_chunk_size = (1, gripper_pcd_arrays.shape[0], gripper_pcd_arrays.shape[1])
            goal_gripper_pcd_chunk_size = (1, goal_gripper_pcd_arrays.shape[0], goal_gripper_pcd_arrays.shape[1])
            original_displacement_gripper_to_object_chunk_size = (1, original_displacement_gripper_to_object_arrays.shape[0], original_displacement_gripper_to_object_arrays.shape[1])
            
            # ====== #
            # saving #
            # ====== #
            zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            # zarr_data.create_dataset('feature_map', data=feature_maps, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            # zarr_data.create_dataset('pcd_mask', data=pcd_masks, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd_arrays, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('displacement_gripper_to_object', data=original_displacement_gripper_to_object_arrays, chunks=original_displacement_gripper_to_object_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

            del state_arrays, point_cloud_arrays, action_arrays, gripper_pcd_arrays, goal_gripper_pcd_arrays, original_displacement_gripper_to_object_arrays


def debug_action(input_dir, prefix=''):

    # group_names = os.listdir(input_dir)
    group_paths = glob.glob(f'{input_dir}/2024*')
    group_paths.sort()

    for group_path in group_paths:

        group_name = group_path.split('/')[-1]
        
        wpt_groups = os.listdir(group_path)
        wpt_groups = sorted(wpt_groups, key = lambda x: int(x))

        for wpt_group in tqdm(wpt_groups):

            zarr_path = f'{group_path}/{wpt_group}'

            group = zarr.open(zarr_path, mode='r')

            # one trajectory
            data_group = group['data']
            data = {}

            cprint(f'{zarr_path}: {list(data_group.keys())}', 'green')
            for key in USED_KEYS:
                data[key] = np.array(data_group[key])

def copy_per_step_to_all_step(input_dir, output_dir, prefix=''):

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

        new_states = []
        new_pcds = []
        new_actions = []

        new_gripper_pcds = []
        new_goal_gripper_pcds = []
        new_displacement_gripper_to_objects = []

        for wpt_group in tqdm(wpt_groups):

            zarr_path = f'{group_path}/{wpt_group}'

            group = zarr.open(zarr_path, mode='r')

            # one trajectory
            data_group = group['data']
            data = {}

            for key in USED_KEYS:
                data[key] = np.array(data_group[key])

            original_state = data['state']
            original_pcd = data['point_cloud']
            original_action = data['action']
            original_gripper_pcd = data['gripper_pcd']
            original_goal_gripper_pcd = data['goal_gripper_pcd']
            original_displacement_gripper_to_object = data['displacement_gripper_to_object']

            # cprint(original_state.shape, 'green')
            # cprint(original_pcd.shape, 'green')
            # cprint(original_action.shape, 'green')
            # cprint(original_gripper_pcd.shape, 'green')
            # cprint(original_goal_gripper_pcd.shape, 'green')
            # cprint(original_displacement_gripper_to_object.shape, 'green')
            # exit(0)

            new_states.append(original_state)
            new_pcds.append(original_pcd)
            new_actions.append(original_action)
            new_gripper_pcds.append(original_gripper_pcd)
            new_goal_gripper_pcds.append(original_goal_gripper_pcd)
            new_displacement_gripper_to_objects.append(original_displacement_gripper_to_object)
        
        # ########################################################################### #
        # save img, state, action arrays into data, and episode ends arrays into meta #
        # ########################################################################### #

        state_arrays = np.stack(new_states, axis=0).squeeze()
        point_cloud_arrays = np.stack(new_pcds, axis=0).squeeze()
        action_arrays = np.stack(new_actions, axis=0).squeeze()
        gripper_pcds_arrays = np.stack(new_gripper_pcds, axis=0).squeeze()
        goal_gripper_pcds_arrays = np.stack(new_goal_gripper_pcds, axis=0).squeeze()
        displacement_gripper_to_objects_arrays = np.stack(new_displacement_gripper_to_objects, axis=0).squeeze()

        # ===================== #
        # config saving options #
        # ===================== #
        
        zarr_root = zarr.group(output_zarr_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')

        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
        action_chunk_size = (100, action_arrays.shape[1])
        gripper_pcd_chunk_size = (100, gripper_pcds_arrays.shape[1], gripper_pcds_arrays.shape[2])
        goal_gripper_pcd_chunk_size = (100, goal_gripper_pcds_arrays.shape[1], goal_gripper_pcds_arrays.shape[2])
        displacement_gripper_to_objects_chunk_size = (100, displacement_gripper_to_objects_arrays.shape[1], displacement_gripper_to_objects_arrays.shape[2])
        
        # ====== #
        # saving #
        # ====== #
        zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('gripper_pcd', data=gripper_pcds_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcds_arrays, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_objects_arrays, chunks=displacement_gripper_to_objects_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

        cprint(f'{output_zarr_dir} has been written', 'green')

def copy_all_step_to_per_step(input_dir, output_dir, prefix=''):

    # group_names = os.listdir(input_dir)


    # group_paths = glob.glob(f'{input_dir}/*')
    # group_paths.sort()

    group_paths = [
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48876-whole-mlp/2024-07-07-10-00-07',
        # '/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-46966-goal/2024-06-27-17-57-07',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45526-whole-mlp/2024-07-06-01-32-05'
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41452-whole-mlp/2024-07-12-03-58-29'
    ]
    # # output_dir = '/scratch/chialiang/dp3_demo/0705-obj-48876'
    # # output_dir = '/scratch/chialiang/dp3_demo/0705-obj-46966'
    # # output_dir = '/scratch/chialiang/dp3_demo/0705-obj-45526'
    output_dir = '/scratch/chialiang/dp3_demo/0712-obj-41452'

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

        group = zarr.open(group_path, mode='r')
        # one trajectory
        data_group = group['data']
        data = {}

        for key in USED_KEYS:
            data[key] = np.array(data_group[key])

        original_state = data['state']
        original_pcd = data['point_cloud']
        original_action = data['action']
        original_gripper_pcd = data['gripper_pcd']
        original_goal_gripper_pcd = data['goal_gripper_pcd']
        original_displacement_gripper_to_object = data['displacement_gripper_to_object']

        # cprint(original_state.shape, 'green')
        # cprint(original_pcd.shape, 'green')
        # cprint(original_action.shape, 'green')
        # cprint(original_gripper_pcd.shape, 'green')
        # cprint(original_goal_gripper_pcd.shape, 'green')
        # cprint(original_displacement_gripper_to_object.shape, 'green')

        ###############################################################################
        # save img, state, action arrays into data, and episode ends arrays into meta #
        ###############################################################################
        
        traj_length = len(original_state)

        for wpt_id in tqdm(range(traj_length)):

            # ===================== #
            # config saving options #
            # ===================== #
            output_zarr_dir = os.path.join(output_dir, group_name, str(wpt_id))
            if os.path.exists(output_zarr_dir): 
                cprint(f'{output_zarr_dir} has been written', 'green')
                continue
                os.system('rm -rf {}'.format(output_zarr_dir))
                cprint(f'{output_zarr_dir} has been overwritten', 'red')
            
            zarr_root = zarr.group(output_zarr_dir)
            zarr_data = zarr_root.create_group('data')
            zarr_meta = zarr_root.create_group('meta')

            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            state_chunk_size = (1, original_state.shape[1])
            point_cloud_chunk_size = (1, original_pcd.shape[1], original_pcd.shape[2])
            action_chunk_size = (1, original_action.shape[1])
            gripper_pcd_chunk_size = (1, original_gripper_pcd.shape[1], original_gripper_pcd.shape[2])
            goal_gripper_pcd_chunk_size = (1, original_goal_gripper_pcd.shape[1], original_goal_gripper_pcd.shape[2])
            displacement_gripper_to_object_chunk_size = (1, original_displacement_gripper_to_object.shape[1], original_displacement_gripper_to_object.shape[2])
            
            # ====== #
            # saving #
            # ====== #
            state = original_state[wpt_id][np.newaxis, :]
            pcd = original_pcd[wpt_id][np.newaxis, :]
            action = original_action[wpt_id][np.newaxis, :]
            gripper_pcd = original_gripper_pcd[wpt_id][np.newaxis, :]
            goal_gripper_pcd = original_goal_gripper_pcd[wpt_id][np.newaxis, :]
            displacement_gripper_to_object = original_displacement_gripper_to_object[wpt_id][np.newaxis, :]
            
            # cprint(state.shape, 'green')
            # cprint(pcd.shape, 'green')
            # cprint(action.shape, 'green')
            # cprint(gripper_pcd.shape, 'green')
            # cprint(goal_gripper_pcd.shape, 'green')
            # cprint(displacement_gripper_to_object.shape, 'green')
            # exit(0)

            zarr_data.create_dataset('state', data=state, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('point_cloud', data=pcd, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('action', data=action, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('gripper_pcd', data=gripper_pcd, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_object, chunks=displacement_gripper_to_object_chunk_size, dtype='float32', overwrite=True, compressor=compressor)


def write_goal_pose_info(input_dir, prefix=''):

    # group_names = os.listdir(input_dir)
    group_paths = glob.glob(f'{input_dir}/2024*')
    group_paths.sort()

    # time_stamp = datetime.datetime.now().strftime("%Y-%m%d-%H%M%S")
    # out_json_path = f'debug/goal_miss_{time_stamp}.json'
    # out_dict = {
    #     'black_list': []
    # }

    for group_path in group_paths:

        cprint(f'processing {group_path}', 'yellow')

        group_name = group_path.split('/')[-1]

        group = zarr.open(group_path, mode='r')
        # one trajectory
        data_group = group['data']

        original_goal_gripper_pcd = data_group['goal_gripper_pcd']

        # goal_switch = False
        # goal_gripper_pcd = original_goal_gripper_pcd[0]
        # for wpt_i in range(1, original_goal_gripper_pcd.shape[0]):
        #     if np.linalg.norm(goal_gripper_pcd - original_goal_gripper_pcd[wpt_i]) > 0.001:
        #         cprint(f'goal switching at {wpt_i}', 'green')
        #         goal_switch = True
        #         break
        
        # if not goal_switch:
        #     cprint(f'no goal switch found in {group_path}', 'red')
        #     out_dict['black_list'].append(group_path)
    
    # json.dump(out_dict, open(out_json_path, 'w'), indent=4)
    # cprint(f'{out_json_path} has been written')

############################################

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

    # # read each step of each episode
    # all_trajs = os.listdir(zarr_path)
    # all_trajs = sorted(all_trajs)

    for traj in tqdm(all_subfolder, desc='processing'):

        zarr_path_traj = os.path.join(zarr_path, traj)
        exp_name = traj
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
            # if keys is None:
            #     keys = src_root['data'].keys()
            # data = dict()
            # for key in keys:
            #     arr = src_root['data'][key]
            #     data[key] = arr[:]
            
            if np.linalg.norm(src_root['data']['goal_gripper_pcd'] - goal_gripper_pcd_arr) > 0.001:
                cprint(f'{zarr_path_step} has been modified', 'green')
                src_root['data']['goal_gripper_pcd'] = goal_gripper_pcd_arr
        
            # # remove the old data
            # cmd = "rm -r " + zarr_path_step
            # os.system(cmd)
        
            # # save new data
            # new_data_save_dir = zarr_path_step
            # # print("Saving new data to: ", new_data_save_dir)
            # save_data(
            #         data['point_cloud'], 
            #         data['state'], 
            #         data['gripper_pcd'], 
            #         data['action'], 
            #         data['goal_gripper_pcd'], 
            #         data['displacement_gripper_to_object'],
            #         new_data_save_dir
            # )

def main(arg):

    # exp_dirs = [
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-00/task_open_the_door_of_the_storagefurniture_by_its_handle/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-05/task_open_the_door_of_the_storagefurniture_by_its_handle/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-49/task_open_the_door_of_the_storagefurniture_by_its_handle/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-45/task_open_the_door_of_the_storagefurniture_by_its_handle/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
    
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45526/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45661/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45694/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45780/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45910/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_45961/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46408/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46417/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46440/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46490/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46762/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46825/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_46893/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47235/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47281/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47315/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47529/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47669/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_47944/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48063/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48177/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48356/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48623/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_48876/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49025/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49062/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49132/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects/open_the_door_49133/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_40417/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_41085/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_41452/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45162/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45176/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45194/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45203/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45248/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45271/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
        # '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45290/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
    #     '/project_data/held/chialiak/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45305/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'

    # ]

    # output_dirs = [
    #     '/scratch/chialiang/dp3_demo/0712-obj-45305-copy',
    # ]

    # for (exp_dir, output_dir) in zip(exp_dirs, output_dirs):
    #     fix_goal(exp_dir, output_dir)
    # exit(0)

    # Load the data
    input_dirs = [

        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #

        '/scratch/yufei/dp3_demo/0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action'

        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45526',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45661',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45694',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45780',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45910',

        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45961',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46408',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46417',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46440',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46490',

        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46762',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46825',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46893',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47235',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47281',

        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47315',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47529',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47669',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47944',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48063',

        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48177',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48356',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48623',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48876',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49025',
        
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49062',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49132',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49133',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-40417',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41085',

        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41452',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45162',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45176',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45194',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45203',

        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45248',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45271',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45290',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45305',
    ]

    intermediate_dirs = [
        
        # G1
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-41510-whole-mlp'
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-45448-whole-mlp',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-46462-whole-mlp',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-46732-whole-mlp',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-46801-whole-mlp',

        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-46874-whole-mlp',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-46922-whole-mlp',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-46966-whole-mlp',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-47570-whole-mlp',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-47578-whole-mlp',
        '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0702-obj-48700-whole-mlp',

        # # G2
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45526-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45661-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45694-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45780-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45910-whole-mlp',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45961-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46408-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46417-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46440-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46490-whole-mlp',

        # # G3
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46762-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46825-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-46893-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47235-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47281-whole-mlp',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47315-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47529-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47669-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-47944-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48063-whole-mlp',

        # # G4
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48177-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48356-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48623-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-48876-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49025-whole-mlp',
        
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49062-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49132-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-49133-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-40417-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41085-whole-mlp',

        # # G5
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-41452-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45162-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45176-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45194-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45203-whole-mlp',

        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45248-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45271-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45290-whole-mlp',
        # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0712-obj-45305-whole-mlp',

    ]

    output_dirs = [

        # G1
        '/scratch/chialiang/dp3_demo/0705-obj-41510'
        '/scratch/chialiang/dp3_demo/0705-obj-45448',
        '/scratch/chialiang/dp3_demo/0705-obj-46462',
        '/scratch/chialiang/dp3_demo/0705-obj-46732',
        '/scratch/chialiang/dp3_demo/0705-obj-46801',

        '/scratch/chialiang/dp3_demo/0705-obj-46874',
        '/scratch/chialiang/dp3_demo/0705-obj-46922',
        '/scratch/chialiang/dp3_demo/0705-obj-46966',
        '/scratch/chialiang/dp3_demo/0705-obj-47570',
        '/scratch/chialiang/dp3_demo/0705-obj-47578',
        '/scratch/chialiang/dp3_demo/0705-obj-48700',
        
        # G2
        # '/scratch/chialiang/dp3_demo/0705-obj-45526',
        # '/scratch/chialiang/dp3_demo/0705-obj-45661',
        # '/scratch/chialiang/dp3_demo/0705-obj-45694',
        # '/scratch/chialiang/dp3_demo/0705-obj-45780',
        # '/scratch/chialiang/dp3_demo/0705-obj-45910',

        # '/scratch/chialiang/dp3_demo/0705-obj-45961',
        # '/scratch/chialiang/dp3_demo/0705-obj-46408',
        # '/scratch/chialiang/dp3_demo/0705-obj-46417',
        # '/scratch/chialiang/dp3_demo/0705-obj-46440',
        # '/scratch/chialiang/dp3_demo/0705-obj-46490',

        # G3
        # '/scratch/chialiang/dp3_demo/0705-obj-46762',
        # '/scratch/chialiang/dp3_demo/0705-obj-46825',
        # '/scratch/chialiang/dp3_demo/0705-obj-46893',
        # '/scratch/chialiang/dp3_demo/0705-obj-47235',
        # '/scratch/chialiang/dp3_demo/0705-obj-47281',

        # '/scratch/chialiang/dp3_demo/0705-obj-47315',
        # '/scratch/chialiang/dp3_demo/0705-obj-47529',
        # '/scratch/chialiang/dp3_demo/0705-obj-47669',
        # '/scratch/chialiang/dp3_demo/0705-obj-47944',
        # '/scratch/chialiang/dp3_demo/0705-obj-48063',
        
        # G4
        # '/scratch/chialiang/dp3_demo/0705-obj-48177',
        # '/scratch/chialiang/dp3_demo/0705-obj-48356',
        # '/scratch/chialiang/dp3_demo/0705-obj-48623',
        # '/scratch/chialiang/dp3_demo/0705-obj-48876',
        # '/scratch/chialiang/dp3_demo/0705-obj-49025',
        
        # '/scratch/chialiang/dp3_demo/0705-obj-49062',
        # '/scratch/chialiang/dp3_demo/0705-obj-49132',
        # '/scratch/chialiang/dp3_demo/0705-obj-49133',
        # '/scratch/chialiang/dp3_demo/0712-obj-40417',
        # '/scratch/chialiang/dp3_demo/0712-obj-41085',

        # G5
        # '/scratch/chialiang/dp3_demo/0712-obj-41452',
        # '/scratch/chialiang/dp3_demo/0712-obj-45162',
        # '/scratch/chialiang/dp3_demo/0712-obj-45176',
        # '/scratch/chialiang/dp3_demo/0712-obj-45194',
        # '/scratch/chialiang/dp3_demo/0712-obj-45203',

        # '/scratch/chialiang/dp3_demo/0712-obj-45248',
        # '/scratch/chialiang/dp3_demo/0712-obj-45271',
        # '/scratch/chialiang/dp3_demo/0712-obj-45290',
        # '/scratch/chialiang/dp3_demo/0712-obj-45305',

    ]

    for output_dir in output_dirs:
        debug_action(output_dir)
    exit(0)

    # for (input_dir, intermediate_dir, output_dir) in zip(input_dirs, intermediate_dirs, output_dirs):
    for (intermediate_dir, output_dir) in zip(intermediate_dirs, output_dirs):
        # copy_per_step_to_all_step(input_dir, intermediate_dir)
        # cprint(f'writing to {output_dir}', 'green')
        copy_all_step_to_per_step(intermediate_dir, output_dir)
        
        # write_goal_pose_info(intermediate_dir)

    # for (input_dir, output_dir) in zip(input_dirs, output_dirs):
    #     copy_per_step(input_dir, output_dir)


    print('process completed')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data_post_processing')
    # parser.add_argument('--input_dir', '-id', type=str, default='/scratch/yufei/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point')
    # parser.add_argument('--input_dir', '-id', type=str, default='/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-48700-goal')
    # parser.add_argument('--input_dir', '-id', type=str, default='dp3_demo/0701-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_part')
    parser.add_argument('--input_dir', '-id', type=str, default='/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-46966-goal')
    args = parser.parse_args()
    main(args)
