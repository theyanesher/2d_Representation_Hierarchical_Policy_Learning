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

    # group_paths = ['/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0705-obj-48876/2024-07-07-10-48-32']
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


    group_paths = glob.glob(f'{input_dir}/*')
    group_paths.sort()

    # group_paths = [
    #     # '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0705-obj-48876-whole-mlp/2024-07-07-10-00-07',
    #     # '/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-46966-goal/2024-06-27-17-57-07',
    #     # '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0705-obj-45526-whole-mlp/2024-07-06-01-32-05'
    #     '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0712-obj-41452-whole-mlp/2024-07-12-03-58-29'
    # ]
    # # output_dir = '/scratch/chialiang/dp3_demo/0705-obj-48876'
    # # output_dir = '/scratch/chialiang/dp3_demo/0705-obj-46966'
    # # output_dir = '/scratch/chialiang/dp3_demo/0705-obj-45526'
    # output_dir = '/scratch/chialiang/dp3_demo/0712-obj-41452'

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


def check_goal(input_dir, prefix=''):

    # singularity shell --bind /ocean/projects/cis240052p/ckuo1/RoboGen-sim2real:/mnt/RoboGen_sim2real/ --nv /ocean/projects/cis240052p/ywang59/robogen-dp3-act3d.sif

    # group_names = os.listdir(input_dir)
    traj_dirs = glob.glob(f'{input_dir}/2024*')
    traj_dirs.sort()

    time_stamp = datetime.datetime.now().strftime("%Y-%m%d-%H%M%S")
    out_json_path = f'debug/goal_miss_{time_stamp}.json'
    out_dict = {
        'black_list': []
    }

    for traj_dir in traj_dirs:
        cprint(f'processing {traj_dir}', 'yellow')

        group_paths = glob.glob(f'{traj_dir}/*')
        group_paths = sorted(group_paths, key=lambda x: int(x.split('/')[-1]))

        goal_switch = False
        for wpt_id, group_path in enumerate(group_paths):


            group_name = group_path.split('/')[-1]

            group = zarr.open(group_path, mode='r')
            # one trajectory
            data_group = group['data']

            original_goal_gripper_pcd = np.asarray(data_group['goal_gripper_pcd'])

            if wpt_id == 0:
                goal_gripper_pcd = original_goal_gripper_pcd
            else :
                if np.linalg.norm(goal_gripper_pcd - original_goal_gripper_pcd) > 0.001:
                    cprint(f'goal switching at {wpt_id}', 'green')
                    goal_switch = True
                    break
            
        if not goal_switch:
            cprint(f'no goal switch found in {group_path}', 'red')
            out_dict['black_list'].append(group_path)
        
    json.dump(out_dict, open(out_json_path, 'w'), indent=4)
    cprint(f'{out_json_path} has been written')

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

    # Load the data
    input_dirs = [
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45413',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45420',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45427',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45594',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45620',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45623',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45636',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45670',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45689',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45696',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45749',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45759',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45916',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45936',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45950',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-45984',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46092',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46130',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46134',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46197',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46401',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46456',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46480',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46481',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46544',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-46641',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-47178',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-47182',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-47227',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-47577',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-47648',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-47747',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-47808',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-47976',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-48010',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-48258',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-48379',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-48797',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-48855',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-48859',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0725-obj-49188',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-35059',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-41004',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-41083',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-41529',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-44781',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-44826',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-44853',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45092',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45130',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45135',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45146',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45164',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45168',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45173',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45212',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45213',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45372',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45374',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45387',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45415',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45419',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45423',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45503',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45505',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45524',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45573',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45575',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45606',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45612',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45621',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45622',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45632',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45638',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45645',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45662',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45671',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45676',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45677',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45687',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45699',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45710',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45746',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45756',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45783',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45784',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45790',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45801',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45822',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45853',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45855',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45915',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45948',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45949',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45963',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-45964',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46002',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46019',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46029',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46033',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46037',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46044',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46045',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46060',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46084',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46108',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46117',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46120',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46123',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46145',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46179',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46180',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46199',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46230',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46277',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46380',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46427',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46430',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46439',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46466',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46537',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46549',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46556',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46598',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46616',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46699',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46700',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46741',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46744',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46847',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46856',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46859',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46889',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46906',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46944',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46955',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-46981',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47024',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47089',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47183',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47207',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47233',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47252',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47278',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47290',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47296',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47438',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47514',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47595',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47601',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47632',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47701',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47729',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47853',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-47926',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48051',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48413',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48452',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48467',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48490',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48513',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48517',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48721',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48746',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-48878',
        '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/data/dp3_demo/0730-obj-49140',
    ]

    replace = '/jet/projects/cis240052p/ywang59/dp3_demo/'
    for input_dir in tqdm(input_dirs):

        # check_goal(input_dir)

        last = input_dir.split('/')[-1]
        pickle_dir = f'{replace}{last}'

        zarr_paths = glob.glob(f'{input_dir}/2024*')
        pickle_paths = glob.glob(f'{pickle_dir}/2024*')

        if len(zarr_paths) != len(pickle_paths):
            print('{}, len(zarr_paths)={}, len(pickle_paths)={}'.format(pickle_dir, len(zarr_paths), len(pickle_paths)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data_post_processing')
    # parser.add_argument('--input_dir', '-id', type=str, default='/scratch/yufei/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point')
    # parser.add_argument('--input_dir', '-id', type=str, default='/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-48700-goal')
    # parser.add_argument('--input_dir', '-id', type=str, default='dp3_demo/0701-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_part')
    parser.add_argument('--input_dir', '-id', type=str, default='/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-46966-goal')
    args = parser.parse_args()
    main(args)
