import argparse, os, json, sys, glob
import zarr
import numpy as np
import open3d as o3d
from termcolor import cprint

from tqdm import tqdm

DATA_KEYS = [   
                'action', 
                # 'feature_map', 
                'gripper_pcd', 
                'goal_gripper_pcd',
                # 'pcd_mask', 
                'point_cloud', 
                'state'
            ]

def post_process_per_step(input_dir, output_dir, prefix=''):

    # group_names = os.listdir(input_dir)
    group_paths = glob.glob(f'{input_dir}/2024*')
    group_paths.sort()

    for group_path in group_paths:

        group_name = group_path.split('/')[-1]
        wpt_groups = os.listdir(group_path)

        for wpt_group in wpt_groups:

            zarr_path = f'{group_path}/{wpt_group}'
            cprint(f'processing {zarr_path}', 'green')

            group = zarr.open(zarr_path, mode='r')

            # one trajectory
            data_group = group['data']
            data = {}

            for key in data_group:
                data[key] = np.array(data_group[key])

            original_state = data['state']
            original_pcd = data['point_cloud']
            original_action = data['action']
            # original_feature_map = data['feature_map']
            # original_pcd_mask = data['pcd_mask']
            original_gripper_pcd = data['gripper_pcd']
            original_goal_gripper_pcd = data['goal_gripper_pcd']

            diff = original_goal_gripper_pcd - original_gripper_pcd
            diff_flat = diff.reshape(1, -1)
            new_state = np.concatenate((original_state, diff_flat), axis=1)

            ###############################################################################
            # save img, state, action arrays into data, and episode ends arrays into meta #
            ###############################################################################
            state_arrays = new_state
            # state_arrays = original_state
            point_cloud_arrays = original_pcd
            # pcd_masks = original_pcd_mask
            action_arrays = original_action
            # gripper_pcds = original_gripper_pcd

            # ===================== #
            # config saving options #
            # ===================== #
                
            output_zarr_dir = os.path.join(output_dir, group_name, wpt_group)
            if os.path.exists(output_zarr_dir): 
                cprint(f'{output_zarr_dir} has been overwritten', 'green')
                continue
                os.system('rm -rf {}'.format(output_zarr_dir))
                cprint(f'{output_zarr_dir} has been overwritten', 'red')
            
            zarr_root = zarr.group(output_zarr_dir)
            zarr_data = zarr_root.create_group('data')
            zarr_meta = zarr_root.create_group('meta')

            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            state_chunk_size = (1, state_arrays.shape[0])
            # feature_map_chunk_size = (1, feature_maps.shape[0], feature_maps.shape[1], feature_maps.shape[2])
            point_cloud_chunk_size = (1, point_cloud_arrays.shape[0], point_cloud_arrays.shape[1])
            # pcd_mask_chunk_size = (1, pcd_masks.shape[0])
            action_chunk_size = (1, action_arrays.shape[0])
            # gripper_pcd_chunk_size = (1, gripper_pcds.shape[0], gripper_pcds.shape[1])
            # goal_gripper_pcd_chunk_size = (1, goal_gripper_pcds.shape[0], goal_gripper_pcds.shape[1])
            
            # ====== #
            # saving #
            # ====== #
            zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            # zarr_data.create_dataset('feature_map', data=feature_maps, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            # zarr_data.create_dataset('pcd_mask', data=pcd_masks, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            # zarr_data.create_dataset('gripper_pcd', data=gripper_pcds, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
            # zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcds, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

def post_process(input_dir, output_dir, prefix=''):

    # group_names = os.listdir(input_dir)
    group_paths = glob.glob(f'{input_dir}/2024*')
    group_paths.sort()

    for group_path in group_paths:

        cprint(f'processing {group_path}', 'green')

        group_name = group_path.split('/')[-1]

        group = zarr.open(group_path, mode='r')
        # one trajectory
        data_group = group['data']
        data = {}

        for key in data_group:
            data[key] = np.array(data_group[key])

        original_states = data['state']
        original_pcds = data['point_cloud']
        original_actions = data['action']
        # original_feature_maps = data['feature_map']
        # original_pcd_masks = data['pcd_mask']
        original_gripper_pcds = data['gripper_pcd']
        original_goal_gripper_pcds = data['goal_gripper_pcd']

        # For DP3 + MLP only
        new_states = []
        for i, (original_state, original_gripper_pcd, original_goal_gripper_pcd) in enumerate(zip(original_states, original_gripper_pcds, original_goal_gripper_pcds)):
            diff = original_goal_gripper_pcd - original_gripper_pcd
            diff_flat = diff.reshape(-1)

            new_state = np.concatenate((original_state, diff_flat), axis=0)

            new_states.append(new_state)
        new_states = np.stack(new_states, axis=0)

        ###############################################################################
        # save img, state, action arrays into data, and episode ends arrays into meta #
        ###############################################################################
        state_arrays = new_states
        # state_arrays = original_states
        # feature_maps = np.stack(new_feature_maps, axis=0)
        point_cloud_arrays = original_pcds
        # pcd_masks = original_pcd_masks
        action_arrays = original_actions
        # gripper_pcds = original_gripper_pcds
        # goal_gripper_pcds = np.stack(goal_gripper_pcds, axis=0)

        # ===================== #
        # config saving options #
        # ===================== #
        output_zarr_dir = os.path.join(output_dir, group_name)
        if os.path.exists(output_zarr_dir): 
            cprint(f'{output_zarr_dir} has been overwritten', 'green')
            continue
            os.system('rm -rf {}'.format(output_zarr_dir))
            cprint(f'{output_zarr_dir} has been overwritten', 'red')
        
        zarr_root = zarr.group(output_zarr_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')

        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        # feature_map_chunk_size = (100, feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
        # pcd_mask_chunk_size = (100, pcd_masks.shape[1])
        action_chunk_size = (100, action_arrays.shape[1])
        # gripper_pcd_chunk_size = (100, gripper_pcds.shape[1], gripper_pcds.shape[2])
        # goal_gripper_pcd_chunk_size = (100, goal_gripper_pcds.shape[1], goal_gripper_pcds.shape[2])
        
        # ====== #
        # saving #
        # ====== #
        zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('feature_map', data=feature_maps, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('pcd_mask', data=pcd_masks, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('gripper_pcd', data=gripper_pcds, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcds, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

def main(arg):

    # input_dir = arg.input_dir
    # assert os.path.exists(input_dir), f"input_dir {input_dir} does not exist"

    # sub_path = '0702-dp3-goal_pcd_cond'
    # output_dir = os.path.join(os.path.dirname(input_dir), sub_path)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)

    # Load the data

    input_dirs = [
        '/scratch/chialiang/dp3_demo/0703-act3d-mlp-obj-46732-goal'
        '/scratch/chialiang/dp3_demo/0703-act3d-mlp-obj-46801-goal'
        '/scratch/chialiang/dp3_demo/0703-act3d-mlp-obj-46874-goal'
        '/scratch/chialiang/dp3_demo/0703-act3d-mlp-obj-46922-goal'
        '/scratch/chialiang/dp3_demo/0703-act3d-mlp-obj-46966-goal'
        '/scratch/chialiang/dp3_demo/0703-act3d-mlp-obj-47570-goal'
        '/scratch/chialiang/dp3_demo/0703-act3d-mlp-obj-47578-goal'
        '/scratch/chialiang/dp3_demo/0703-act3d-mlp-obj-48700-goal'
    ]

    output_dirs = [
        '/scratch/chialiang/dp3_demo/0703-dp3-obj-46732-goal_gripper_on_agent'
        '/scratch/chialiang/dp3_demo/0703-dp3-obj-46801-goal_gripper_on_agent'
        '/scratch/chialiang/dp3_demo/0703-dp3-obj-46874-goal_gripper_on_agent'
        '/scratch/chialiang/dp3_demo/0703-dp3-obj-46922-goal_gripper_on_agent'
        '/scratch/chialiang/dp3_demo/0703-dp3-obj-46966-goal_gripper_on_agent'
        '/scratch/chialiang/dp3_demo/0703-dp3-obj-47570-goal_gripper_on_agent'
        '/scratch/chialiang/dp3_demo/0703-dp3-obj-47578-goal_gripper_on_agent'
        '/scratch/chialiang/dp3_demo/0703-dp3-obj-48700-goal_gripper_on_agent'
    ]

    for (input_dir, output_dir) in zip(input_dirs, output_dirs):
        post_process_per_step(input_dir, output_dir)

    print('process completed')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data_post_processing')
    # parser.add_argument('--input_dir', '-id', type=str, default='/scratch/yufei/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point')
    # parser.add_argument('--input_dir', '-id', type=str, default='/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0702-act3d-obj-48700-goal')
    # parser.add_argument('--input_dir', '-id', type=str, default='dp3_demo/0701-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_part')
    parser.add_argument('--input_dir', '-id', type=str, default='dp3_demo/0702-dp3-goal_gripper_on_agent')
    args = parser.parse_args()
    main(args)