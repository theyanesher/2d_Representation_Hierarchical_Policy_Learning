import argparse, os, json, sys, glob
import zarr
import numpy as np
import open3d as o3d
from termcolor import cprint

from tqdm import tqdm

DATA_KEYS = [   
                'action', 
                'feature_map', 
                'gripper_pcd', 
                'goal_gripper_pcd',
                'pcd_mask', 
                'point_cloud', 
                'state'
            ]

# STEP_INFO_PATH = "../../project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
STEP_INFO_PATH = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
STEP_INFO_PATHS = os.listdir(STEP_INFO_PATH)
STAGE_LENGTH_DICT = {
    path: f'{STEP_INFO_PATH}/{path}/grasp_the_door_handle_primitive/stage_lengths.json' for path in STEP_INFO_PATHS
}

def post_process(input_dir, output_dir, prefix=''):

    # group_names = os.listdir(input_dir)
    group_paths = glob.glob(f'{input_dir}/2024*')

    for group_path in group_paths:

        cprint(f'processing {group_path}', 'green')

        group_name = group_path.split('/')[-1]

        group = zarr.open(group_path, mode='r')
        # one trajectory
        data_group = group['data']
        data = {}

        for key in DATA_KEYS:
            if key in data_group:
                data[key] = np.array(data_group[key])
            else :
                raise ValueError(f"key {key} not found in {group_name}")
            
        # append goal gripper conditioning on point clouds

        original_states = data['state']
        original_pcds = data['point_cloud']
        original_actions = data['action']
        original_feature_maps = data['feature_map']
        original_pcd_masks = data['pcd_mask']
        original_gripper_pcds = data['gripper_pcd']
        original_goal_gripper_pcds = data['goal_gripper_pcd']

        step_info_json = STAGE_LENGTH_DICT[group_name]
        step_info = json.load(open(step_info_json, 'r')) 

        # traj_len = len(original_pcds)
        # grasp_achieved_step = (step_info['reach_handle']  + (step_info['reach_to_contact'] - 2)  + step_info['close_gripper']) // 2
        # goal_1 = original_gripper_pcds[grasp_achieved_step-1]
        # goal_2 = original_gripper_pcds[-1]

        # For DP3 + MLP only
        new_states = []
        for i, (original_state, original_gripper_pcd, original_goal_gripper_pcd) in enumerate(zip(original_states, original_gripper_pcds, original_goal_gripper_pcds)):
            diff = original_goal_gripper_pcd - original_gripper_pcd
            diff_flat = diff.reshape(-1)

            # new_state = np.concatenate((original_state, diff_flat), axis=0)
            new_state = np.concatenate((original_state, original_goal_gripper_pcd.reshape(-1)), axis=0)

            new_states.append(new_state)
        new_states = np.stack(new_states, axis=0)

        # # For DP3 + MLP only
        # new_pcds = []
        # for wpt_id in range(traj_len):

        #     goal = goal_1 if wpt_id < grasp_achieved_step else goal_2

        #     new_pcd = [original_pcds[wpt_id]]
        #     for goal_i in goal:
        #         new_pcd_i = goal_i - original_pcds[wpt_id]
        #         new_pcd.append(new_pcd_i)
            
        #     new_pcd = np.concatenate(new_pcd, axis=-1)
        #     new_pcds.append(new_pcd)
        # point_cloud_arrays = np.stack(new_pcds, axis=0)

        # # For DP3 + MLP only

        # new_feature_maps = []
        # goal_gripper_pcds = []
        # for wpt_id in range(traj_len):

        #     goal = goal_1 if wpt_id < grasp_achieved_step else goal_2
        #     N, H, W, C = original_feature_maps[wpt_id].shape
        #     flattened_feat_map = original_feature_maps[wpt_id].reshape(N * H * W, C)
        #     flattened_pcd_mask = original_pcd_masks[wpt_id].astype(bool)

        #     additional_feat = np.zeros((N * H * W, 3 * len(goal)))

        #     for i, goal_i in enumerate(goal):
        #         additional_feat[flattened_pcd_mask, i*3: i*3+3] = goal_i - flattened_feat_map[flattened_pcd_mask][:,2:]

        #     flattened_full_feat = np.concatenate([flattened_feat_map, additional_feat], axis=-1)
        #     full_feat = flattened_full_feat.reshape(N, H, W, -1)
        #     new_feature_maps.append(full_feat)
        #     goal_gripper_pcds.append(goal)
            
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

    input_dir = arg.input_dir
    assert os.path.exists(input_dir), f"input_dir {input_dir} does not exist"

    sub_path = '0702-dp3-goal_pcd_cond_abs'
    output_dir = os.path.join(os.path.dirname(input_dir), sub_path)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)

    # Load the data
    post_process(input_dir, output_dir)

    print('process completed')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data_post_processing')
    parser.add_argument('--input_dir', '-id', type=str, default='/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0630-dp3-goal-whole')
    # parser.add_argument('--input_dir', '-id', type=str, default='data/dp3_demo/0624-act3d-obj-45448-remove-reaching-collision-resize-2-full-goal_pcd_cond')
    args = parser.parse_args()
    main(args)