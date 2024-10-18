import zarr
import os
import numpy as np
import json
from tqdm import tqdm
from termcolor import cprint
import scipy
import pickle

def save_data(pc_list, state_list, gripper_pcd_list, action_list, 
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              feature_map_list,
              pcd_mask_list,
              save_dir):

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)
    gripper_pcd_arrays = np.array(gripper_pcd_list)
    feature_map_arrays = np.array(feature_map_list)
    pcd_mask_list = np.array(pcd_mask_list)
    
    chunk_size = 1
    state_chunk_size = (chunk_size, state_arrays.shape[1])
    point_cloud_chunk_size = (chunk_size, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (chunk_size, action_arrays.shape[1])
    gripper_pcd_chunk_size = (chunk_size, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    goal_gripper_pcd_chunk_size = (chunk_size, goal_gripper_pcd.shape[1], goal_gripper_pcd.shape[2])
    displacement_gripper_to_object_chunk_size = (chunk_size, displacement_gripper_to_object.shape[1], displacement_gripper_to_object.shape[2])
    feature_map_chunk_size = (chunk_size, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3], feature_map_arrays.shape[4]) # there can be mutiple cameras
    pcd_mask_chunk_size = (chunk_size, pcd_mask_list.shape[1])
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_object, chunks=displacement_gripper_to_object_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('feature_map', data=feature_map_arrays, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('pcd_mask', data=pcd_mask_list, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        
    del state_arrays, point_cloud_arrays, gripper_pcd_arrays, action_arrays
    del pc_list, state_list, gripper_pcd_list, action_list
    del zarr_root, zarr_data, zarr_meta
    del goal_gripper_pcd
    del displacement_gripper_to_object, feature_map_arrays, pcd_mask_list
    

def resave(zarr_path):
    all_subfolder = os.listdir(zarr_path)
    for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
        if string in all_subfolder:
            all_subfolder.remove(string)
            
    all_subfolder = sorted(all_subfolder)

    keys = ['state', 'action', 'point_cloud']
    keys += ['gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']
    combine_action_steps = 2

    new_data_save_dir = zarr_path.replace("/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/", 
            "/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/")
            # "/jet/projects/cis240052p/ywang59/dp3_demo/")
    already_moved_data_num = 0
    if os.path.exists(new_data_save_dir):
        already_moved_data_num = len(os.listdir(new_data_save_dir))
    already_moved_data_num = max(1, already_moved_data_num)
    
    # print("already_moved_data_num", already_moved_data_num)
    for traj in tqdm(all_subfolder, desc='processing'):
        zarr_path_traj = os.path.join(zarr_path, traj)
        all_steps = os.listdir(zarr_path_traj)  
        all_steps = sorted(all_steps, key=lambda x: int(x))

        new_data_save_dir_traj = zarr_path_traj.replace("/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/", 
                                                        "/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/")
                                                        # "/jet/projects/cis240052p/ywang59/dp3_demo/")
            

        # for every step after opening, change the goal gripper pcd
        try:
            for step in all_steps:                    
                zarr_path_step = os.path.join(zarr_path_traj, step)
                new_data_save_dir = zarr_path_step.replace("/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/", 
                                                        "/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/")
                                                        # "/jet/projects/cis240052p/ywang59/dp3_demo/")
            
                
                # save new data
                meta_dir = new_data_save_dir.split("/")[:-1]
                meta_dir = "/".join(meta_dir)
                if not os.path.exists(meta_dir):
                    os.makedirs(meta_dir)
                    
                new_pickle_data_save_path = new_data_save_dir + ".pkl"
                if not os.path.exists(new_pickle_data_save_path):
                    cprint(f"Saving new data to: {new_pickle_data_save_path}", "green")
                    zarr_path_step = os.path.join(zarr_path_traj, step)
                    group = zarr.open(zarr_path_step, 'r')
                    src_store = group.store

                    # numpy backend
                    src_root = zarr.group(src_store)
                    data = dict()
                    for key in keys:
                        arr = src_root['data'][key]
                        data[key] = arr[:]
                    with open(new_pickle_data_save_path, "wb") as f:
                        pickle.dump(data, f)
                    
                else:
                    cprint(f"Already exists: {new_pickle_data_save_path}", "yellow")
                    continue
        except:
            print("Error in", zarr_path_traj)
            os.system(f"rm -rf {new_data_save_dir_traj}")
                
            # save_data(
            #         data['point_cloud'], 
            #         data['state'], 
            #         data['gripper_pcd'], 
            #         data['action'], 
            #         data['goal_gripper_pcd'], 
            #         data['displacement_gripper_to_object'],
            #         data['feature_map'],
            #         data['pcd_mask'],
            #         new_data_save_dir
            # )
            
            # import pdb; pdb.set_trace()

output_dirs = [

        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-41510',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-45448',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46462',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46732',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46801',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46874',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46922',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46966',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-47570',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-47578',
        # '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-48700',
        
        # g1
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-45526',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-45661', # 2024-07-06-16-58-56
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-45694',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-45780',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-45910',

        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-45961',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46408',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46417',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46440',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46490',

        # g2
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46762',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46825',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-46893',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-47235',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-47281',

        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-47315',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-47529',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-47669', # 2024-07-06-23-48-11
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-47944',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-48063',
        
        # g3
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-48177', # 2024-07-07-02-20-38
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-48356',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-48623',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-48876',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-49025',
        
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-49062',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-49132',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0705-obj-49133',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-40417',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-41085',

        # # g4
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-41452',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-45162',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-45176',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-45194',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-45203',

        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-45248',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-45271', # 2024-07-12-02-53-33
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-45290',
        '/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo/0712-obj-45305',
]


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default=None)
args = parser.parse_args()
resave(args.output_dir)
# for output_dir in output_dirs:
#     resave(output_dir)