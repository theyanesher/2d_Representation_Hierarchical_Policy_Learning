import zarr
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from manipulation.utils import rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
from scipy.spatial.transform import Rotation as R

# zarr_path = "data/dp3_demo/0527-act3d/"
zarr_path = "data/dp3_demo/0531-act3d-obj-46462"
# zarr_path = "data/dp3_demo/0527-act3d-always-close/"
# zarr_path = "data/dp3_demo/0528-act3d-close-during-open-filter-small-action-after-grasp/"
demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
all_subfolder = os.listdir(zarr_path)
for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
    if string in all_subfolder:
        all_subfolder.remove(string)
all_subfolder = sorted(all_subfolder)
zarr_paths = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]
path_list = zarr_paths[:10]

per_episode_root = []
keys = ['action', 'state', 'point_cloud']
for zarr_path in path_list:
    exp_name = zarr_path.split('/')[-1]
    # stage_lengths_json_file = os.path.join(demo_path, exp_name, "grasp_the_door_handle_primitive", 'stage_lengths.json')
    # with open(stage_lengths_json_file, 'r') as f:
    #     stage_lengths = json.load(f)
    # open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
    
    group = zarr.open(zarr_path, 'r')
    src_store = group.store

    # numpy backend
    src_root = zarr.group(src_store)
    meta = dict()
    
    data = dict()
    for key in keys:
        arr = src_root['data'][key]
        data[key] = arr[:]
        
    pcd = data['point_cloud'][0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2])
    plt.show()
        
    orientations = data['state'][:, 3:9]
    quaternion_diffs = []
    for i in range(len(orientations) - 1):
        cur_ori_6d = orientations[i]
        target_ori_6d = orientations[i+1]
        cur_ori_matrix = rotation_transfer_6D_to_matrix(cur_ori_6d)
        target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)
        
        cur_ori_quat =  R.from_matrix(cur_ori_matrix).as_quat()
        target_ori_quat = R.from_matrix(target_ori_matrix).as_quat()
        quat_diff = np.arccos(2 * np.dot(cur_ori_quat, target_ori_quat)**2 - 1)
        one_step_quaternion_diff = np.arccos(2 * np.dot(cur_ori_quat, target_ori_quat)**2 - 1)
        quaternion_diffs.append(quat_diff)

    finger_movement = data['action'][:, -1]
    delta_translation = np.linalg.norm(data['action'][:, :3], axis=1)
    plt.plot(finger_movement)
    # plt.axvline(open_time_idx, color='r')
    plt.show()
    plt.plot(delta_translation, "-*")
    # plt.axvline(open_time_idx, color='r')
    plt.show()
    plt.plot(quaternion_diffs, "-*")
    # plt.axvline(open_time_idx, color='r')
    plt.show()