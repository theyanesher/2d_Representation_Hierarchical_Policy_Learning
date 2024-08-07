import zarr
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from diffuser_actor_3d.robogen_utils import *
import pybullet as p
from scipy.spatial.transform import Rotation as R

def rotation_transfer_6D_to_matrix(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix

def get_pc_nindices_within_gripper(cur_eef_pos, cur_eef_orient, pc_points):
    
    cur_pos, cur_orient = cur_eef_pos, cur_eef_orient

    X_GW = p.invertTransform(cur_pos, cur_orient)
    translation = np.array(X_GW[0])
    rotation = np.array(p.getMatrixFromQuaternion(X_GW[1])).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation ### this is the transformation from world frame to gripper frame

    pc_homogeneous = np.hstack((pc_points, np.ones((pc_points.shape[0], 1))))  # Convert to homogeneous coordinates Nx4
    pc_transformed_homogeneous = T @ pc_homogeneous.T # 4x4 @ 4xN = 4xN
    p_GC = pc_transformed_homogeneous[:3, :] # 3xN

    ### Crop to a region inside of the finger box.
    crop_min = [-0.03, -0.04, -0.06] 
    crop_max = [0.03, 0.04, 0.01]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )
    
    return indices

def extract_data_from_dataset(zarr_path, index=None):
    if os.path.exists(zarr_path) is False:
        import pdb; pdb.set_trace()
    group = zarr.open(zarr_path, 'r')
    src_store = group.store
    src_root = zarr.group(src_store)
    pointcloud = src_root['data']['point_cloud'][:] # (N, 3)
    if index is not None:
        pointcloud = pointcloud[index]
    else:
        pointcloud = pointcloud[0]

    gripper_pcd = src_root['data']['gripper_pcd'][:] # (N, 3)
    if index is not None:
        gripper_pcd = gripper_pcd[index]
    else:
        gripper_pcd = gripper_pcd[0]

    gripper_state = src_root['data']['state'][:] 
    gripper_pos = gripper_state[0, :3]
    gripper_orn = gripper_state[0, 3:9]
    gripper_orn = rotation_transfer_6D_to_matrix(gripper_orn)
    gripper_orn = R.from_matrix(gripper_orn).as_quat()

    # gripper_pos, gripper_orn = get_gripper_pos_orient_from_4_points(gripper_pcd)
    label_indices = get_pc_nindices_within_gripper(gripper_pos, gripper_orn, pointcloud)
    binary_mask = np.zeros(pointcloud.shape[0])
    binary_mask[label_indices] = 1


    return pointcloud, gripper_pcd, binary_mask


diverse_data_dirs = ["0705-obj-41510", "0705-obj-45448", "0705-obj-46462", "0705-obj-46732", "0705-obj-46801", "0705-obj-46874", "0705-obj-46922", "0705-obj-46966", "0705-obj-47570", "0705-obj-47578", "0705-obj-48700", "0705-obj-45526", "0705-obj-45661", "0705-obj-45694", "0705-obj-45780", "0705-obj-45910", "0705-obj-45961", "0705-obj-46408", "0705-obj-46417", "0705-obj-46440", "0705-obj-46490", "0705-obj-46762", "0705-obj-46825", "0705-obj-46893", "0705-obj-47235", "0705-obj-47281", "0705-obj-47315", "0705-obj-47529", "0705-obj-47669", "0705-obj-47944", "0705-obj-48063", "0705-obj-48177", "0705-obj-48356", "0705-obj-48623", "0705-obj-48876", "0705-obj-49025", "0705-obj-49062", "0705-obj-49132", "0705-obj-49133", "0712-obj-40417", "0712-obj-41085", "0712-obj-41452", "0712-obj-45162", "0712-obj-45176", "0712-obj-45194", "0712-obj-45203", "0712-obj-45248", "0712-obj-45271", "0712-obj-45290", "0712-obj-45305"]
exp_data_dirs = ["/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-00/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-05/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-49/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-45/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                "/project_data/held/ziyuw2/Robogen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first",
                ]

test_data_dirs = ["0705-obj-40147", "0705-obj-44817", "0705-obj-44962", "0705-obj-45132", "0705-obj-45219", "0705-obj-45243", "0705-obj-45332", "0705-obj-45378", "0705-obj-45384", "0705-obj-45463"]


def get_all_zarr_paths():
    for i, data_dir in enumerate(diverse_data_dirs):
        if i > 10 and i <= 38:
            exp_data_dir = "/project_data/held/ziyuw2/Robogen-sim2real/data/diverse_objects/" + "open_the_door_" + data_dir.split("-")[-1] + "/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
            exp_data_dirs.append(exp_data_dir)
        if i > 38:
            exp_data_dir = "/project_data/held/ziyuw2/Robogen-sim2real/data/diverse_objects_2/" + "open_the_door_" + data_dir.split("-")[-1] + "/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
            exp_data_dirs.append(exp_data_dir)


    all_zarr_paths = []
    for i, data_dir in enumerate(diverse_data_dirs):
        # if i > 0:
        #     break
        temp_dir = "/scratch/chialiang/dp3_demo/" + data_dir
        all_traj_data_folders = os.listdir(temp_dir)
        for traj_data_folder in all_traj_data_folders:
            # if traj_data_folder does not contain "2024", skip
            if "2024" not in traj_data_folder:
                continue
            traj_exp_folder = exp_data_dirs[i] + "/" + traj_data_folder
            # find a folder under that directory
            for root, dirs, files in os.walk(traj_exp_folder):
                # find the file named stage_lengths.json
                for file in files:
                    if file == "stage_lengths.json":
                        json_file = os.path.join(root, file)
                        with open(json_file) as f:
                            stage_lengths = json.load(f)
                        break

            grasped_index = stage_lengths["reach_handle"] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
            grasped_index = grasped_index // 2 - 1

            zarr_path = os.path.join(temp_dir, traj_data_folder, str(grasped_index))
            all_zarr_paths.append(zarr_path)

    return all_zarr_paths

def get_all_test_zarr_paths():
    test_exp_data_dir = []
    for i, data_dir in enumerate(test_data_dirs):
        exp_data_dir = "/project_data/held/ziyuw2/Robogen-sim2real/data/diverse_objects/" + "open_the_door_" + data_dir.split("-")[-1] + "/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
        test_exp_data_dir.append(exp_data_dir)

    all_zarr_paths = []
    for i, data_dir in enumerate(test_data_dirs):
        temp_dir = "/project_data/held/ziyuw2/Robogen-sim2real/data/dp3_demo/" + data_dir
        all_traj_data_folders = os.listdir(temp_dir)
        for j, traj_data_folder in enumerate(all_traj_data_folders):
            if j >= 25:
                break   
            # if traj_data_folder does not contain "2024", skip
            if "2024" not in traj_data_folder:
                continue
            traj_exp_folder = test_exp_data_dir[i] + "/" + traj_data_folder
            # find a folder under that directory
            for root, dirs, files in os.walk(traj_exp_folder):
                # find the file named stage_lengths.json
                for file in files:
                    if file == "stage_lengths.json":
                        json_file = os.path.join(root, file)
                        with open(json_file) as f:
                            stage_lengths = json.load(f)
                        break

            grasped_index = stage_lengths["reach_handle"] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
            grasped_index = grasped_index // 2 - 1

            zarr_path = os.path.join(temp_dir, traj_data_folder, str(grasped_index))
            all_zarr_paths.append(zarr_path)

    return all_zarr_paths



def build_up_data():
    # all_zarr_paths = get_all_zarr_paths()
    all_zarr_paths = get_all_test_zarr_paths()
    all_point_cloud = []
    all_gripper_pcd = []
    all_binary_masks = []

    for zarr_path in all_zarr_paths:
        pointcloud, gripper_pcd, binary_mask = extract_data_from_dataset(zarr_path)
        all_point_cloud.append(pointcloud)
        all_gripper_pcd.append(gripper_pcd)
        all_binary_masks.append(binary_mask)

    all_point_cloud = np.array(all_point_cloud)
    all_gripper_pcd = np.array(all_gripper_pcd)
    all_binary_masks = np.array(all_binary_masks)

    # Save the data to zarr
    save_path = "/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/data_eval_25"

    if os.path.exists(save_path):
        print("Remove the existing data folder")
        os.system("rm -rf " + save_path)


    zarr_root = zarr.group(save_path)
    zarr_data = zarr_root.create_group('data')
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    point_cloud_chunk_size = (100, 4500, 3)
    gripper_pcd_chunk_size = (100, 4, 3)
    binary_mask_chunk_size = (100, 4500)
    zarr_data.create_dataset('point_cloud', data=all_point_cloud, chunks=point_cloud_chunk_size, compressor=compressor)
    zarr_data.create_dataset('gripper_pcd', data=all_gripper_pcd, chunks=gripper_pcd_chunk_size, compressor=compressor)
    zarr_data.create_dataset('binary_mask', data=all_binary_masks, chunks=binary_mask_chunk_size, compressor=compressor)


if __name__ == "__main__":
    build_up_data()
   
    
    
            

        

    
