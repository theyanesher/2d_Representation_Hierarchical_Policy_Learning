import zarr
from termcolor import cprint
import time
import pickle
from manipulation.utils import rotation_transfer_6D_to_matrix
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from matplotlib import pyplot as plt
import pickle as pkl
from tqdm import tqdm

# def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle):
#     original_gripper_pcd = np.array([[ 0.43856215, -0.40922496,  0.6756892 ],
#        [ 0.41712594, -0.42972428,  0.6489343 ],
#        [ 0.4379155 , -0.43029538,  0.6417602 ],
#        [ 0.41987222, -0.44440767,  0.6243291 ]])
#     original_gripper_orn = np.array([ 0.69285525, -0.64422789,  0.08350163,  0.31296886])
#     original_joint_angle = 0.001 # 0.03999999999999995

#     # joint angle 0.03999999999999995
#     # gripper_pcd_right_finger_open = np.array([ 0.3991713 , -0.42923108,  0.65513015])
#     # gripper_pcd_left_finger_open = np.array([ 0.45587012, -0.43078858,  0.6355644 ])
    
#     # original_gripper_pcd[1] = original_gripper_pcd[1] + (gripper_pcd_right_finger_open - original_gripper_pcd[1]) * (cur_joint_angle - 0.001) / (0.03999999999999995 - 0.001)
#     # original_gripper_pcd[2] = original_gripper_pcd[2] + (gripper_pcd_left_finger_open - original_gripper_pcd[2]) * (cur_joint_angle - 0.001) / (0.03999999999999995 - 0.001)

#     goal_R = R.from_matrix(gripper_orn)
#     original_R = R.from_quat(original_gripper_orn)
#     rotation_transfer = goal_R * original_R.inv()
#     original_pcd = original_gripper_pcd - original_gripper_pcd[3]
#     rotated_pcd = rotation_transfer.apply(original_pcd)
#     gripper_pcd = rotated_pcd + gripper_pos
#     return gripper_pcd

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle):
    # original_gripper_pcd = np.array([[ 0.10432111,  0.00228697,  0.8474241 ],
    #         [ 0.12816067, -0.04368229,  0.8114649 ],
    #         [ 0.08953098,  0.0484529 ,  0.80711854],
    #         [ 0.11198021,  0.00245327,  0.7828771 ]])
    # original_gripper_orn = np.array([0.97841681, 0.19802945, 0.0581003 , 0.01045192])
    # original_gripper_pcd = np.array([[ 0.43856215, -0.40922496,  0.6756892 ],
    #    [ 0.3991713 , -0.42923108,  0.65513015 ],
    #    [ 0.45587012, -0.43078858,  0.6355644  ],
    #    [ 0.41987222, -0.44440767,  0.6243291 ]])
    # original_gripper_orn = np.array([ 0.69285525, -0.64422789,  0.08350163,  0.31296886])
    original_gripper_pcd = np.array([[ 0.5648266,   0.05482348,  0.34434554],
        [ 0.5642125,   0.02702148,  0.2877661 ],
        [ 0.53906703,  0.01263776,  0.38347825],
        [ 0.54250515, -0.00441092,  0.32957944]]
    )
    original_gripper_orn = np.array([0.21120763,  0.75430543, -0.61925177, -0.05423936])
    
    gripper_pcd_right_finger_closed = np.array([ 0.55415434,  0.02126799,  0.32605097])
    gripper_pcd_left_finger_closed = np.array([ 0.54912525,  0.01839125,  0.3451934 ])
    gripper_pcd_closed_finger_angle = 2.6652539383870777e-05
 
    original_gripper_pcd[1] = gripper_pcd_right_finger_closed + (original_gripper_pcd[1] - gripper_pcd_right_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
    original_gripper_pcd[2] = gripper_pcd_left_finger_closed + (original_gripper_pcd[2] - gripper_pcd_left_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
 
    # goal_R = R.from_quat(gripper_orn)
    # import pdb; pdb.set_trace()
    goal_R = R.from_matrix(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd

def load_data(pickle_path, keys):
    group = zarr.open(pickle_path, 'r')
    src_store = group.store

    # numpy backend
    src_root = zarr.group(src_store)
    meta = dict()

    for key, value in src_root['meta'].items():
        if len(value.shape) == 0:
            meta[key] = np.array(value)
        else:
            meta[key] = value[:]

    if keys is None:
        keys = src_root['data'].keys()
    data = dict()
    for key in keys:
        arr = src_root['data'][key]
        data[key] = arr[:]
        
    return data


def find_first_stage(traj_path, num_steps):
    first_goal = None
    for i in range(num_steps):
        substep_path = os.path.join(traj_path, str(i) + ".pkl")

        data = pickle.load(open(substep_path, 'rb'))
        action = data['action'][:]

        current_goal = data['goal_gripper_pcd'][:]
        if first_goal is None:
            first_goal = current_goal
        elif not np.allclose(first_goal, current_goal):
            return i    
        
def restore_second_stage(traj_path, total_steps, second_stage_start_idx, new_goal_idx):
    new_goal_pickle_path = os.path.join(traj_path, f"{new_goal_idx}.pkl")
    data = pickle.load(open(new_goal_pickle_path, 'rb'))
    new_goal_gripper_pcd = data['gripper_pcd'][:]
    
    for idx in range(second_stage_start_idx, new_goal_idx + 1):
        pickle_path = os.path.join(traj_path, f"{idx}.pkl")
        data = pickle.load(open(pickle_path, 'rb'))
        data['goal_gripper_pcd'] = new_goal_gripper_pcd
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
            
    for idx in range(new_goal_idx + 1, total_steps):
        path = os.path.join(traj_path, f"{idx}.pkl")
        cmd = f"rm {path}"
        print(cmd)
        # import pdb; pdb.set_trace()
        # os.system(cmd)

keys = ['state', 'action', 'point_cloud']
keys += ['feature_map', 'gripper_pcd', 'pcd_mask', "goal_gripper_pcd"]

data_path = "/scratch/yufeiw2/dp3_demo_clean_distorted_goal"
all_obj_dirs = os.listdir(data_path)
all_obj_dirs = sorted(all_obj_dirs)

all_obj_dirs = [x for x in all_obj_dirs if "1121-other" in x]

for obj_folder in tqdm(all_obj_dirs):
    all_traj_dirs = os.listdir(os.path.join(data_path, obj_folder))
    if len(all_traj_dirs) == 0:
        cmd = f"rm -rf {os.path.join(data_path, obj_folder)}"
        os.system(cmd)
    
    all_traj_dirs = sorted(all_traj_dirs)
    
    for traj_path in all_traj_dirs:
        traj_path = os.path.join(data_path, obj_folder, traj_path)

        all_pickle_files = os.listdir(traj_path)
        num_steps = len(all_pickle_files)

        first_stage_num = find_first_stage(traj_path, num_steps)
        second_stage_num = num_steps - first_stage_num
        second_stage_start_idx = first_stage_num
        # print(f"{traj_path} first stage {first_stage_num} second stage {second_stage_num}")

        # for idx in range(116, 117):
        for idx in range(num_steps):
            pickle_path_step = os.path.join(traj_path, str(idx) + ".pkl")
            with open(pickle_path_step, "rb") as f:
                data = pkl.load(f)

            pcd = data['point_cloud'].reshape(-1, 3)
            gripper_pcd = data['gripper_pcd'].reshape(-1, 3)
            gripper_pos = data['state'][0, :3]
            gripper_orient = data['state'][0, 3:9]
            gripper_joint_angle = data['state'][0, 9]
            # print("grpper_joint_angle: ", gripper_joint_angle)
            gripper_orient_matrix = rotation_transfer_6D_to_matrix(gripper_orient)
            analytical_gripper_pcd = get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orient_matrix, gripper_joint_angle)
                
            distance = np.linalg.norm(gripper_pcd - analytical_gripper_pcd, axis=-1).mean()
            # print(f"{idx} distance between cur_gripper and analytic gripper: {distance}")


            if distance > 0.02 and idx > num_steps - 10:
                # print(f"{traj_path} gripper distorted at step {idx} with distortion {distance}")
                restore_second_stage(traj_path, num_steps, second_stage_start_idx, idx-1)

