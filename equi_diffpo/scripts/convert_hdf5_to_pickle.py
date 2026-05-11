import h5py
import numpy as np
import argparse
import os
import pickle
# import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument("--ep", type=int)
    parser.add_argument("--task", type=str, default='square_d2')
    args = parser.parse_args()
    ep = args.ep
    task = args.task
    file_root = Path('/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/robomimic/datasets/')
    save_root = Path('/scratch/minon/')

    file = h5py.File(file_root / task / f'{task}_pcd_abs.hdf5', 'r')
    print("Keys: %s" % file['data']['demo_0'].keys())
    target_dir = save_root / f'{task}_abs/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir,exist_ok=True)
    os.chdir(target_dir)
    folder_name = 'episode_' + str(ep)
    aa26d_transformer = RotationTransformer('axis_angle', 'rotation_6d')
    q26d_transformer = RotationTransformer('quaternion', 'rotation_6d')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    episode = file['data']['demo_' + str(ep)]
    length = episode['obs']['point_cloud'].shape[0]

    action_pos = episode['actions'][:,:3]
    action_rot = episode['actions'][:,3:6]
    action_grippers = episode['actions'][:,[-1]]
    action_6d_rot = aa26d_transformer.forward(action_rot)
    actions = np.concatenate([action_pos, action_6d_rot, action_grippers], axis=-1)

    for i in tqdm(range(0,length), leave=False):
        episode_pointcloud = episode['obs']['point_cloud']
        point_cloud = episode_pointcloud[i]

        eef_pos = episode['obs']['robot0_eef_pos'][i]
        eef_quat = episode['obs']['robot0_eef_quat'][i]
        eef_qpos = episode['obs']['robot0_gripper_qpos'][i]
        gripper_pcd = episode['obs']['gripper_pcd'][i]
        goal_gripper_pcd = episode['obs']['goal_gripper_pcd'][i]

        # convert to 10d agent position
        eef_6d = q26d_transformer.forward(eef_quat)
        agent_state = np.concatenate([eef_pos, eef_6d, eef_qpos[[0]]])
        
        action = np.expand_dims(actions[i], axis=0)
        point_cloud = np.expand_dims(point_cloud, axis=0)
        gripper_pcd = np.expand_dims(gripper_pcd, axis=0)
        goal_gripper_pcd = np.expand_dims(goal_gripper_pcd, axis=0)
        agent_state = np.expand_dims(agent_state, axis=0)
        data = {'point_cloud': point_cloud, 
                'action': action, 'gripper_pcd': gripper_pcd, 
                'goal_gripper_pcd': goal_gripper_pcd,
                'state': agent_state,
                }
        with open(folder_name + '/' + str(i) + '.pkl', 'wb') as f:
            pickle.dump(data, f)
