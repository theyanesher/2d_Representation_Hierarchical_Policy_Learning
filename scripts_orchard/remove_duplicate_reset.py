import os
import pickle as pkl
import numpy as np
from tqdm import tqdm


def remove_duplicate_reset_traj_folder(traj_path):
    new_path = traj_path.replace("reset", "reset_only")
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        
    all_pickle_files = os.listdir(traj_path)
    all_pickle_files = [f for f in all_pickle_files if f.endswith('.pkl')]
    num_pickle_files = len(all_pickle_files)
    
    with open(os.path.join(traj_path, f'{num_pickle_files - 1}.pkl'), 'rb') as f:
        data = pkl.load(f)
        last_goal_gripper = data['goal_gripper_pcd']
    
    for t in range(num_pickle_files - 2, -1, -1):
        with open(os.path.join(traj_path, f'{t}.pkl'), 'rb') as f:
            data = pkl.load(f)
            
        diff = np.abs(data['goal_gripper_pcd'] - last_goal_gripper).sum()
        if diff > 1e-3:
            last_t_time = t
            break
        
    beg_t = last_t_time + 1
    print(f'Copying from {beg_t} to {num_pickle_files} into {new_path}')
    for t in range(beg_t, num_pickle_files):
        os.system(f'cp {os.path.join(traj_path, f"{t}.pkl")} {os.path.join(new_path, f"{t - beg_t}.pkl")}')
        
def remove_duplicate_reset_whole_folder(root_path):
    all_obj_folders = os.listdir(root_path)
    all_obj_folders = sorted(all_obj_folders)
    
    for obj_folder in all_obj_folders:
        all_traj_folders = os.listdir(os.path.join(root_path, obj_folder))
        all_traj_folders = sorted(all_traj_folders)
        for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
            if s in all_traj_folders:
                all_traj_folders.remove(s)
                
        for traj_folder in tqdm(all_traj_folders):
            traj_path = os.path.join(root_path, obj_folder, traj_folder)
            print(f'Processing {traj_path}')
            remove_duplicate_reset_traj_folder(traj_path)            
            
if __name__ == '__main__':
    root_path = '/tmp/165-obj_reset_1203/'
    root_path = '/tmp/165-obj_reset_1203/'
    remove_duplicate_reset_whole_folder(root_path)
        
        
