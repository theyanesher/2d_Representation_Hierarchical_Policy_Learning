import h5py
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path("data/robomimic/datasets/")
# ROOT = Path("data/robomimic/alltracker/")
task = 'square_d2'

# dataset = ROOT / task / f'{task}_flow_abs.hdf5'
dataset = ROOT / task / f'{task}_pcd_abs_images.hdf5'

robot_ids = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]

episode_idx = 0
with h5py.File(dataset, 'r') as f:
    f = f['data']
    t = 0
    # segmentations = f[f'demo_{episode_idx}']['obs']['agentview_robot_mask'][40]
    # plt.imshow(segmentations); plt.show()
    # rgb = f[f'demo_{episode_idx}']['obs']['agentview_image'][t]
    # plt.imshow(rgb); plt.show()
    # rmask = f[f'demo_{episode_idx}']['obs']['agentview_robot_mask'][t]
    # plt.imshow(rmask); plt.show()
    
    # plt.imshow(rgb * segmentations[:,:,None]); plt.show()
    # plt.imshow(rgb * ~segmentations[:,:,None]); plt.show()
    # visconf_mask = f[f'demo_{episode_idx}']['obs']['agentview_image_visconf_mask'][t]
    # plt.imshow(visconf_mask); plt.show()

    # visconf_mask = f[f'demo_{episode_idx}']['obs']['agentview_image_visconf_maps_e'][t]
    # plt.imshow(visconf_mask[...,0]); plt.show()
    # breakpoint()


    for key in f[f'demo_{episode_idx}']['obs'].keys():
        print(key, f[f'demo_{episode_idx}']['obs'][key][0].shape, f[f'demo_{episode_idx}']['obs'][key].dtype)
        # if 'segmentation' in key:
        #     print(f[f'demo_{episode_idx}']['obs'][key][:].max())
        # scene_mask = np.isin(segmentations, robot_ids)
        # plt.imshow(~scene_mask); plt.show()