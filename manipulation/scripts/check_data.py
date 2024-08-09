import zarr
from termcolor import cprint
import time
import pickle

def load_data(zarr_path, keys):
    group = zarr.open(zarr_path, 'r')
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

keys = ['state', 'action', 'point_cloud']
keys += ['feature_map', 'gripper_pcd', 'pcd_mask', "goal_gripper_pcd"]
zarr_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/data/debug/2024-07-30-21-58-36/100"
data = load_data(zarr_path, keys)

from matplotlib import pyplot as plt
pcd = data['point_cloud'][0]
goal_gripper_pcd = data['goal_gripper_pcd'][0]
gripper_pcd = data['gripper_pcd'][0]
ax = plt.axes(projection='3d')
ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], c='r', s=1)
ax.scatter(gripper_pcd[:,0], gripper_pcd[:,1], gripper_pcd[:,2], c='b', s=1)
ax.scatter(goal_gripper_pcd[:,0], goal_gripper_pcd[:,1], goal_gripper_pcd[:,2], c='g', s=1)
plt.show()

