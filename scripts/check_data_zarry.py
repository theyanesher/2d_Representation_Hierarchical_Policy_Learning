import os
import pickle as pkl
import numpy as np
import zarr
import open3d as o3d

### bad ones that we should not use
# /data/yufeiw2/RoboGen_sim2real/data/dp3_demo_rerender/1105-rerender-obj-45135

data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/data/debug/2024-08-30-04-02-48/0"
group = zarr.open(data_path, 'r')
src_store = group.store

# numpy backend
src_root = zarr.group(src_store)
data = src_root['data']
    
obj_pcd_np = data['point_cloud'][:].reshape(-1, 3)
gripper_pcd_np = data['gripper_pcd'][:].reshape(-1, 3)
goal_gripper_pcd_np = data['goal_gripper_pcd'][:].reshape(-1, 3)

### plot them together with different colors
obj_pcd = o3d.geometry.PointCloud()
obj_pcd.points = o3d.utility.Vector3dVector(obj_pcd_np)

gripper_pcd = o3d.geometry.PointCloud()
gripper_pcd.points = o3d.utility.Vector3dVector(gripper_pcd_np)

goal_gripper_pcd = o3d.geometry.PointCloud()
goal_gripper_pcd.points = o3d.utility.Vector3dVector(goal_gripper_pcd_np)

### set to different colors
obj_pcd.paint_uniform_color([0.0, 0.0, 1.0])
gripper_pcd.paint_uniform_color([1.0, 0.0, 0.0])
goal_gripper_pcd.paint_uniform_color([0.0, 1.0, 0.0])

o3d.visualization.draw_geometries([obj_pcd, gripper_pcd, goal_gripper_pcd])