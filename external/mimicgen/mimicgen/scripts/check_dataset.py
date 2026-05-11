import numpy as np
import open3d as o3d

data_path = "/data/robogen/smith_mimicgen/datasets/articubot_format/mug_cleanup_debug/demo_0/0.npz"
data = np.load(data_path, allow_pickle=True)
pointcloud = data['point_cloud'][:][0].astype(np.float32)
gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)

action = data['action'][:][0].astype(np.float32)
state = data['state'][:][0].astype(np.float32)
print(action)
print(state)
print(gripper_pcd)
print(goal_gripper_pcd)

obj_pcd_np = data['point_cloud'].reshape(-1, 3)
gripper_pcd_np = data['gripper_pcd'].reshape(-1, 3)
goal_gripper_pcd_np = data['goal_gripper_pcd'].reshape(-1, 3)

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