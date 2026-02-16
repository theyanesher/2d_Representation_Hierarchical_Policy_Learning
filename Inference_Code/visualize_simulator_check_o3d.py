# import pickle
# import numpy as np
# import open3d as o3d

# # --- Load data ---
# with open("pointcloud_and_pose_SIMULATOR_AFTER_REVERSE_TRANSFORM21.pkl", "rb") as f:
#     data = pickle.load(f)

# # Point cloud
# pcd = np.asarray(data["pointcloud"])

# # Gripper points (predicted): expected shape (8,3)
# gripper_points = np.asarray(data["gripper"]).reshape(-1, 3)

# # Present gripper points (current / actual gripper position)
# present_gripper = np.asarray(data["present_gripper"]).reshape(-1, 3)  # make sure your pickle has this key
# # import pdb; pdb.set_trace();
# # --- Open3D geometries ---
# pcd_o3d = o3d.geometry.PointCloud()
# pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
# pcd_colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (pcd.shape[0], 1))
# pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_colors)

# def make_sphere(center, radius=0.01, color=(1.0, 0.0, 0.0), resolution=20):
#     mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
#     mesh.translate(center)
#     mesh.compute_vertex_normals()
#     mesh.paint_uniform_color(color)
#     return mesh

# # Set marker radius relative to point cloud scale
# bounds = np.asarray(pcd_o3d.get_max_bound() - pcd_o3d.get_min_bound())
# diag = np.linalg.norm(bounds)
# marker_radius = 0.01 #max(0.002, 0.01 * (diag / 0.5))  # heuristic

# # Create spheres
# gripper_spheres = [make_sphere(pt, radius=marker_radius, color=(1.0, 0.0, 0.0)) for pt in gripper_points]
# present_spheres = [make_sphere(pt, radius=marker_radius, color=(0.0, 1.0, 0.0)) for pt in present_gripper]

# # Coordinate frame
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05 * (diag / 0.5))

# # --- Visualizer ---
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="PCD + Gripper Comparison", width=1600, height=1200, visible=True)
# vis.add_geometry(pcd_o3d)
# for s in gripper_spheres:
#     vis.add_geometry(s)
# for s in present_spheres:
#     vis.add_geometry(s)
# vis.add_geometry(frame)

# opt = vis.get_render_option()
# opt.background_color = np.asarray([1.0, 1.0, 1.0])
# opt.point_size = 2.0

# vis.poll_events()
# vis.update_renderer()
# vis.capture_screen_image("o3d_gripper_pred_vs_present.png", do_render=True)

# print("Screenshot saved as o3d_gripper_pred_vs_present.png. Close the window to finish.")
# vis.run()
# vis.destroy_window()



import pickle
import numpy as np
import open3d as o3d

# --- Load data ---
i = 18
with open(f"/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/Wrist_Camera_ALL_DATA_POINTS_PREDICTIONS/pointcloud_and_pose_SIMULATOR_AFTER_REVERSE_TRANSFORM{i}.pkl", "rb") as f:
    data = pickle.load(f)

with open("/home/pratik_final/Downloads/Bimanual/Original_RVT2_Inference/RVT/data/insert_onto_square_peg/all_variations/episodes/episode0/low_dim_obs.pkl", "rb") as k:
    data_goal = pickle.load(k)
goal_grippers = []
for d in data_goal:
    goal_grippers.append(d.gripper_pose[:3])

# import pdb; pdb.set_trace();
# Point cloud
pcd = np.asarray(data["pointcloud"])

# Gripper points (predicted): expected shape (8,3)
gripper_points = np.asarray(data["gripper"]).reshape(-1, 3)

# Present gripper points (current / actual gripper position)
present_gripper = np.asarray(data["present_gripper"]).reshape(-1, 3)
# import pdb; pdb.set_trace();
# --- Open3D geometries ---
pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
pcd_colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (pcd.shape[0], 1))
pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_colors)

def make_sphere(center, radius=0.01, color=(1.0, 0.0, 0.0), resolution=20):
    print("GRIPPER POINT", center, "COLOUR", color)
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    mesh.translate(center)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

# Set marker radius relative to point cloud scale
bounds = np.asarray(pcd_o3d.get_max_bound() - pcd_o3d.get_min_bound())
diag = np.linalg.norm(bounds)
marker_radius = 0.01

# --- FOUR UNIQUE COLORS FOR THE FOUR GRIPPERS ---
gripper_colors = [
    (1.0, 0.0, 0.0),   # red
    (1.0, 0.0, 0.0),   # red
    (0.0, 1.0, 0.0),   # green
    (0.0, 1.0, 0.0),   # green
    (0.0, 0.0, 1.0),   # blue
    (0.0, 0.0, 1.0),   # blue
    (1.0, 0.5, 0.0),   # orange
    (1.0, 0.5, 0.0),   # orange
]

# Create spheres (predicted gripper)
gripper_spheres = [
    make_sphere(pt, radius=marker_radius, color=gripper_colors[i])
    for i, pt in enumerate(gripper_points)
]
# import pdb; pdb.set_trace();
# Create spheres (present/actual gripper)
goal_spheres = [make_sphere(pt, radius=marker_radius, color=(0.5, 0.5, 0.5)) for pt in goal_grippers]
present_spheres = [make_sphere(pt, radius=marker_radius, color=(0.0, 0.0, 0.0)) for pt in present_gripper]

# Coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05 * (diag / 0.5))

# --- Visualizer ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="PCD + Gripper Comparison", width=1600, height=1200, visible=True)
vis.add_geometry(pcd_o3d)
for s in gripper_spheres:
    vis.add_geometry(s)
for s in present_spheres:
    vis.add_geometry(s)
for g in goal_spheres:
    vis.add_geometry(g)
vis.add_geometry(frame)

opt = vis.get_render_option()
opt.background_color = np.asarray([1.0, 1.0, 1.0])
opt.point_size = 2.0

vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("o3d_gripper_pred_vs_present.png", do_render=True)

print("Screenshot saved as o3d_gripper_pred_vs_present.png. Close the window to finish.")
vis.run()
vis.destroy_window()

