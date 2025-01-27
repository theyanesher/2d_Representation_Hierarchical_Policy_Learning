import open3d as o3d
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
import torch
import os

### load data
### Chialiang's data
data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-green_cabinat/2025-01-13-20-32-49-formal4"
step = 3
step = 8

### mobile base data
# data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/2025-01-19-21-38-14-good-20250122T024000Z-001/2025-01-19-21-38-14-good"
# step = 3
# step = 13

### our lab
data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/2025-01-09-21-54-55-trial-1-20250122T071434Z-001/2025-01-09-21-54-55-trial-1"
step = 1
step = 8


pkl_path = os.path.join(data_path, f"step_{step}.pkl")
with open(pkl_path, "rb") as f:
    data = pkl.load(f)

high_level_input_dict = data["high_level_input_dict"]

high_level_pcd = high_level_input_dict["high_level_point_cloud"]    
high_level_gripper_pcd = high_level_input_dict["high_level_gripper_pcd"]
# high_level_output = high_level_input_dict["high_level_outputs"]

obj_pcd_np = high_level_pcd.squeeze(0)[-1].cpu().detach().numpy()
if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-green_cabinat/2025-01-13-20-32-49-formal4" and step == 8:
    obj_pcd_np = obj_pcd_np[obj_pcd_np[:, 0] > 0.59]
if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/2025-01-19-21-38-14-good-20250122T024000Z-001/2025-01-19-21-38-14-good":
    obj_pcd_np = obj_pcd_np[obj_pcd_np[:, 0] < 1]
gripper_pcd_np = high_level_gripper_pcd.squeeze(0)[-1].cpu().detach().numpy()

### use open3d to visualize the pcd
# obj_pcd = o3d.geometry.PointCloud()
# obj_pcd.points = o3d.utility.Vector3dVector(obj_pcd_np)
# gripper_pcd = o3d.geometry.PointCloud()
# gripper_pcd.points = o3d.utility.Vector3dVector(gripper_pcd_np)
# ## paint object points based on normal
# obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# normals = np.asarray(obj_pcd.normals)
# normal_mag = np.linalg.norm(normals, axis=1)
# # do softmax to normalize the normal magnitude
# normal_mag = np.exp(normal_mag) / np.exp(normal_mag).sum()
# colors = np.stack([normal_mag, normal_mag, normal_mag], axis=1)
# obj_pcd.colors = o3d.utility.Vector3dVector(np.stack([normal_mag, normal_mag, normal_mag], axis=1))
# o3d.visualization.draw_geometries([obj_pcd, gripper_pcd])

### plot the input
# x_value = obj_pcd_np[:, 0]
# ## normalize the x_value
# x_value = (x_value - x_value.min()) / (x_value.max() - x_value.min())
# colors = plt.cm.viridis(x_value)
# down_sample_idx = np.arange(0, len(obj_pcd_np), 1)
# ax = plt.axes(projection='3d')
# # ax.scatter(obj_pcd_np[down_sample_idx,0], obj_pcd_np[down_sample_idx,1], obj_pcd_np[down_sample_idx,2], c=colors, s=4)
# ax.scatter(obj_pcd_np[down_sample_idx,0], obj_pcd_np[down_sample_idx,1], obj_pcd_np[down_sample_idx,2], c="grey", s=4)
# ax.scatter(gripper_pcd_np[:,0], gripper_pcd_np[:,1], gripper_pcd_np[:,2], c='green', s=16)
# ax.axis('equal')
# ax.axis('off')
# plt.show()
# plt.close("all")

### load model
load_model_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/model_8.pth"
load_model_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/model_36.pth"
from test_PointNet2.model_invariant import PointNet2_super
pointnet2_model = PointNet2_super(num_classes=13).to("cuda")
pointnet2_model.load_state_dict(torch.load(load_model_path))
pointnet2_model.eval()
goal_policy = pointnet2_model

### get model prediction
pointcloud = high_level_pcd[0, -1].to("cuda").unsqueeze(0)
if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-green_cabinat/2025-01-13-20-32-49-formal4" and step == 8:
    pointcloud = pointcloud[:, pointcloud[0, :, 0] > 0.59, :]
if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/2025-01-19-21-38-14-good-20250122T024000Z-001/2025-01-19-21-38-14-good":
    pointcloud = pointcloud[:, pointcloud[0, :, 0] < 1, :]
gripper_pcd = high_level_gripper_pcd[0, -1].to("cuda").unsqueeze(0)
inputs = torch.cat([pointcloud, gripper_pcd], dim=1) # B, N+4, 3, B=1

inputs_ = inputs.permute(0, 2, 1)
outputs = goal_policy(inputs_) # B, N, 13
weights = outputs[:, :, -1] # B, N
outputs = outputs[:, :-4, :-1] # B, N, 12
weights = weights[:, :-4]
weights = torch.nn.functional.softmax(weights, dim=1)

B, N, _ = outputs.shape
outputs = outputs.view(B, N, 4, 3)
prediction = outputs + inputs[:, :-4, :3].unsqueeze(2)
prediction = prediction * weights.unsqueeze(-1).unsqueeze(-1)
prediction = prediction.sum(dim=1)
prediction = prediction.unsqueeze(1)

weights_numpy = weights.squeeze(0).cpu().detach().numpy()
print("weights_numpy", weights_numpy.shape)
outputs_numpy = outputs.squeeze(0).cpu().detach().numpy()
print("outputs_numpy", outputs_numpy.shape)
prediction_numpy = prediction.squeeze(0).squeeze(0).cpu().detach().numpy()
print("prediction_numpy", prediction_numpy.shape)




### plot the weight
ax = plt.axes(projection='3d')
# x_color = colors
# weight_color = plt.cm.seismic(weights_numpy)
# final_color = x_color * 0.5 + weight_color * 0.5
ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c=weights_numpy, s=10,  cmap='seismic')
# ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c=final_color, s=10)
ax.axis('equal')
ax.axis('off')
plt.show()
plt.close("all")

### plot the final preidction
ax = plt.axes(projection='3d')
# ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c=colors, s=1, zorder=1, depthshade=True)
ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c="grey", s=1, zorder=1, depthshade=False)
ax.scatter(gripper_pcd_np[:,0], gripper_pcd_np[:,1], gripper_pcd_np[:,2], c='green', s=16)
ax.scatter(prediction_numpy[:, 0], prediction_numpy[:, 1], prediction_numpy[:, 2], c='red', s=32, zorder=10, depthshade=False)
ax.axis('equal')
ax.axis('off')
plt.show()

