import os
import pickle as pkl
import numpy as np
import zarr
import open3d as o3d
import torch
import matplotlib.pyplot as plt

# data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/2024-08-25-01-46-57/70.pkl"
data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/2024-11-03-09-49-31/80.pkl"
# data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/2024-11-03-09-49-31/40.pkl"
# data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/2024-07-30-00-31-18/0.pkl"
# data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/2024-08-02-19-00-50/0.pkl"
# data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/2024-08-21-15-36-49/0.pkl"
with open(data_path, "rb") as f:
    data = pkl.load(f)

obj_pcd_np = data['point_cloud'].reshape(-1, 3)
gripper_pcd_np = data['gripper_pcd'].reshape(-1, 3)
goal_gripper_pcd_np = data['goal_gripper_pcd'].reshape(-1, 3)

### use matplotlib to show the low-level input
ax = plt.axes(projection='3d')
ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c='grey', s=1)
ax.scatter(gripper_pcd_np[:,0], gripper_pcd_np[:,1], gripper_pcd_np[:,2], c='blue', s=30)
ax.scatter(goal_gripper_pcd_np[:,0], goal_gripper_pcd_np[:,1], goal_gripper_pcd_np[:,2], c='red', s=30)
ax.axis('equal')
ax.axis('off')
plt.show()


### show input of high-level policy
ax = plt.axes(projection='3d')
ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c='grey', s=1)
ax.scatter(gripper_pcd_np[:,0], gripper_pcd_np[:,1], gripper_pcd_np[:,2], c='grey', s=30)
# ax.scatter(goal_gripper_pcd_np[:,0], goal_gripper_pcd_np[:,1], goal_gripper_pcd_np[:,2], c='blue', s=30)
ax.axis('equal')
ax.axis('off')
plt.show()


### load model
load_model_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/model_8.pth"
from test_PointNet2.model_invariant import PointNet2_super
pointnet2_model = PointNet2_super(num_classes=13).to("cuda")
pointnet2_model.load_state_dict(torch.load(load_model_path))
pointnet2_model.eval()
goal_policy = pointnet2_model

### get model prediction
pointcloud = torch.from_numpy(obj_pcd_np).unsqueeze(0).to("cuda")
gripper_pcd = torch.from_numpy(gripper_pcd_np).unsqueeze(0).to("cuda")
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
# ax = plt.axes(projection='3d')
# ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c=weights_numpy, s=12,  cmap='seismic')
# ax.axis('equal')
# ax.axis('off')
# plt.show()

### plot the displacement
# downsample_ratio = 50
# obj_pcd = obj_pcd_np # [::downsample_ratio]
# start = obj_pcd_np[::downsample_ratio]
# end = start + outputs_numpy[::downsample_ratio, :3]
# weights = weights_numpy[::downsample_ratio]

# ax = plt.axes(projection='3d')
# ax.scatter(obj_pcd[:,0], obj_pcd[:,1], obj_pcd[:,2], c='grey', s=1)


# for idx, (start, end) in enumerate(zip(start, end)):
#     X = [start[0], end[0]]
#     Y = [start[1], end[1]]
#     Z = [start[2], end[2]]
#     ax.plot(X, Y, Z, c='r', linewidth=0.5)
# ax.axis('equal')
# ax.axis('off')
# plt.show()

### plot the final prediction
# ax = plt.axes(projection='3d')
# ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c='grey', s=1)
# ax.scatter(gripper_pcd_np[:,0], gripper_pcd_np[:,1], gripper_pcd_np[:,2], c='grey', s=30)
# ax.scatter(prediction_numpy[:,0], prediction_numpy[:,1], prediction_numpy[:,2], c='red', s=30)
# ax.axis('equal')
# ax.axis('off')
# plt.show()


