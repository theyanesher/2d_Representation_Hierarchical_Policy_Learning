import pickle5 as pickle
import torch
import pytorch3d.ops as torch3d_ops
import fpsample
import time
import numpy as np
import tqdm
from matplotlib import pyplot as plt

data_file = "data/dp3_demo/test_different_init_joint_angle_world/raw_data.pkl"
with open(data_file, "rb") as f:
    data = pickle.load(f)
    
num_points = 3000
pc_list, state_list, action_list, last_state_indices = data
torch_times = []
fps_times = []

random_indices = np.random.choice(len(pc_list), 100)
pc_list = [pc_list[i] for i in random_indices]

for pc in tqdm.tqdm(pc_list):
    point_cloud = np.array(pc).reshape(-1, 3)
    
    beg = time.time()
    torch_point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=torch_point_cloud[...,:3], K=num_points)
    torch_point_cloud = torch_point_cloud.squeeze(0).cpu().numpy()
    torch_point_cloud = torch_point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    torch_times.append(time.time() - beg)
    
    beg = time.time()
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(point_cloud, num_points, h=9)
    fps_point_cloud = point_cloud[kdline_fps_samples_idx]
    fps_times.append(time.time() - beg)
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(torch_point_cloud[:,0], torch_point_cloud[:,1], torch_point_cloud[:,2])
    ax.view_init(azim=-90, elev=10)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(fps_point_cloud[:,0], fps_point_cloud[:,1], fps_point_cloud[:,2])
    ax.view_init(azim=-90, elev=10)
    
    # plt.show()
    
print("torch time: ", np.mean(torch_times))
print("fps time: ", np.mean(fps_times))

    
    
    
    