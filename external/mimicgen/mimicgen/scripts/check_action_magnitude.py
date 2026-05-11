import numpy as np
import open3d as o3d
import os

delta_translation_magnitudes = []
for demo_i in range(50):

    demo_path = f"/data/robogen/mimicgen/datasets/articubot_format/square_d2/demo_{demo_i}"
    all_steps = os.listdir(demo_path)
    for t in range(len(all_steps)):
        data_path = f"{demo_path}/{t}.npz"
        data = np.load(data_path, allow_pickle=True)
        pointcloud = data['point_cloud'][:][0].astype(np.float32)
        gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
        goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)

        action = data['action'][:][0].astype(np.float32)
        state = data['state'][:][0].astype(np.float32)

        delta_translation = action[:3]
        mag = np.max(np.abs(delta_translation))
        delta_translation_magnitudes.append(mag)

import matplotlib.pyplot as plt
plt.plot(delta_translation_magnitudes)
plt.show()