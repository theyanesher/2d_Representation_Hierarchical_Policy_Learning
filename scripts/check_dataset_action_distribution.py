import numpy as np
import pickle as pkl
import os

data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/dp3_demo/bucket_100435/2025-05-11-15-06-27"
all_pickle_files = [x for x in os.listdir(data_path) if x.endswith('.pkl')]
traj_len = len(all_pickle_files)

actions = []
for t in range(traj_len):
    with open(os.path.join(data_path, all_pickle_files[t]), 'rb') as f:
        data = pkl.load(f)
    
    action = data['action']  # Assuming actions are stored under the key 'actions'
    action = action.reshape(1, 10)
    actions.append(action)
    
actions = np.concatenate(actions, axis=0)
delta_trans = actions[:, :3]
sim_magnitude = np.linalg.norm(delta_trans, axis=1)

data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha/plate/traj_0001"
data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha_combined_2_step_0/plate/traj_0000"
all_np_files = [x for x in os.listdir(data_path) if x.endswith('.npz')]
all_np_files = sorted(all_np_files, key=lambda x: int(x.split('.')[0]))
traj_len = len(all_np_files)

actions = []
for t in range(traj_len):
    data = np.load(os.path.join(data_path, all_np_files[t]))
    
    action = data['action']  # Assuming actions are stored under the key 'actions'
    action = action.reshape(1, 10)
    actions.append(action)

actions = np.concatenate(actions, axis=0)
delta_trans = actions[:, :3]
real_magnitude = np.linalg.norm(delta_trans, axis=1)


import matplotlib.pyplot as plt
# plt.hist(sim_magnitude, bins=50, color='blue', alpha=0.5, label='Simulated')
# plt.hist(real_magnitude, bins=50, color='orange', alpha=0.5, label='Real-world')
plt.plot(range(len(real_magnitude)), real_magnitude, color='orange', label='Real-world')
plt.xlabel('Action Translation Magnitude')
plt.ylabel('Frequency')
plt.title('Action Translation Magnitude Distribution')
plt.legend()
plt.show()

