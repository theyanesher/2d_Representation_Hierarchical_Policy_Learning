from matplotlib import pyplot as plt
import json
import os
import numpy as np

# json_data_no_smoothing = "3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/vary_robot_init_joint_near_handle_2024.04.19_14.26.08_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles_pregrasped_False.json"
# json_data_no_smooting_after_reaching = "3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/vary_robot_init_joint_near_handle_after_reaching_2024.04.20_15.48.13_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles_pregrasped_True.json"
# json_data_after_reaching_filter_action = "3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/vary_robot_init_joint_near_handle_after_reaching_filter_small_action_2024.04.21_03.47.03_train_dp3_robogen_open_door_checkpoints_epoch=1500-test_mean_score=0.528.ckpt/opened_joint_angles_pregrasped_True.json"
# josn_data_after_reaching_filter_action_long_horizon = "3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/vary_robot_init_joint_near_handle_after_reaching_filter_small_action_long_horizon_2024.04.21_04.13.39_train_dp3_robogen_open_door_checkpoints_epoch=0600-test_mean_score=0.474.ckpt/opened_joint_angles_pregrasped_True.json"
# json_data_full_process_filter_small_action_after_motion_planning = "3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/vary_robot_init_joint_near_handle_filter_small_action_only_after_reaching_2024.04.21_18.06.07_train_dp3_robogen_open_door_checkpoints_epoch-1800-test_mean_score=0.051.ckpt/opened_joint_angles_pregrasped_False.json"
# json_data_full_process_filter_small_action_whole_traj = "3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/vary_robot_init_joint_near_handle_filter_small_action_2024.04.21_18.09.10_train_dp3_robogen_open_door_checkpoints_epoch=1800-test_mean_score=0.154.ckpt/opened_joint_angles_pregrasped_False.json"


# with open(json_data_after_reaching_filter_action) as f:
#     data_after_reaching_filter_action = json.load(f)
# with open(json_data_full_process_filter_small_action_after_motion_planning) as f:
#     data_full_process_filter_small_action_after_motion_planning = json.load(f)
# with open(josn_data_after_reaching_filter_action_long_horizon) as f:
#     josn_data_after_reaching_filter_action_long_horizon = json.load(f)
# with open(json_data_full_process_filter_small_action_whole_traj) as f:
#     data_full_process_filter_small_action_whole_traj = json.load(f)
# with open(json_data_no_smoothing) as f:
#     data_no_smoothing = json.load(f)
# with open(json_data_no_smooting_after_reaching) as f:
#     data_no_smooting_after_reaching = json.load(f)
    
# datas = [
#     data_after_reaching_filter_action, 
#     data_no_smooting_after_reaching,
#     # josn_data_after_reaching_filter_action_long_horizon, 
#     data_full_process_filter_small_action_after_motion_planning,
#     # data_full_process_filter_small_action_whole_traj,
#     data_no_smoothing,
# ]
# labels = [
#     "After reaching", 
#     "After reaching no filtering", 
#     # "After reaching long horizon",  
#     "Full filter after reaching", 
#     # "Full filter whole traj",
#     "Full no filtering",
# ]
# all_configs = list(data_after_reaching_filter_action.keys())
# colors = ['tab:blue', "tab:green", 'tab:orange', 'tab:red', 'tab:purple', "tab:brown"]
# fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# axes = axes.flatten()
# for idx, (data, label) in enumerate(zip(datas, labels)):
#     opened_joint_angles = []
#     expert_angles = []
#     for key in data:
#         if key in all_configs:
#             if label == "Full no filtering":
#                 opened_joint_angles.append(data[key])
#             else:
#                 opened_joint_angles.append(data[key][0])
#             if idx == 0:
#                 expert_angles.append(data[key][1])
        
#     opened_joint_angles = np.array(opened_joint_angles)
#     # bar plot the average joint angles
#     x = (idx + 1) * 0.5 
#     axes[0].bar(x, opened_joint_angles.mean(axis=0), width=0.25, yerr=opened_joint_angles.std(axis=0), label=label, color=colors[idx])
#     axes[0].text(x, opened_joint_angles.mean(axis=0) + 0.05, str(round(opened_joint_angles.mean(axis=0), 2)))
#     axes[0].set_ylabel("opened joint angle")
#     axes[1].plot(opened_joint_angles, label=label, color=colors[idx])
#     axes[1].set_xlabel("configuration idx")
#     axes[1].set_ylabel("opened joint angle")
#     if idx == 0:
#         axes[0].bar(0,  np.mean(expert_angles), width=0.25, yerr=np.std(expert_angles), label="Expert", color='black')
#         axes[0].text(0, np.mean(expert_angles) + 0.05, str(round(np.mean(expert_angles), 2)))
#         axes[1].plot(expert_angles, label="Expert", color='black')
    
# axes[0].legend()
# plt.show()

# json_results = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/0502-vary-obj-init-angle-robot-init-joint-near-handle-larger_2024.05.03_01.50.11_train_dp3_robogen_open_door_checkpoints_epoch-2700-test_mean_score=0.767.ckpt/opened_joint_angles_pregrasped_False.json"
# json_results = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first_2024.05.11_14.45.04_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
json_results = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/0514-vary-obj-loc-ori-only-handle-points_2024.05.14_02.50.18_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
# json_results = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/0513-vary-obj-loc-ori-init-segmask_2024.05.13_19.01.52_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
with open(json_results) as f:
    data = json.load(f)
    
opened_joint_angles = []
expert_angles = []
initial_angles = []
for key in data:
    opened_joint_angles.append(data[key][0])
    expert_angles.append(data[key][1])
    initial_angles.append(data[key][2])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes = axes.flatten()
expert_angles = np.array(expert_angles) - np.array(initial_angles)

ax = axes[0]
ax.bar(0, np.mean(expert_angles), width=0.25, yerr=np.std(expert_angles), label="Expert", color='black')
ax.text(0, np.mean(expert_angles) + 0.05, str(round(np.mean(expert_angles), 2)))
ax.bar(0.5, np.mean(opened_joint_angles), width=0.25, yerr=np.std(opened_joint_angles), label="Our method", color='tab:blue')
ax.text(0.5, np.mean(opened_joint_angles) + 0.05, str(round(np.mean(opened_joint_angles), 2)))
ax.set_ylabel("opened joint angle")
ax.legend()

ax = axes[1]
ax.plot(expert_angles, "-*", label="Expert", color='black', markersize=5,  )
ax.plot(opened_joint_angles, "-o", label="Our method", color='tab:blue',  markersize=5)
ax.legend()
ax.set_ylabel("opened joint angle")
ax.set_xlabel("configuration idx")
plt.show()