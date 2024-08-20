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
# json_results = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug/0602-act3d-obj-46462-train-ratio-0.2_2024.06.02_19.34.59_train_dp3_robogen_open_door_checkpoints_epoch600.ckpt/opened_joint_angles.json"
# json_results = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_results/0513-vary-obj-loc-ori-init-segmask_2024.05.13_19.01.52_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"

# json_results = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/0622-per-step-load-ddp-obj-45448-horizon-8-train-episodes-260-gripper-goal-with-gripper-displacement-to-closest-obj-point_2024.06.22_01.51.29_train_dp3_robogen_open_door_checkpoints_epoch-200.ckpt/opened_joint_angles.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/06301910-dp3_goal_gripper_whole-horizon-8-num_load_episodes-260_2024.06.30_19.10.41_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"

# act + mlp
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07010750-act3d_goal-horizon-8-num_load_episodes-52_2024.07.01_07.51.00_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011806-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-260_2024.07.01_18.06.53_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"

# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011815-act3d_goal_mlp-horizon-8-num_load_episodes-260_2024.07.01_18.15.27_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011815-act3d_goal_mlp-horizon-8-num_load_episodes-260_2024.07.01_18.15.27_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011815-act3d_goal_mlp-horizon-8-num_load_episodes-260_2024.07.01_18.15.27_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-1.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011815-act3d_goal_mlp-horizon-8-num_load_episodes-260_2024.07.01_18.15.27_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-2.json"

# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011806-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-260_2024.07.01_18.06.53_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011806-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-260_2024.07.01_18.06.53_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011806-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-260_2024.07.01_18.06.53_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-1.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07011806-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-260_2024.07.01_18.06.53_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-2.json"

# act + mlp
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07012321-dp3_goal_gripper_part-horizon-8-num_load_episodes-260_2024.07.01_23.21.58_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07012323-dp3_goal_gripper_whole-horizon-8-num_load_episodes-260_2024.07.01_23.23.26_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07020049-dp3_goal_gripper_on_agent-horizon-8-num_load_episodes-260_2024.07.02_00.49.27_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07020049-dp3_goal_gripper_on_agent-horizon-8-num_load_episodes-260_2024.07.02_00.49.27_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles.json"

# act + mlp abs
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07021653-dp3_goal_gripper_on_agent_abs-horizon-8-num_load_episodes-260_2024.07.02_16.53.16_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07021653-dp3_goal_gripper_on_agent_abs-horizon-8-num_load_episodes-260_2024.07.02_16.53.16_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-1.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07021653-dp3_goal_gripper_on_agent_abs-horizon-8-num_load_episodes-260_2024.07.02_16.53.16_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-2.json"

# act + mlp abs
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07021957-dp3-horizon-8-num_load_episodes-260_2024.07.02_19.57.22_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07021957-dp3-horizon-8-num_load_episodes-260_2024.07.02_19.57.22_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-1.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07021957-dp3-horizon-8-num_load_episodes-260_2024.07.02_19.57.22_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-2.json"

# [Chialiang]   
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-0-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-1-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-2-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-3-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-4-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-5-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-6-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-7-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-8-0.json"
# json_results = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt/opened_joint_angles-9-0.json"

import glob 
from pathlib import Path

object_ids = [
    '41510', '40147',
    '45448', '44817',
    '46462', '44962',
    '46732', '45132',
    '46801', '45219',
    '46874', '45243',
    '46922', #'45297',
    '46966', '45332',
    '47570', '45378',
    '47578', '45384',
    '48700', '45463', # we should use it
    '45526',
    '45661',
    '45694',
    '45780',
    '45910',
    '45961',
    '46408',
    '46417',
    '46440',
    '46490',
    '46762',
    '46825',
    '46893',
    '47235',
    '47281',
    '47315',
    '47529',
    '47669',
    '47944',
    '48063',
    '48177',
    '48356',
    '48623',
    '48876',
    '49025',
    '49062',
    '49132',
    '49133',
    '40417',
    '41085',
    '41452',
    '45162',
    '45176',
    '45194',
    '45203',
    '45248',
    '45271',
    '45290',
    '45305',
]

# # dp3_goal_gripper_on_agent (fixed)
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07080715-dp3_goal_gripper_on_agent-horizon-8-num_load_episodes-1000_2024.07.08_07.16.03_train_dp3_robogen_open_door_checkpoints_latest.ckpt'

# # dp3_goal_gripper_dense
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07070127-dp3_goal_gripper_dense-horizon-8-num_load_episodes-1000_2024.07.07_01.27.53_train_dp3_robogen_open_door_checkpoints_latest.ckpt'

# # act3d_goal_mlp
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.03_19.08.43_train_dp3_robogen_open_door_checkpoints_latest.ckpt'

# # act3d_goal_mlp + closest point
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07102327-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-1000_2024.07.10_23.27.48_train_dp3_robogen_open_door_checkpoints_latest.ckpt'

# ==== #
# 0720 #
# ==== #
# # act3d_goal_mlp + closest point
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07191257-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-1000_2024.07.19_12.57.05_train_dp3_robogen_open_door_checkpoints_latest.ckpt'

# act3d_goal_mlp
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.20_15.26.54_train_dp3_robogen_open_door_checkpoints_latest.ckpt'


# # ==== #
# # 0720 #
# # ==== #
# # act3d_goal_mlp 
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/07221724-act3d_goal_mlp-horizon-8-num_load_episodes-1000_2024.07.22_17.24.11_train_dp3_robogen_open_door_checkpoints_latest.ckpt'


# # ==== #
# # 0810 #
# # ==== #

# delta wpt, sample, aug_rot 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/08062012-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_pcd_2024.08.06_20.12.30_train_dp3_robogen_open_door_checkpoints_latest.ckpt'

# # delta wpt, sample, aug_rot 
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/08081744-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_rot_2024.08.08_17.44.49_train_dp3_robogen_open_door_checkpoints_latest.ckpt'

# # delta wpt, sample, aug_scale 
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/debug-2/08090103-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_scale_2024.08.09_01.03.17_train_dp3_robogen_open_door_checkpoints_latest.ckpt'

# delta wpt, sample, all 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08091652-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_all_2024.08.09_16.53.26_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# ######################################

# delta wpt, epsilon, normalize action 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08101108-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-normalize_action-epsilon_2024.08.10_11.08.09_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# delta wpt, epsilon, aug_rot
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08101115-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_rot-epsilon_2024.08.10_11.15.56_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# # ######################################

# delta wpt, v_prediction, normalize action 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08120322-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-normalize_action-v_prediction_2024.08.12_03.22.34_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# delta wpt, v_prediction, aug_rot
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08120006-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_rot-v_prediction_2024.08.12_00.06.46_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# ######################################

# # all objects, normalize action 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08121539-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-normalize_action_2024.08.12_15.40.03_train_dp3_robogen_open_door_checkpoints_latest.ckpt/0'

# all objects, normalize action 
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08130042-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-normalize_action-30_2024.08.13_00.42.27_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# ######################################

# # # abs wpt, epsilon 
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08110104-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-abs_wpt-epsilon_2024.08.11_01.04.17_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# # abs wpt, v_prediction 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08131154-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-abs_wpt-v_prediction_2024.08.13_11.54.56_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# ######################################

# # abs wpt, v_prediction 
# json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0810/08141110-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-normalize_action-30_2024.08.14_11.10.47_train_dp3_robogen_open_door_checkpoints_latest.ckpt/0'

# ######################################

# week 8.14 - 8.20

# pn++
# # delta wpt, pointnet2, sample, normalize action, 50 objects
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0815/08140150-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-pn2-normalize_action_2024.08.14_01.50.59_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# delta wpt, pointnet2, sample, normalize action, all objects
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0815/08150052-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-pn2-aug_pcd-sample_2024.08.15_00.52.14_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# # delta wpt, pointnet2, epsilon, normalize action, all objects
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0815/08161917-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-pn2-aug_pcd-epsilon-new_2024.08.16_19.17.46_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

# dense pcd
# delta wpt, dense pcd, sample, normalize action 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0815/08142025-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-dense_gripper_2024.08.14_20.25.35_train_dp3_robogen_open_door_checkpoints_latest.ckpt_backup/0'

# combine action
# delta wpt, combine-2-new, sample, normalize action 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0815/08191542-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-combine-pcd_noise-new_2024.08.19_15.42.15_train_dp3_robogen_open_door_checkpoints_latest.ckpt/0'

# dense pcd with combined action
# delta wpt, dense pcd, sample, normalize action 
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0815/08191250-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-dense_gripper-combine-pcd_noise-new_2024.08.19_12.50.15_train_dp3_robogen_open_door_checkpoints_latest.ckpt/2'

######################################
# delta wpt, dense pcd, sample, normalize action, observation step=4
json_results_root = '/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/eval_train_10_obj_test_new_10_act3d_mlp_0815/08191826-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-nouse_2024.08.19_18.26.15_train_dp3_robogen_open_door_checkpoints_latest.ckpt/0'

json_results_list = glob.glob(f'{json_results_root}/*-unseen-*.json')
tag = 'all_unseen'

if len(json_results_list) == 0:
    print(f'no files issue')
    exit(0)

print()
print('--- Start Parsing ---')
print()

nperf_dict = {}
grasped_dict = {}
plot_object_ids = []
plot_nperf = []

debug = 0

for json_results in json_results_list: 

    parent_path = Path(json_results).parent.absolute()

    with open(json_results) as f:
        data = json.load(f)
        
    opened_joint_angles_dict = {}
    opened_joint_angles = []
    expert_angles = []
    initial_angles = []
    grasped_handle = []

    for i, key in enumerate(data):
        
        object_id = None
        for oid in object_ids:
            if oid in key:
                object_id = oid 
                break
        assert object_id is not None

        if object_id not in opened_joint_angles_dict.keys():
            opened_joint_angles_dict[object_id] = [{
                'opened_joint_angles': data[key]["final_door_joint_angle"],
                'expert_angles': data[key]["expert_door_joint_angle"] if "46462" not in key else 0.27,
                'initial_angles': data[key]['initial_joint_angle'],
                'grasped_handle': data[key]['grasped_handle'],
            }]
        else :
            opened_joint_angles_dict[object_id].append( {
                'opened_joint_angles': data[key]["final_door_joint_angle"],
                'expert_angles': data[key]["expert_door_joint_angle"] if "46462" not in key else 0.27,
                'initial_angles': data[key]['initial_joint_angle'],
                'grasped_handle': data[key]['grasped_handle'],
            })

        opened_joint_angles.append(data[key]["final_door_joint_angle"])
        expert_angles.append(data[key]["expert_door_joint_angle"] if "46462" not in key else 0.27)
        initial_angles.append(data[key]['initial_joint_angle'])
        grasped_handle.append(data[key]['grasped_handle'])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()
    # expert_angles = np.array(expert_angles) - np.array(initial_angles)
    
    plot_object_ids = []
    plot_nperf = []
    cnt = 0
    
    for object_id in opened_joint_angles_dict.keys():

        obj_opened_joint_angles = []
        obj_initial_angles = []
        obj_expert_angles = []
        obj_grasped_handle = []

        for element in opened_joint_angles_dict[object_id]:
            obj_opened_joint_angles.append(element['opened_joint_angles'])    
            obj_expert_angles.append(element['expert_angles'])
            obj_initial_angles.append(element['initial_angles'])
            obj_grasped_handle.append(element['grasped_handle'])

        cnt += len(obj_opened_joint_angles)
        normalized_performance = (np.array(obj_opened_joint_angles) - np.array(obj_initial_angles)) / (np.array(obj_expert_angles) - np.array(obj_initial_angles))
        normalized_performance[normalized_performance > 1.0] = 1.0
        # normalized_performance[np.isnan(normalized_performance)] = 0
        # normalized_performance[np.isinf(normalized_performance)] = 0
        cond = ~(np.isnan(normalized_performance) | np.isinf(normalized_performance))

        # Debug
        debug += np.mean(normalized_performance[cond]) * len(normalized_performance[cond])
        normalized_performance = np.mean(normalized_performance[cond])

        mean_grasped_handle = np.mean(obj_grasped_handle)

        if object_id not in nperf_dict.keys():
            nperf_dict[object_id] = []
        nperf_dict[object_id].append(np.round(normalized_performance, 3))

        if object_id not in grasped_dict.keys():
            grasped_dict[object_id] = []
        grasped_dict[object_id].append(np.round(mean_grasped_handle, 3))
        
        # plot_object_ids.append(object_id)
        # plot_nperf.append(np.round(normalized_performance, 3))

    assert len(opened_joint_angles) == cnt

    normalized_performance = (np.array(opened_joint_angles) - np.array(initial_angles)) / (np.array(expert_angles) - np.array(initial_angles))
    normalized_performance[normalized_performance > 1] = 1
    cond = ~(np.isnan(normalized_performance) | np.isinf(normalized_performance))

    # Debug
    debug = debug / len(normalized_performance[cond])
    normalized_performance = np.mean(normalized_performance[cond])

    mean_grasped_handle = np.mean(grasped_handle)
    
    # write results

    if 'overall' not in nperf_dict.keys():
        nperf_dict['overall'] = []
    nperf_dict['overall'].append(np.round(normalized_performance, 3))

    if 'overall' not in grasped_dict.keys():
        grasped_dict['overall'] = []
    grasped_dict['overall'].append(np.round(mean_grasped_handle, 3))

    # plot_object_ids.append('overall')
    # plot_nperf.append(np.round(normalized_performance, 3))
    
    ################################################################################################################

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()

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
    # plt.show()

    parent_path = Path(json_results).parent.absolute()
    final_path = f'{parent_path}/result.jpg'
    plt.savefig('{}'.format(final_path))
    print(f'{final_path} has been written')

    plt.clf()

# dump 

for k, v in zip(nperf_dict.keys(), nperf_dict.values()):
    # assert len(v) == len(json_results_list), 'It seems that the validation process is still running or the data is not complete'
    if k == 'overall':
        print(f'{k}, {np.round(np.mean(v),3)}, {np.round(np.std(v),3)}')
    else:
        print(f'{k}, {np.round(np.mean(v),3)}')

exit(0)
# Figure Size
fig, ax = plt.subplots(figsize =(18, 9))
# ax.hlines(y=np.arange(-1, 1.2, 0.2), xmin=-10000, xmax=1000, colors=[0.7, 0.7, 0.7])

# Horizontal Bar Plot
X_axis = np.arange(len(nperf_dict.values())) 

grasped_colors = [[83,191,254] for _ in range(len(grasped_dict.values()) - 1)]
grasped_colors += [[17, 0, 255]]
grasped_colors = np.asarray(grasped_colors) / 255
grasped_mean = [np.mean(value) for value in grasped_dict.values()]
grasped_std = [np.std(value) for value in grasped_dict.values()]
bar1 = ax.bar(X_axis - 0.2, grasped_mean, width=0.4, yerr=grasped_std, color=grasped_colors, zorder=3, label = 'grasping success rate')

nperf_colors = [[83,255,126] for _ in range(len(nperf_dict.values()) - 1)]
nperf_colors += [[29,254,0]]
nperf_colors = np.asarray(nperf_colors) / 255
nperf_mean = [np.mean(value) for value in nperf_dict.values()]
nperf_std = [np.std(value) for value in nperf_dict.values()]
bar2 = ax.bar(X_axis + 0.2, nperf_mean, width=0.4, yerr=nperf_std, color=nperf_colors, zorder=3, label = 'normalized performance')

plt.legend()
plt.xticks(X_axis, nperf_dict.keys(), fontsize=14)
plt.title('Normalized Performance for each object', fontsize=20)
plt.xlabel('Object ID', fontsize=18)
# plt.ylabel('Normalized Performance', fontsize=18)
# plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 1)

strategy = [
    # 'left',
    # 'left',
    # 'back',
    # 'right',
    # 'right',
    # 'right',
    # 'down',
    # 'left',
    # 'right',
    # 'back',
    # '',


    'right',
    'back',
    'back',
    'back',
    'right',
    'back',
    'right',
    'right',
    'right',
    'right',
    '',
]

# Add counts above the two bar graphs
for rect, strat in zip(bar1+bar2, strategy+strategy):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{strat}\n{height:.02f}', ha='center', va='bottom', fontsize=14)

plt.tight_layout()

final_path = f'{parent_path}/{tag}_mean_nperf_with_std.jpg'
plt.savefig('{}'.format(final_path))
print(f'{final_path} has been written')

plt.clf()