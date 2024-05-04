# from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

# zarr_path = "data/dp3_demo/vary_robot_init_joint_near_handle_after_reaching_remove_zero_action"
# replay_buffer = ReplayBuffer.copy_from_path(
#             zarr_path, keys=['state', 'action', 'point_cloud'])

# episode_ends = replay_buffer.episode_ends[:]
# episodes_lengths = [episode_ends[i] - (episode_ends[i-1] if i > 0 else 0) for i in range(len(episode_ends))]
# print("number of episodes: ", len(episodes_lengths))
# print(episodes_lengths[2:])

# print("================")

# zarr_path = "data/dp3_demo/vary_robot_init_joint_near_handle_perturbation_for_open_filter_small_action_after_reaching"
# replay_buffer = ReplayBuffer.copy_from_path(
#             zarr_path, keys=['state', 'action', 'point_cloud'])

# episode_ends = replay_buffer.episode_ends[:]
# episodes_lengths = [episode_ends[i] - (episode_ends[i-1] if i > 0 else 0) for i in range(len(episode_ends))]
# print("number of episodes: ", len(episodes_lengths))
# print(episodes_lengths[:59])

# import os
# import json
# all_experiment_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0502-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle/"
# step_name = "grasp_the_door_handle_primitive"
# all_experiments = os.listdir(all_experiment_path)
# all_experiments = sorted(all_experiments)
# for exp_path in all_experiments:
#     # extracted_pkl_path = os.path.join(all_experiment_path, exp_path, step_name, "extracted.pkl")
#     # command = "rm " + extracted_pkl_path
#     # os.system(command)
    
#     stage_length_info_path = os.path.join(all_experiment_path, exp_path, step_name, "stage_lengths.json")
#     if not os.path.exists(stage_length_info_path):
#         continue
#     with open(stage_length_info_path, "r") as f:
#         stage_length_info = json.load(f)
    
#     if stage_length_info['reach_to_contact'] > stage_length_info['reach_handle']:
#         stage_length_info['reach_to_contact'] = stage_length_info['reach_to_contact'] - stage_length_info['reach_handle']
#         with open(stage_length_info_path, "w") as f:
#             json.dump(stage_length_info, f, indent=4)

import pybullet as p
quat = [0.0, -0.0, -0.17893988687742693, 0.9838600087839193]
euler = p.getEulerFromQuaternion(quat)
print(euler)
