# python manipulation/gen_demo/gen_demo.py

# demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0628-act3d-obj-46732
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-00/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=5
# task_end_idx=6

# demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0628-act3d-obj-46801
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-05/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=6
# task_end_idx=7

# demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0628-act3d-obj-46874
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-49/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=7
# task_end_idx=8

# demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0628-act3d-obj-46922
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-45/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=8
# task_end_idx=9

# demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0628-act3d-obj-46966
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=9
# task_end_idx=10

# demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0628-act3d-obj-47570
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=10
# task_end_idx=11

# demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0628-act3d-obj-47578
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=11
# task_end_idx=12

demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0628-act3d-obj-48700
save_data_name=debug
exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle
task_beg_idx=12
task_end_idx=13

observation_mode=act3d
pointcloud_num=4500

python 3d_diffusion_policy/extract_data_from_states_2.py \
    --folder_name data/temp/ --object_name storagefurniture \
    --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
    --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
    --pointcloud_num "${pointcloud_num}" \
    --num_experiment 1000 \
    --observation_mode "${observation_mode}" \
    --parallel 0 \

exit


demo_path="${exp_folder}/experiment/${demo_name}"
add_gripper_goal_obs=1
add_gripper_distance_to_closest_point=1
combine_action_steps=2
remove_collision=0
filter_close_zero_action=1
python 3d_diffusion_policy/post_process_demo_per_step.py \
    --zarr_path "data/dp3_demo/${save_data_name}" \
    --new_zarr_path "/scratch/yufei/dp3_demo/${save_data_name}-gripper-goal-${add_gripper_goal_obs}-displacement-to-object-${add_gripper_distance_to_closest_point}-combined-steps-${combine_action_steps}-filter-zero-close-action-${filter_close_zero_action}" \
    --demo_path "${demo_path}" \
    --add_gripper_goal_obs "${add_gripper_goal_obs}" \
    --add_gripper_distance_to_closest_point "${add_gripper_distance_to_closest_point}" \
    --combine_action_steps "${combine_action_steps}" \
    --remove_collision "${remove_collision}" \
    --filter_close_zero_action "${filter_close_zero_action}" \







