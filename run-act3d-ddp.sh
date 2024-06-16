# python manipulation/old_test_opening_primitve.py

# demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0527-act3d-always-close
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=0
# task_end_idx=1
# opened_threshold=0.65

demo_name=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
save_data_name=0531-act3d-obj-45448
exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
task_beg_idx=2
task_end_idx=3
opened_threshold=0.4

# demo_name=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# save_data_name=0531-act3d-obj-46462
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=4
# task_end_idx=5
# opened_threshold=2.6

observation_mode=act3d
# observation_mode=act3d_goal
pointcloud_num=4500

# python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-46462
# python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-45448

# python 3d_diffusion_policy/extract_data_from_states_2.py --folder_name data/temp/ --object_name storagefurniture \
#     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
#     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
#     --pointcloud_num "${pointcloud_num}" \
#     --use_extracted 0 \
#     --num_experiment 1000 \
#     --observation_mode "${observation_mode}" \
#     --parallel 0 \
#     --opened_threshold "${opened_threshold}" 

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

# save_data_name_0=0527-act3d-always-close
# save_data_name_0=0527-act3d-always-close-with-goal
save_data_name_0=0607-act3d-obj-41510-remove-reaching-collision-resize-2
# save_data_name_1=0531-act3d-obj-45448
save_data_name_1=0607-act3d-obj-45448-remove-reaching-collision-resize-2-full
# save_data_name_2=0531-act3d-obj-46462
save_data_name_2=0607-act3d-obj-46462-remove-reaching-collision-resize-2

demo_name_0=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
demo_name_1=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
demo_name_2=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first

exp_folder_0=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
exp_folder_1=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
exp_folder_2=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle

horizon=4
# horizon=8
n_obs_steps=2
train_ratio=0.1
# exp_name="0528-act3d-train-ratio-${train_ratio}"
# exp_name="0602-act3d-obj-45448-train-ratio-${train_ratio}"
# exp_name="0604-act3d-obj-46462-train-ratio-${train_ratio}-filtered"
# exp_name="0603-act3d-3-obj-train-ratio-${train_ratio}"
# exp_name="0606-act3d-3-obj-train-ratio-${train_ratio}"
# exp_name="0608-act3d-obj-41510-goal-train-ratio-${train_ratio}"
# exp_name="0609-act3d-obj-45448-horizon-${horizon}-train-ratio-${train_ratio}"
# exp_name="0612-act3d-3-obj-horizon-${horizon}-train-ratio-${train_ratio}"
exp_name="debug-ddp"

action_dim=10
agent_pos_dim=10
pc_channel=3

torchrun --standalone --nproc_per_node=4 \
    train_ddp.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    task.dataset.zarr_path="[${PROJECT_DIR}/data/dp3_demo/${save_data_name_0}]" \
    task.env_runner.demo_experiment_path="[${PROJECT_DIR}/data/dp3_demo/${save_data_name_0}]" \
    task.env_runner.experiment_name="[${demo_name_0}]" \
    task.env_runner.experiment_folder="[${exp_folder_0}]" \
    task.env_runner.num_point_in_pc="${pointcloud_num}" \
    task.env_runner.use_joint_angle="${use_joint_angle}" \
    task.env_runner.use_segmask="${use_segmask}" \
    task.env_runner.only_handle_points="${only_handle_points}" \
    horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
    task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
    task.shape_meta.action.shape="[${action_dim}]" \
    policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
    task.dataset.train_ratio="${train_ratio}" \
    task.env_runner.observation_mode="${observation_mode}" \
    task.dataset.observation_mode="${observation_mode}" \
    policy.encoder_type=act3d \
    policy.encoder_output_dim=60 \
    task.dataset.enumerate=True \
    training.rollout_every=2000 \
    training.checkpoint_every=100 \
    task.env_runner.max_steps=70 \
    training.val_every=5 \
    # policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \

# python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
# python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
# singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif