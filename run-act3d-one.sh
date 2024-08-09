
# python manipulation/old_test_opening_primitve.py

func=${1}

if [ $# -lt 1 ]; then
    echo "Usage: $0 [func]"
    exit
fi

# demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # save_data_name=0527-act3d-always-close
# save_data_name=0626-act3d-obj-41510-displacement-to-handle
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=0
# task_end_idx=1 # for debugging
# opened_threshold=0.65

# # demo_name=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # save_data_name=0531-act3d-obj-46462
# # exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle
# # task_beg_idx=4
# # task_end_idx=5
# # opened_threshold=2.6

# observation_mode=act3d
# observation_mode=act3d_goal
# observation_mode=dp3_goal_gripper
# observation_mode=dp3_goal_gripper
# observation_mode=act3d_goal_gripper_4
pointcloud_num=4500

# # python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-46462
# # python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-45448

if [ $func = 'collect' ]; then 

    demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first

    save_data_name=0705-dp3-obj-41510-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-45448-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46462-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46732-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46801-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46874-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46922-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-46966-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-47570-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-47578-goal_dense_gripper_on_pcd
    save_data_name=0705-dp3-obj-48700-goal_dense_gripper_on_pcd

    task_beg_idx=0
    task_end_idx=1

    task_beg_idx=2
    task_end_idx=3
    
    task_beg_idx=4
    task_end_idx=5
    
    task_beg_idx=5
    task_end_idx=6
    
    task_beg_idx=6
    task_end_idx=7
    
    task_beg_idx=7
    task_end_idx=8
    
    task_beg_idx=8
    task_end_idx=9
    
    task_beg_idx=9
    task_end_idx=10
    
    task_beg_idx=10
    task_end_idx=11
    
    task_beg_idx=11
    task_end_idx=12
    
    task_beg_idx=12
    task_end_idx=13

    post_fix='dp3_goal_gripper_dense'
    observation_mode="${post_fix}"

    python 3d_diffusion_policy/extract_data_from_states_2.py --folder_name data/temp/ --object_name storagefurniture \
        --save_path "dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
        --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
        --pointcloud_num "${pointcloud_num}" \
        --observation_mode "${observation_mode}" \
        --parallel 1 \
        --add_distractors 0 \
        --num_experiment 1000
fi

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy


export CUDA_VISIBLE_DEVICES=4,5,6,7

if [ $func = 'train' ]; then 

    observation_mode="act3d_goal_mlp"
    # observation_mode='act3d_goal_mlp_displacement_gripper_to_object'
    encoding_mode="keep_position_feature_in_attention_feature"
    # encoding_mode="keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"

    # horizon=4
    horizon=8
    n_obs_steps=2 # 2 or 4
    # num_load_episodes=10 # for debuging

    ##########
    num_epochs=101
    train_ratio=0.9 # for generalization
    num_load_episodes=1000    # for generalization
    pc_channel=3 # we should modify this
    # batch_size=256 #######
    batch_size=320 #######
    encoder_type=act3d
    use_mlp=1
    use_lightweight_unet=0
    in_channels=3 ####
    self_attention=false
    final_attention=false
    # normalize_action=true
    # augmentation_rot=false
    # augmentation_pcd=false
    normalize_action=true
    augmentation_rot=true
    augmentation_pcd=false
    # use_absolute_waypoint=false
    use_absolute_waypoint=false
    ##########

    time_stamp=$(date +%m%d%H%M)
    # exp_name="${time_stamp}-${observation_mode}-n_obs_steps-${n_obs_steps}-horizon-${horizon}-num_load_episodes-${num_load_episodes}-normalize_action"
    exp_name="${time_stamp}-${observation_mode}-n_obs_steps-${n_obs_steps}-horizon-${horizon}-num_load_episodes-${num_load_episodes}-one-aug_rot"

    action_dim=10
    agent_pos_dim=10
    
    # saved data paths
    save_data_name_1=0705-obj-45448
    exp_folder_1=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
    demo_name_1=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    
    torchrun --standalone --nproc_per_node=4 \
        train_ddp.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
        task.dataset.zarr_path="[\
            /scratch/chialiang/dp3_demo/${save_data_name_1}\
        ]"\
        task.env_runner.demo_experiment_path="[\
            /scratch/chialiang/dp3_demo/${save_data_name_1}\
        ]" \
        task.env_runner.experiment_name="[\
            ${demo_name_1}\
        ]" \
        task.env_runner.experiment_folder="[\
            ${exp_folder_1}\
        ]" \
        task.env_runner.num_point_in_pc="${pointcloud_num}" \
        task.env_runner.use_joint_angle="${use_joint_angle}" \
        task.env_runner.use_segmask="${use_segmask}" \
        task.env_runner.only_handle_points="${only_handle_points}" \
        horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
        task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
        task.shape_meta.action.shape="[${action_dim}]" \
        policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
        task.dataset.observation_mode="${observation_mode}" \
        task.env_runner.observation_mode="${observation_mode}" \
        policy.encoder_type="${encoder_type}" \
        policy.encoder_output_dim=60 \
        policy.normalize_action=${normalize_action} \
        policy.act3d_encoder_cfg.in_channels=${in_channels} \
        policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \
        policy.act3d_encoder_cfg.mode="${encoding_mode}" \
        policy.act3d_encoder_cfg.use_mlp="${use_mlp}" \
        policy.act3d_encoder_cfg.use_lightweight_unet="${use_lightweight_unet}" \
        policy.act3d_encoder_cfg.self_attention="${self_attention}" \
        policy.act3d_encoder_cfg.final_attention="${final_attention}" \
        task.dataset.enumerate=True \
        training.num_epochs=${num_epochs} \
        training.rollout_every=50 \
        training.checkpoint_every=2 \
        task.env_runner.max_steps=35 \
        task.dataset.train_ratio="${train_ratio}" \
        task.dataset.num_load_episodes="${num_load_episodes}" \
        task.dataset.kept_in_disk=true \
        task.dataset.load_per_step=true \
        task.dataset.augmentation_rot="${augmentation_rot}" \
        task.dataset.augmentation_pcd="${augmentation_pcd}" \
        task.dataset.use_absolute_waypoint="${use_absolute_waypoint}" \
        dataloader.batch_size="${batch_size}" \
        val_dataloader.batch_size="${batch_size}"

fi 


if [ $func = 'eval' ]; then 
    python eval_robogen_parallel_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
    # singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif
fi 



# # python manipulation/gen_demo/gen_demo.py

# demo_name=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # save_data_name=0527-act3d-always-close
# save_data_name=0617-act3d-obj-41510-displacement-to-handle
# exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
# task_beg_idx=0
# task_end_idx=1
# opened_threshold=0.65

# # demo_name=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # # save_data_name=0531-act3d-obj-45448
# # save_data_name=0611-act3d-obj-45448-distractors-wzy
# # exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
# # task_beg_idx=2
# # task_end_idx=3
# # opened_threshold=0.4

# # # demo_name=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# # # save_data_name=0531-act3d-obj-46462
# # # exp_folder=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle
# # # task_beg_idx=4
# # # task_end_idx=5
# # # opened_threshold=2.6

# # observation_mode=act3d
# observation_mode=act3d_goal
# # observation_mode=act3d_displacement_to_handle
# pointcloud_num=4500

# # # python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-46462
# # # python 3d_diffusion_policy/filter_simulation_error.py --folder_name data/temp/ --object_name storagefurniture     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}"     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}"     --pointcloud_num "${pointcloud_num}"     --use_extracted 0     --num_experiment 1000     --observation_mode "${observation_mode}"     --parallel 0     --opened_threshold "${opened_threshold}" --demo_folder /project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0531-act3d-obj-45448

# # python 3d_diffusion_policy/extract_data_from_states_2.py --folder_name data/temp/ --object_name storagefurniture \
# #     --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
# #     --task_beg_idx "${task_beg_idx}" --task_end_idx "${task_end_idx}" \
# #     --pointcloud_num "${pointcloud_num}" \
# #     --use_extracted 0 \
# #     --num_experiment 40 \
# #     --observation_mode "${observation_mode}" \
# #     --parallel 0 \
# #     --opened_threshold "${opened_threshold}" \
# #     --add_distractors 0

# # exit

# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

# # save_data_name_0=0527-act3d-always-close
# # save_data_name_0=0527-act3d-always-close-with-goal
# # save_data_name_0=0607-act3d-obj-41510-remove-reaching-collision-resize-2
# save_data_name_0=0617-act3d-obj-41510-remove-reaching-collision-resize-2-goal
# # save_data_name_1=0531-act3d-obj-45448
# save_data_name_1=0607-act3d-obj-45448-remove-reaching-collision-resize-2-full
# # save_data_name_2=0531-act3d-obj-46462
# save_data_name_2=0607-act3d-obj-46462-remove-reaching-collision-resize-2

# demo_name_0=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# demo_name_1=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
# demo_name_2=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first

# exp_folder_0=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
# exp_folder_1=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
# exp_folder_2=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle

# # horizon=4
# horizon=8
# n_obs_steps=2
# train_ratio=1
# num_load_episodes=10
# # exp_name="0528-act3d-train-ratio-${train_ratio}"
# # exp_name="0602-act3d-obj-45448-train-ratio-${train_ratio}"
# # exp_name="0604-act3d-obj-46462-train-ratio-${train_ratio}-filtered"
# # exp_name="0603-act3d-3-obj-train-ratio-${train_ratio}"
# # exp_name="0606-act3d-3-obj-train-ratio-${train_ratio}"
# # exp_name="0608-act3d-obj-41510-goal-train-ratio-${train_ratio}"
# # exp_name="0609-act3d-obj-45448-horizon-${horizon}-train-ratio-${train_ratio}"
# # exp_name="0612-act3d-3-obj-horizon-${horizon}-num_load_episodes-${num_load_episodes}"
# exp_name="0618-act3d-goal-horizon-${horizon}-num_load_episodes-${num_load_episodes}"
# exp_name="test-robo-cluster-singularity"

# action_dim=10
# agent_pos_dim=10
# pc_channel=3 
# batch_size=60

# # python train.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
# #     task.dataset.zarr_path="[${PROJECT_DIR}/data/dp3_demo/${save_data_name_0}]" \
# #     task.env_runner.demo_experiment_path="[${PROJECT_DIR}/data/dp3_demo/${save_data_name_0}]" \
# #     task.env_runner.experiment_name="[${demo_name_0}]" \
# #     task.env_runner.experiment_folder="[${exp_folder_0}]" \
# #     task.env_runner.num_point_in_pc="${pointcloud_num}" \
# #     task.env_runner.use_joint_angle="${use_joint_angle}" \
# #     task.env_runner.use_segmask="${use_segmask}" \
# #     task.env_runner.only_handle_points="${only_handle_points}" \
# #     horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
# #     task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
# #     task.shape_meta.action.shape="[${action_dim}]" \
# #     policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
# #     task.dataset.observation_mode="${observation_mode}" \
# #     task.env_runner.observation_mode="${observation_mode}" \
# #     policy.encoder_type=act3d \
# #     policy.encoder_output_dim=60 \
# #     policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \
# #     policy.act3d_encoder_cfg.mode=keep_position_feature_in_attention_feature \
# #     task.dataset.enumerate=True \
# #     training.rollout_every=200 \
# #     training.checkpoint_every=200 \
# #     task.env_runner.max_steps=35 \
# #     task.dataset.train_ratio="${train_ratio}" \
# #     task.dataset.num_load_episodes="${num_load_episodes}" \
# #     task.dataset.kept_in_disk=false \
# #     task.dataset.load_per_step=false \

# python eval_robogen_parallel_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
# # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
# # singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif
