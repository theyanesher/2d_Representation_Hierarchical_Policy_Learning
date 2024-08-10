
func=${1}

if [ $# -lt 1 ]; then
    echo "Usage: $0 [func]"
    exit
fi

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

if [ $func = 'train' ]; then 

    # observation_mode="dp3_goal_gripper_whole"
    # observation_mode="dp3_goal_gripper_part"
    # observation_mode="act3d_goal"
    observation_mode="act3d_goal_mlp"
    # observation_mode='act3d_goal_mlp_displacement_gripper_to_object'

    # saved data paths
    save_data_name_0=0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action
    save_data_name_1=0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point
    save_data_name_2=0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action
    save_data_name_3=0703-act3d-mlp-obj-46732-goal
    save_data_name_4=0703-act3d-mlp-obj-46801-goal
    save_data_name_5=0703-act3d-mlp-obj-46874-goal
    save_data_name_6=0703-act3d-mlp-obj-46922-goal
    save_data_name_7=0703-act3d-mlp-obj-46966-goal
    save_data_name_8=0703-act3d-mlp-obj-47570-goal
    save_data_name_9=0703-act3d-mlp-obj-47578-goal
    # save_data_name_10=0703-act3d-mlp-obj-48700-goal

    demo_name_0=0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_1=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_2=0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_3=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_4=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_5=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_6=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_7=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_8=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    demo_name_9=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
    # demo_name_10=0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first

    exp_folder_0=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_1=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_2=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_3=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-10/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_4=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-00/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_5=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-05/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_6=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-49/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_7=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_8=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle
    exp_folder_9=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle
    # exp_folder_10=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle

    # horizon=4
    horizon=8
    n_obs_steps=2
    # num_load_episodes=10 # for debuging

    ##########
    train_ratio=0.9 # for generalization
    num_load_episodes=1000 # for generalization
    pc_channel=3 # we should modify this
    batch_size=512 #######
    encoder_type=act3d
    use_mlp=1
    in_channels=3 ####
    ##########

    time_stamp=$(date +%m%d%H%M)
    exp_name="${time_stamp}-${observation_mode}-horizon-${horizon}-num_load_episodes-${num_load_episodes}"

    action_dim=10
    agent_pos_dim=10
    
    python train.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
        task.dataset.zarr_path="[/scratch/yufei/dp3_demo/${save_data_name_0}, /scratch/yufei/dp3_demo/${save_data_name_1}, /scratch/yufei/dp3_demo/${save_data_name_2}, /scratch/chialiang/dp3_demo/${save_data_name_3}, /scratch/chialiang/dp3_demo/${save_data_name_4}, /scratch/chialiang/dp3_demo/${save_data_name_5}, /scratch/chialiang/dp3_demo/${save_data_name_6}, /scratch/chialiang/dp3_demo/${save_data_name_7}, /scratch/chialiang/dp3_demo/${save_data_name_8}, /scratch/chialiang/dp3_demo/${save_data_name_9}]" \
        task.env_runner.demo_experiment_path="[/scratch/yufei/dp3_demo/${save_data_name_0}, /scratch/yufei/dp3_demo/${save_data_name_1}, /scratch/yufei/dp3_demo/${save_data_name_2}, /scratch/chialiang/dp3_demo/${save_data_name_3}, /scratch/chialiang/dp3_demo/${save_data_name_4}, /scratch/chialiang/dp3_demo/${save_data_name_5}, /scratch/chialiang/dp3_demo/${save_data_name_6}, /scratch/chialiang/dp3_demo/${save_data_name_7}, /scratch/chialiang/dp3_demo/${save_data_name_8}, /scratch/chialiang/dp3_demo/${save_data_name_9}]" \
        task.env_runner.experiment_name="[${demo_name_0}, ${demo_name_1}, ${demo_name_2}, ${demo_name_3}, ${demo_name_4}, ${demo_name_5}, ${demo_name_6}, ${demo_name_7}, ${demo_name_8}, ${demo_name_9}]" \
        task.env_runner.experiment_folder="[${exp_folder_0}, ${exp_folder_1}, ${exp_folder_2}, ${exp_folder_3}, ${exp_folder_4}, ${exp_folder_5}, ${exp_folder_6}, ${exp_folder_7}, ${exp_folder_8}, ${exp_folder_9}]" \
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
        policy.act3d_encoder_cfg.in_channels=${in_channels} \
        policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \
        policy.act3d_encoder_cfg.mode=keep_position_feature_in_attention_feature \
        policy.act3d_encoder_cfg.use_mlp="${use_mlp}" \
        task.dataset.enumerate=True \
        training.num_epochs=210 \
        training.rollout_every=50 \
        training.checkpoint_every=20 \
        task.env_runner.max_steps=35 \
        task.dataset.train_ratio="${train_ratio}" \
        task.dataset.num_load_episodes="${num_load_episodes}" \
        task.dataset.kept_in_disk=true \
        task.dataset.load_per_step=true \
        dataloader.batch_size="${batch_size}" \
        val_dataloader.batch_size="${batch_size}" \

fi 


if [ $func = 'eval' ]; then 
    python eval_robogen_parallel_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
    # singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif
fi 

