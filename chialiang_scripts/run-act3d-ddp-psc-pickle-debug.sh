
# python manipulation/old_test_opening_primitve.py

func=${1}


if [ $# -lt 1 ]; then
    echo "Usage: $0 [func]"
    exit
fi


if [ $func = 'post_process' ]; then 
    python3 copy_post_processing.py
fi


cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

export CUDA_VISIBLE_DEVICES=1,2,3,6
if [ $func = 'train' ]; then 

    source_dir="/jet/projects/cis240052p/ywang59/dp3_demo"
    source_dir="/scratch/chialiang/dp3_demo"
    # source_dir="/jet/projects/cis240052p/ckuo1/dp3_demo"

    observation_mode="act3d_goal_mlp"
    # observation_mode='act3d_goal_mlp_displacement_gripper_to_object'
    encoding_mode="" # only useful for UNet based methods
    # encoding_mode="keep_position_feature_in_attention_feature"
    # encoding_mode="keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"

    # horizon=4
    horizon=8
    n_obs_steps=2 # 2 or 4
    # num_load_episodes=10 # for debuging

    ##########
    training_epoches=10
    train_ratio=0.9 # for generalization
    num_load_episodes=1000    # for generalization
    pc_channel=3 # we should modify this
    batch_size=400 #######
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
    augmentation_rot=false
    augmentation_pcd=false
    use_absolute_waypoint=false
    dense_pcd_for_goal=false
    ##########
    use_attn_for_point_features=false
    pointcloud_backbone='mlp'
    ##########
    is_pickle=true
    ##########

    time_stamp=$(date +%m%d%H%M)
    exp_name="${time_stamp}-${observation_mode}-n_obs_steps-${n_obs_steps}-horizon-${horizon}-num_load_episodes-${num_load_episodes}-test"

    action_dim=10
    agent_pos_dim=10
    
    # saved data paths
    # save_data_name_0=0702-obj-45448-dense_pcd_on_goal
    save_data_name_0=0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point
    exp_folder_0=data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle
    demo_name_0=0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first

    torchrun --standalone --nproc_per_node=4 \
        train_ddp.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
        task.dataset.zarr_path="[\
            ${source_dir}/${save_data_name_0}\
        ]"\
        task.env_runner.demo_experiment_path="[\
            ${source_dir}/${save_data_name_0}\
        ]" \
        task.env_runner.experiment_name="[\
            ${demo_name_0}\
        ]" \
        task.env_runner.experiment_folder="[\
            ${exp_folder_0}\
        ]" \
        task.env_runner.num_point_in_pc="${pointcloud_num}" \
        task.env_runner.use_joint_angle="${use_joint_angle}" \
        task.env_runner.use_segmask="${use_segmask}" \
        task.env_runner.only_handle_points="${only_handle_points}" \
        task.env_runner.dense_pcd_for_goal="${dense_pcd_for_goal}" \
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
        policy.act3d_encoder_cfg.self_attention="${self_attention}" \
        policy.act3d_encoder_cfg.use_attn_for_point_features="${use_attn_for_point_features}" \
        policy.act3d_encoder_cfg.pointcloud_backbone="${pointcloud_backbone}" \
        policy.act3d_encoder_cfg.use_lightweight_unet="${use_lightweight_unet}" \
        policy.act3d_encoder_cfg.final_attention="${final_attention}" \
        task.dataset.enumerate=True \
        training.num_epochs=${training_epoches} \
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
        task.dataset.is_pickle="${is_pickle}" \
        dataloader.batch_size="${batch_size}" \
        val_dataloader.batch_size="${batch_size}"

fi 


if [ $func = 'eval' ]; then 
    python eval_robogen_parallel_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=eval
    # python eval_robogen_new.py --config-name=dp3.yaml task=robogen_open_door exp_name=debug
    # singularity shell --bind /project_data/held/yufeiw2/RoboGen_sim2real/:/mnt/RoboGen_sim2real/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif
fi 

