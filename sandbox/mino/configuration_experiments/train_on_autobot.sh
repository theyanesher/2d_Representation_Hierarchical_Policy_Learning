source /mnt/RoboGen_sim2real/sandbox/mino/config_train_low_on_high.sh
source /mnt/RoboGen_sim2real/sandbox/mino/init_singularity.sh
source /mnt/RoboGen_sim2real/prepare.sh

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

# run training script
torchrun --standalone --nproc_per_node=4 \
    train_ddp.py --config-name=train_low_level_200_objects.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    use_pretrained_high_level_policy_as_low_level_input=${use_pretrained_high_level_policy_as_low_level_input} \
    task.dataset.zarr_path="[${zarr_path[@]}]"\
    task.env_runner.demo_experiment_path="[${zarr_path[@]}]"\
    task.env_runner.experiment_name="[]" \
    task.env_runner.experiment_folder="[]" \
    task.env_runner.num_point_in_pc="${pointcloud_num}" \
    task.env_runner.use_joint_angle="${use_joint_angle}" \
    task.env_runner.use_segmask="${use_segmask}" \
    task.env_runner.only_handle_points="${only_handle_points}" \
    task.env_runner.use_absolute_waypoint="${use_absolute_waypoint}" \
    horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
    task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
    task.shape_meta.action.shape="[${action_dim}]" \
    policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
    task.dataset.observation_mode="${observation_mode}" \
    task.env_runner.observation_mode="${observation_mode}" \
    policy.encoder_type="${encoder_type}" \
    policy.encoder_output_dim=${encoder_output_dim} \
    policy.normalize_action=${normalize_action} \
    policy.scale_scene_by_pcd=${scale_scene_by_pcd} \
    policy.act3d_encoder_cfg.in_channels=${in_channels} \
    policy.act3d_encoder_cfg.goal_mode=${goal_mode} \
    policy.act3d_encoder_cfg.mode="${encoding_mode}" \
    policy.act3d_encoder_cfg.use_mlp="${use_mlp}" \
    policy.act3d_encoder_cfg.self_attention="${self_attention}" \
    policy.prediction_target="${prediction_target}" \
    policy.act3d_encoder_cfg.use_attn_for_point_features="${use_attn_for_point_features}" \
    policy.act3d_encoder_cfg.pointcloud_backbone="${pointcloud_backbone}" \
    policy.act3d_encoder_cfg.use_lightweight_unet="${use_lightweight_unet}" \
    policy.act3d_encoder_cfg.final_attention="${final_attention}" \
    task.dataset.enumerate=True \
    training.num_epochs="${training_epoches}" \
    training.rollout_every=${rollout_every} \
    training.checkpoint_every=${checkpoint_every} \
    task.env_runner.max_steps=${max_steps} \
    task.dataset.train_ratio="${train_ratio}" \
    task.dataset.num_load_episodes=${num_load_episodes} \
    task.dataset.kept_in_disk=${kept_in_disk} \
    task.dataset.load_per_step=${load_per_step} \
    task.dataset.augmentation_rot="${augmentation_rot}" \
    task.dataset.augmentation_pcd="${augmentation_pcd}" \
    task.dataset.augmentation_scale="${augmentation_scale}" \
    task.dataset.dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']" \
    task.dataset.scale_scene_by_pcd="${scale_scene_by_pcd}" \
    task.dataset.use_absolute_waypoint="${use_absolute_waypoint}" \
    task.dataset.is_pickle="${is_pickle}" \
    dataloader.batch_size="${batch_size}" \
    val_dataloader.batch_size="${batch_size}"
