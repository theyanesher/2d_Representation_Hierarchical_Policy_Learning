cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

horizon=4
n_obs_steps=2
n_action_steps=2
train_ratio=1.0
num_load_episodes=75

observation_mode=act3d_goal_displacement_gripper_to_object
pointcloud_num=4500
action_dim=10
agent_pos_dim=10
pc_channel=3
prediction_target=goal_gripper_pcd

use_mlp=0
exp_name="paper-unet-diffusion-high-level-100-obj-camera-random-0107"

torchrun --standalone --nproc_per_node=7 train_ddp.py \
    --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    task.dataset.zarr_path=camera_random_100_obj_high_level  \
    task.env_runner.demo_experiment_path="[]" \
    task.env_runner.experiment_name="[]" \
    task.env_runner.experiment_folder="[]" \
    task.env_runner.num_point_in_pc="${pointcloud_num}" \
    task.env_runner.use_joint_angle="${use_joint_angle}" \
    task.env_runner.use_segmask="${use_segmask}" \
    task.env_runner.only_handle_points="${only_handle_points}" \
    horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
    task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
    task.shape_meta.action.shape="[${action_dim}]" \
    policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
    policy.act3d_encoder_cfg.in_channels="${pc_channel}" \
    task.env_runner.observation_mode="${observation_mode}" \
    task.dataset.observation_mode="${observation_mode}" \
    policy.encoder_type=act3d \
    policy.encoder_output_dim=60 \
    task.dataset.enumerate=True \
    training.num_epochs=61 \
    training.rollout_every=2000 \
    training.checkpoint_every=5 \
    task.env_runner.max_steps=35 \
    training.val_every=5 \
    task.dataset.kept_in_disk=true \
    task.dataset.load_per_step=true \
    task.dataset.num_load_episodes="${num_load_episodes}" \
    task.dataset.train_ratio="${train_ratio}" \
    dataloader.batch_size=25 \
    val_dataloader.batch_size=25 \
    policy.act3d_encoder_cfg.mode=keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object \
    policy.act3d_encoder_cfg.self_attention=true \
    policy.prediction_target="${prediction_target}" \
    task.dataset.prediction_target="${prediction_target}" \
    policy.act3d_encoder_cfg.use_mlp="${use_mlp}" \
    policy.act3d_encoder_cfg.pointcloud_backbone=pointnet2 \
    policy.noise_model_type=unet \
    task.dataset.dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']" \
    policy.noise_scheduler.prediction_type=epsilon \
    task.dataset.is_pickle=true \
    task.dataset.augmentation_pcd=false \

