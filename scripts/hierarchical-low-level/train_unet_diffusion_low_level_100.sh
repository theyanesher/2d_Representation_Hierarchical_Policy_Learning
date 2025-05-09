#!/bin/bash


cd /mnt/RoboGen_sim2real
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/rl_games:$PYTHONPATH
export PYTHONPATH=${PWD}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy:$PYTHONPATH
export PROJECT_DIR=${PWD}
export WANDB_API_KEY=c9187c7dfcc339af75f2f47c3b80c95743057b42
export HYDRA_FULL_ERROR=1
pointcloud_num=4500

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

source_dir="/mnt/RoboGen_sim2real/data/dp3_demo_combined_2_step_0"


observation_mode="act3d_goal_mlp"
encoding_mode="keep_position_feature_in_attention_feature"

horizon=8
n_obs_steps=2 # 2 or 4

##########
training_epoches=100
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
normalize_action=true
augmentation_rot=false
augmentation_pcd=true
use_absolute_waypoint=false
dense_pcd_for_goal=false
##########
use_attn_for_point_features=false
pointcloud_backbone='mlp'
##########
is_pickle=true
##########
use_pretrained_high_level_policy_as_low_level_input=false
##########

time_stamp=$(date +%m%d%H%M)
exp_name="test-low-level"


action_dim=10
agent_pos_dim=10

torchrun --standalone --nproc_per_node=8 \
    train_ddp.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    use_pretrained_high_level_policy_as_low_level_input=${use_pretrained_high_level_policy_as_low_level_input} \
    task.dataset.zarr_path=articulated \
    task.env_runner.demo_experiment_path="[]" \
    task.env_runner.experiment_name="[]" \
    task.env_runner.experiment_folder="[]" \
    task.env_runner.num_point_in_pc="${pointcloud_num}" \
    task.env_runner.use_absolute_waypoint="${use_absolute_waypoint}" \
    horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
    task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
    task.shape_meta.action.shape="[${action_dim}]" \
    policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
    task.dataset.observation_mode="${observation_mode}" \
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
    training.num_epochs="${training_epoches}" \
    training.rollout_every=2000 \
    training.checkpoint_every=4 \
    task.env_runner.max_steps=35 \
    task.dataset.train_ratio="${train_ratio}" \
    task.dataset.num_load_episodes=${num_load_episodes} \
    task.dataset.kept_in_disk=true \
    task.dataset.load_per_step=true \
    task.dataset.augmentation_rot="${augmentation_rot}" \
    task.dataset.augmentation_pcd="${augmentation_pcd}" \
    task.dataset.use_absolute_waypoint="${use_absolute_waypoint}" \
    task.dataset.is_pickle="${is_pickle}" \
    dataloader.batch_size="${batch_size}" \
    val_dataloader.batch_size="${batch_size}" \
    task.dataset.dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']" \
    policy.noise_model_type=unet \
    policy.policy_type=low_level \
    load_checkpoint_path=/mnt/RoboGen_sim2real/ckpt/low_level/checkpoints/epoch-96.ckpt



    
