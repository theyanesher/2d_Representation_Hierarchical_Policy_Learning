pointcloud_num=4500
source scripts/datasets/train_dataset_50.sh

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

source_dir="/home/yufei/projects/RoboGen-sim2real/data/dp3_demo_combined_2_step_0"


observation_mode="act3d"
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
# exp_name="1107-200-combined-low-level-unet-diffusion-chialiang-hyper-parameter"
exp_name="1121-no-hierrachical-50-training-objs"


action_dim=10
agent_pos_dim=10

torchrun --standalone --nproc_per_node=3 \
    train_ddp.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    use_pretrained_high_level_policy_as_low_level_input=${use_pretrained_high_level_policy_as_low_level_input} \
    task.dataset.zarr_path="[\
        ${source_dir}/${save_data_name_0},\
        ${source_dir}/${save_data_name_1},\
        ${source_dir}/${save_data_name_2},\
        ${source_dir}/${save_data_name_3},\
        ${source_dir}/${save_data_name_4},\
        ${source_dir}/${save_data_name_5},\
        ${source_dir}/${save_data_name_6},\
        ${source_dir}/${save_data_name_7},\
        ${source_dir}/${save_data_name_8},\
        ${source_dir}/${save_data_name_9},\
        ${source_dir}/${save_data_name_10},\
        ${source_dir}/${save_data_name_11},\
        ${source_dir}/${save_data_name_12},\
        ${source_dir}/${save_data_name_13},\
        ${source_dir}/${save_data_name_14},\
        ${source_dir}/${save_data_name_15},\
        ${source_dir}/${save_data_name_16},\
        ${source_dir}/${save_data_name_17},\
        ${source_dir}/${save_data_name_18},\
        ${source_dir}/${save_data_name_19},\
        ${source_dir}/${save_data_name_20},\
        ${source_dir}/${save_data_name_21},\
        ${source_dir}/${save_data_name_22},\
        ${source_dir}/${save_data_name_23},\
        ${source_dir}/${save_data_name_24},\
        ${source_dir}/${save_data_name_25},\
        ${source_dir}/${save_data_name_26},\
        ${source_dir}/${save_data_name_27},\
        ${source_dir}/${save_data_name_28},\
        ${source_dir}/${save_data_name_29},\
        ${source_dir}/${save_data_name_30},\
        ${source_dir}/${save_data_name_31},\
        ${source_dir}/${save_data_name_32},\
        ${source_dir}/${save_data_name_33},\
        ${source_dir}/${save_data_name_34},\
        ${source_dir}/${save_data_name_35},\
        ${source_dir}/${save_data_name_36},\
        ${source_dir}/${save_data_name_37},\
        ${source_dir}/${save_data_name_38},\
        ${source_dir}/${save_data_name_39},\
        ${source_dir}/${save_data_name_40},\
        ${source_dir}/${save_data_name_41},\
        ${source_dir}/${save_data_name_42},\
        ${source_dir}/${save_data_name_43},\
        ${source_dir}/${save_data_name_44},\
        ${source_dir}/${save_data_name_45},\
        ${source_dir}/${save_data_name_46},\
        ${source_dir}/${save_data_name_47},\
        ${source_dir}/${save_data_name_48},\
        ${source_dir}/${save_data_name_49}\
        ]" \
    task.env_runner.demo_experiment_path="[\
        ${source_dir}/${save_data_name_0},\
        ${source_dir}/${save_data_name_1},\
        ${source_dir}/${save_data_name_2},\
        ${source_dir}/${save_data_name_3},\
        ${source_dir}/${save_data_name_4},\
        ${source_dir}/${save_data_name_5},\
        ${source_dir}/${save_data_name_6},\
        ${source_dir}/${save_data_name_7},\
        ${source_dir}/${save_data_name_8},\
        ${source_dir}/${save_data_name_9},\
        ${source_dir}/${save_data_name_10}\
    ]" \
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
    policy.act3d_encoder_cfg.goal_mode=null \
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
    policy.policy_type=low_level



    
