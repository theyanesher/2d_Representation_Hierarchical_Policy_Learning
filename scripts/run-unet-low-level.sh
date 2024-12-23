pointcloud_num=4500
source scripts/dataset.sh

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

source_dir="/scratch/chialiang/dp3_demo_combine_2_new"
source_dir="/scratch/yufeiw2/dp3_demo_combined_2_step_0"


observation_mode="act3d_goal_mlp"
# observation_mode='act3d_goal_mlp_displacement_gripper_to_object'
encoding_mode="keep_position_feature_in_attention_feature"

horizon=8
n_obs_steps=2 # 2 or 4

##########
training_epoches=100
train_ratio=0.9 # for generalization
num_load_episodes=1000    # for generalization
pc_channel=3 # we should modify this
batch_size=96 #######
# batch_size=112 #######
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
exp_name="1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal"


action_dim=10
agent_pos_dim=10

torchrun --standalone --nproc_per_node=8 \
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


# ${source_dir}/${save_data_name_50},${source_dir}/${save_data_name_51},${source_dir}/${save_data_name_52},${source_dir}/${save_data_name_53},${source_dir}/${save_data_name_54},${source_dir}/${save_data_name_55},${source_dir}/${save_data_name_56},${source_dir}/${save_data_name_57},${source_dir}/${save_data_name_58},${source_dir}/${save_data_name_59}, \
#         ${source_dir}/${save_data_name_60},${source_dir}/${save_data_name_61},${source_dir}/${save_data_name_62},${source_dir}/${save_data_name_63},${source_dir}/${save_data_name_64},${source_dir}/${save_data_name_65},${source_dir}/${save_data_name_66},${source_dir}/${save_data_name_67},${source_dir}/${save_data_name_68},${source_dir}/${save_data_name_69}, \
#         ${source_dir}/${save_data_name_70},${source_dir}/${save_data_name_71},${source_dir}/${save_data_name_72},${source_dir}/${save_data_name_73},${source_dir}/${save_data_name_74},${source_dir}/${save_data_name_75},${source_dir}/${save_data_name_76},${source_dir}/${save_data_name_77},${source_dir}/${save_data_name_78},${source_dir}/${save_data_name_79}, \
#         ${source_dir}/${save_data_name_80},${source_dir}/${save_data_name_81},${source_dir}/${save_data_name_82},${source_dir}/${save_data_name_83},${source_dir}/${save_data_name_84},${source_dir}/${save_data_name_85},${source_dir}/${save_data_name_86},${source_dir}/${save_data_name_87},${source_dir}/${save_data_name_88},${source_dir}/${save_data_name_89}, \
#         ${source_dir}/${save_data_name_90},${source_dir}/${save_data_name_91},${source_dir}/${save_data_name_92},${source_dir}/${save_data_name_93},${source_dir}/${save_data_name_94},${source_dir}/${save_data_name_95},${source_dir}/${save_data_name_96},${source_dir}/${save_data_name_97},${source_dir}/${save_data_name_98},${source_dir}/${save_data_name_99}, \
#         ${source_dir}/${save_data_name_100},${source_dir}/${save_data_name_101},${source_dir}/${save_data_name_102},${source_dir}/${save_data_name_103},${source_dir}/${save_data_name_104},${source_dir}/${save_data_name_105},${source_dir}/${save_data_name_106},${source_dir}/${save_data_name_107},${source_dir}/${save_data_name_108},${source_dir}/${save_data_name_109}, \
#         ${source_dir}/${save_data_name_110},${source_dir}/${save_data_name_111},${source_dir}/${save_data_name_112},${source_dir}/${save_data_name_113},${source_dir}/${save_data_name_114},${source_dir}/${save_data_name_115},${source_dir}/${save_data_name_116},${source_dir}/${save_data_name_117},${source_dir}/${save_data_name_118},${source_dir}/${save_data_name_119}, \
#         ${source_dir}/${save_data_name_120},${source_dir}/${save_data_name_121},${source_dir}/${save_data_name_122},${source_dir}/${save_data_name_123},${source_dir}/${save_data_name_124},${source_dir}/${save_data_name_125},${source_dir}/${save_data_name_126},${source_dir}/${save_data_name_127},${source_dir}/${save_data_name_128},${source_dir}/${save_data_name_129}, \
#         ${source_dir}/${save_data_name_130},${source_dir}/${save_data_name_131},${source_dir}/${save_data_name_132},${source_dir}/${save_data_name_133},${source_dir}/${save_data_name_134},${source_dir}/${save_data_name_135},${source_dir}/${save_data_name_136},${source_dir}/${save_data_name_137},${source_dir}/${save_data_name_138},${source_dir}/${save_data_name_139}, \
#         ${source_dir}/${save_data_name_140},${source_dir}/${save_data_name_141},${source_dir}/${save_data_name_142},${source_dir}/${save_data_name_143},${source_dir}/${save_data_name_144},${source_dir}/${save_data_name_145},${source_dir}/${save_data_name_146},${source_dir}/${save_data_name_147},${source_dir}/${save_data_name_148},${source_dir}/${save_data_name_149}, \
#         ${source_dir}/${save_data_name_150},${source_dir}/${save_data_name_151},${source_dir}/${save_data_name_152},${source_dir}/${save_data_name_153},${source_dir}/${save_data_name_154},${source_dir}/${save_data_name_155},${source_dir}/${save_data_name_156},${source_dir}/${save_data_name_157},${source_dir}/${save_data_name_158},${source_dir}/${save_data_name_159}, \
#         ${source_dir}/${save_data_name_160},${source_dir}/${save_data_name_161},${source_dir}/${save_data_name_162},${source_dir}/${save_data_name_163},${source_dir}/${save_data_name_164},${source_dir}/${save_data_name_165},${source_dir}/${save_data_name_166},${source_dir}/${save_data_name_167},${source_dir}/${save_data_name_168},${source_dir}/${save_data_name_169}, \
#         ${source_dir}/${save_data_name_170},${source_dir}/${save_data_name_171},${source_dir}/${save_data_name_172},${source_dir}/${save_data_name_173},${source_dir}/${save_data_name_174},${source_dir}/${save_data_name_175},${source_dir}/${save_data_name_176},${source_dir}/${save_data_name_177},${source_dir}/${save_data_name_178},${source_dir}/${save_data_name_179}, \
#         ${source_dir}/${save_data_name_180},${source_dir}/${save_data_name_181},${source_dir}/${save_data_name_182},${source_dir}/${save_data_name_183},${source_dir}/${save_data_name_184},${source_dir}/${save_data_name_185},${source_dir}/${save_data_name_186},${source_dir}/${save_data_name_187},${source_dir}/${save_data_name_188},${source_dir}/${save_data_name_189}, \
#         ${source_dir}/${save_data_name_190},${source_dir}/${save_data_name_191},${source_dir}/${save_data_name_192},${source_dir}/${save_data_name_193},${source_dir}/${save_data_name_194},${source_dir}/${save_data_name_195},${source_dir}/${save_data_name_196}




    
