source scripts/dataset.sh

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

source_dir="/scratch/chialiang/dp3_demo_combine_2_new"
source_dir="/scratch/yufeiw2/dp3_demo_combine_2_new"
source_dir2="/scratch/yufeiw2/dp3_demo_combined_2_step"
source_dir="/scratch/yufeiw2/dp3_demo_combined_2_step_0"

observation_mode="act3d_goal_mlp"
# observation_mode='act3d_goal_mlp_displacement_gripper_to_object'
encoding_mode="keep_position_feature_in_attention_feature"

horizon=8
n_obs_steps=2 # 2 or 4

##########
pointcloud_num=4500
training_epoches=100
train_ratio=0.9 # for generalization
num_load_episodes=1000    # for generalization
pc_channel=3 # we should modify this
batch_size=30 #######
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

# exp_name="1110-600-combined-low-level-transformer-diffusion-chialiang-hyper-parameter"
# exp_name="1111-200-combined-low-level-transformer-diffusion-chialiang-hyper-parameter-w-noise"
# exp_name="1114-600-combined-low-level-transformer-diffusion-chialiang-hyper-parameter-load-200-model"
exp_name="1121-50-combined-low-level-transformer-diffusion-no-dense-step-around-goal"

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
        ]"\
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
    policy.noise_model_type=transformer \
    policy.policy_type=low_level \
    policy.noise_scheduler.prediction_type=sample \
    # load_checkpoint_path=/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1107-200-combined-low-level-transformer-diffusion-chialiang-hyper-parameter/2024.11.07/01.15.22_train_dp3_robogen_open_door/checkpoints/epoch-54.ckpt \
    # training.pretrained_weighted_displacement_goal_model=/project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-10-09_use_75_episodes_500-obj/model_57.pth \
    # task.dataset.augmentation_goal_gripper_pcd=true \
    # training.add_noise_to_goal_gripper_pcd=true \


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
#         ${source_dir}/${save_data_name_190},${source_dir}/${save_data_name_191},${source_dir}/${save_data_name_192},${source_dir}/${save_data_name_193},${source_dir}/${save_data_name_194},${source_dir}/${save_data_name_195},${source_dir}/${save_data_name_196},\
#         ${source_dir2}/${save_data_name_197},${source_dir2}/${save_data_name_198},${source_dir2}/${save_data_name_199},\
#         ${source_dir2}/${save_data_name_200}, ${source_dir2}/${save_data_name_201}, ${source_dir2}/${save_data_name_202}, ${source_dir2}/${save_data_name_203}, ${source_dir2}/${save_data_name_204}, ${source_dir2}/${save_data_name_205}, ${source_dir2}/${save_data_name_206}, ${source_dir2}/${save_data_name_207}, ${source_dir2}/${save_data_name_208}, ${source_dir2}/${save_data_name_209}, \
#         ${source_dir2}/${save_data_name_210}, ${source_dir2}/${save_data_name_211}, ${source_dir2}/${save_data_name_212}, ${source_dir2}/${save_data_name_213}, ${source_dir2}/${save_data_name_214}, ${source_dir2}/${save_data_name_215}, ${source_dir2}/${save_data_name_216}, ${source_dir2}/${save_data_name_217}, ${source_dir2}/${save_data_name_218}, ${source_dir2}/${save_data_name_219}, \
#         ${source_dir2}/${save_data_name_220}, ${source_dir2}/${save_data_name_221}, ${source_dir2}/${save_data_name_222}, ${source_dir2}/${save_data_name_223}, ${source_dir2}/${save_data_name_224}, ${source_dir2}/${save_data_name_225}, ${source_dir2}/${save_data_name_226}, ${source_dir2}/${save_data_name_227}, ${source_dir2}/${save_data_name_228}, ${source_dir2}/${save_data_name_229}, \
#         ${source_dir2}/${save_data_name_230}, ${source_dir2}/${save_data_name_231}, ${source_dir2}/${save_data_name_232}, ${source_dir2}/${save_data_name_233}, ${source_dir2}/${save_data_name_234}, ${source_dir2}/${save_data_name_235}, ${source_dir2}/${save_data_name_236}, ${source_dir2}/${save_data_name_237}, ${source_dir2}/${save_data_name_238}, ${source_dir2}/${save_data_name_239}, \
#         ${source_dir2}/${save_data_name_240}, ${source_dir2}/${save_data_name_241}, ${source_dir2}/${save_data_name_242}, ${source_dir2}/${save_data_name_243}, ${source_dir2}/${save_data_name_244}, ${source_dir2}/${save_data_name_245}, ${source_dir2}/${save_data_name_246}, ${source_dir2}/${save_data_name_247}, ${source_dir2}/${save_data_name_248}, ${source_dir2}/${save_data_name_249}, \
#         ${source_dir2}/${save_data_name_250}, ${source_dir2}/${save_data_name_251}, ${source_dir2}/${save_data_name_252}, ${source_dir2}/${save_data_name_253}, ${source_dir2}/${save_data_name_254}, ${source_dir2}/${save_data_name_255}, ${source_dir2}/${save_data_name_256}, ${source_dir2}/${save_data_name_257}, ${source_dir2}/${save_data_name_258}, ${source_dir2}/${save_data_name_259}, \
#         ${source_dir2}/${save_data_name_260}, ${source_dir2}/${save_data_name_261}, ${source_dir2}/${save_data_name_262}, ${source_dir2}/${save_data_name_263}, ${source_dir2}/${save_data_name_264}, ${source_dir2}/${save_data_name_265}, ${source_dir2}/${save_data_name_266}, ${source_dir2}/${save_data_name_267}, ${source_dir2}/${save_data_name_268}, ${source_dir2}/${save_data_name_269}, \
#         ${source_dir2}/${save_data_name_270}, ${source_dir2}/${save_data_name_271}, ${source_dir2}/${save_data_name_272}, ${source_dir2}/${save_data_name_273}, ${source_dir2}/${save_data_name_274}, ${source_dir2}/${save_data_name_275}, ${source_dir2}/${save_data_name_276}, ${source_dir2}/${save_data_name_277}, ${source_dir2}/${save_data_name_278}, ${source_dir2}/${save_data_name_279}, \
#         ${source_dir2}/${save_data_name_280}, ${source_dir2}/${save_data_name_281}, ${source_dir2}/${save_data_name_282}, ${source_dir2}/${save_data_name_283}, ${source_dir2}/${save_data_name_284}, ${source_dir2}/${save_data_name_285}, ${source_dir2}/${save_data_name_286}, \
#         ${source_dir2}/${save_data_name_287}, ${source_dir2}/${save_data_name_288}, ${source_dir2}/${save_data_name_289}, ${source_dir2}/${save_data_name_290}, ${source_dir2}/${save_data_name_291}, ${source_dir2}/${save_data_name_292}, ${source_dir2}/${save_data_name_293}, ${source_dir2}/${save_data_name_294}, ${source_dir2}/${save_data_name_295}, ${source_dir2}/${save_data_name_296}, ${source_dir2}/${save_data_name_297}, ${source_dir2}/${save_data_name_298}, ${source_dir2}/${save_data_name_299}, ${source_dir2}/${save_data_name_300}, ${source_dir2}/${save_data_name_301}, ${source_dir2}/${save_data_name_302}, ${source_dir2}/${save_data_name_303}, ${source_dir2}/${save_data_name_304}, ${source_dir2}/${save_data_name_305}, ${source_dir2}/${save_data_name_306}, ${source_dir2}/${save_data_name_307}, ${source_dir2}/${save_data_name_308}, ${source_dir2}/${save_data_name_309}, ${source_dir2}/${save_data_name_310}, ${source_dir2}/${save_data_name_311}, ${source_dir2}/${save_data_name_312}, ${source_dir2}/${save_data_name_313}, ${source_dir2}/${save_data_name_314}, ${source_dir2}/${save_data_name_315}, ${source_dir2}/${save_data_name_316}, ${source_dir2}/${save_data_name_317}, ${source_dir2}/${save_data_name_318}, ${source_dir2}/${save_data_name_319}, ${source_dir2}/${save_data_name_320}, ${source_dir2}/${save_data_name_321}, ${source_dir2}/${save_data_name_322}, ${source_dir2}/${save_data_name_323}, ${source_dir2}/${save_data_name_324}, ${source_dir2}/${save_data_name_325}, ${source_dir2}/${save_data_name_326}, ${source_dir2}/${save_data_name_327}, ${source_dir2}/${save_data_name_328}, ${source_dir2}/${save_data_name_329}, ${source_dir2}/${save_data_name_330}, ${source_dir2}/${save_data_name_331}, ${source_dir2}/${save_data_name_332}, ${source_dir2}/${save_data_name_333}, ${source_dir2}/${save_data_name_334}, ${source_dir2}/${save_data_name_335}, ${source_dir2}/${save_data_name_336}, ${source_dir2}/${save_data_name_337}, ${source_dir2}/${save_data_name_338}, ${source_dir2}/${save_data_name_339}, ${source_dir2}/${save_data_name_340}, ${source_dir2}/${save_data_name_341}, ${source_dir2}/${save_data_name_342}, ${source_dir2}/${save_data_name_343}, ${source_dir2}/${save_data_name_344}, ${source_dir2}/${save_data_name_345}, ${source_dir2}/${save_data_name_346}, ${source_dir2}/${save_data_name_347}, ${source_dir2}/${save_data_name_348}, ${source_dir2}/${save_data_name_349}, ${source_dir2}/${save_data_name_350}, ${source_dir2}/${save_data_name_351}, ${source_dir2}/${save_data_name_352}, ${source_dir2}/${save_data_name_353}, ${source_dir2}/${save_data_name_354}, ${source_dir2}/${save_data_name_355}, ${source_dir2}/${save_data_name_356}, ${source_dir2}/${save_data_name_357}, ${source_dir2}/${save_data_name_358}, ${source_dir2}/${save_data_name_359}, ${source_dir2}/${save_data_name_360}, ${source_dir2}/${save_data_name_361}, ${source_dir2}/${save_data_name_362}, ${source_dir2}/${save_data_name_363}, ${source_dir2}/${save_data_name_364}, ${source_dir2}/${save_data_name_365}, ${source_dir2}/${save_data_name_366}, ${source_dir2}/${save_data_name_367}, ${source_dir2}/${save_data_name_368}, ${source_dir2}/${save_data_name_369}, ${source_dir2}/${save_data_name_370}, ${source_dir2}/${save_data_name_371}, ${source_dir2}/${save_data_name_372}, ${source_dir2}/${save_data_name_373}, ${source_dir2}/${save_data_name_374}, ${source_dir2}/${save_data_name_375}, ${source_dir2}/${save_data_name_376}, ${source_dir2}/${save_data_name_377}, ${source_dir2}/${save_data_name_378}, ${source_dir2}/${save_data_name_379}, ${source_dir2}/${save_data_name_380}, ${source_dir2}/${save_data_name_381}, ${source_dir2}/${save_data_name_382}, ${source_dir2}/${save_data_name_383}, ${source_dir2}/${save_data_name_384}, ${source_dir2}/${save_data_name_385}, ${source_dir2}/${save_data_name_386}, ${source_dir2}/${save_data_name_387}, ${source_dir2}/${save_data_name_388}, ${source_dir2}/${save_data_name_389}, ${source_dir2}/${save_data_name_390}, ${source_dir2}/${save_data_name_391}, ${source_dir2}/${save_data_name_392}, ${source_dir2}/${save_data_name_393}, ${source_dir2}/${save_data_name_394}, ${source_dir2}/${save_data_name_395}, ${source_dir2}/${save_data_name_396}, ${source_dir2}/${save_data_name_397}, ${source_dir2}/${save_data_name_398}, ${source_dir2}/${save_data_name_399}, ${source_dir2}/${save_data_name_400}, ${source_dir2}/${save_data_name_401}, ${source_dir2}/${save_data_name_402}, ${source_dir2}/${save_data_name_403}, ${source_dir2}/${save_data_name_404}, ${source_dir2}/${save_data_name_405}, ${source_dir2}/${save_data_name_406}, ${source_dir2}/${save_data_name_407}, ${source_dir2}/${save_data_name_408}, ${source_dir2}/${save_data_name_409}, ${source_dir2}/${save_data_name_410}, ${source_dir2}/${save_data_name_411}, ${source_dir2}/${save_data_name_412}, ${source_dir2}/${save_data_name_413}, ${source_dir2}/${save_data_name_414}, ${source_dir2}/${save_data_name_415}, ${source_dir2}/${save_data_name_416}, ${source_dir2}/${save_data_name_417}, ${source_dir2}/${save_data_name_418}, ${source_dir2}/${save_data_name_419}, ${source_dir2}/${save_data_name_420}, ${source_dir2}/${save_data_name_421}, ${source_dir2}/${save_data_name_422}, ${source_dir2}/${save_data_name_423}, ${source_dir2}/${save_data_name_424}, ${source_dir2}/${save_data_name_425}, ${source_dir2}/${save_data_name_426}, ${source_dir2}/${save_data_name_427}, ${source_dir2}/${save_data_name_428}, ${source_dir2}/${save_data_name_429}, ${source_dir2}/${save_data_name_430}, ${source_dir2}/${save_data_name_431}, ${source_dir2}/${save_data_name_432}, ${source_dir2}/${save_data_name_433}, ${source_dir2}/${save_data_name_434}, ${source_dir2}/${save_data_name_435}, ${source_dir2}/${save_data_name_436}, ${source_dir2}/${save_data_name_437}, ${source_dir2}/${save_data_name_438}, ${source_dir2}/${save_data_name_439}, ${source_dir2}/${save_data_name_440}, ${source_dir2}/${save_data_name_441}, ${source_dir2}/${save_data_name_442}, ${source_dir2}/${save_data_name_443}, ${source_dir2}/${save_data_name_444}, ${source_dir2}/${save_data_name_445}, ${source_dir2}/${save_data_name_446}, ${source_dir2}/${save_data_name_447}, ${source_dir2}/${save_data_name_448}, ${source_dir2}/${save_data_name_449}, ${source_dir2}/${save_data_name_450}, ${source_dir2}/${save_data_name_451}, ${source_dir2}/${save_data_name_452}, ${source_dir2}/${save_data_name_453}, ${source_dir2}/${save_data_name_454}, ${source_dir2}/${save_data_name_455}, ${source_dir2}/${save_data_name_456}, ${source_dir2}/${save_data_name_457}, ${source_dir2}/${save_data_name_458}, ${source_dir2}/${save_data_name_459}, ${source_dir2}/${save_data_name_460}, ${source_dir2}/${save_data_name_461}, ${source_dir2}/${save_data_name_462}, \
#         ${source_dir2}/${save_data_name_463},${source_dir2}/${save_data_name_464},${source_dir2}/${save_data_name_465},${source_dir2}/${save_data_name_466},${source_dir2}/${save_data_name_467},${source_dir2}/${save_data_name_468},${source_dir2}/${save_data_name_469},${source_dir2}/${save_data_name_470},${source_dir2}/${save_data_name_471},${source_dir2}/${save_data_name_472},${source_dir2}/${save_data_name_473},${source_dir2}/${save_data_name_474},${source_dir2}/${save_data_name_475},${source_dir2}/${save_data_name_476},${source_dir2}/${save_data_name_477},${source_dir2}/${save_data_name_478},${source_dir2}/${save_data_name_479},${source_dir2}/${save_data_name_480},${source_dir2}/${save_data_name_481},${source_dir2}/${save_data_name_482},${source_dir2}/${save_data_name_483},${source_dir2}/${save_data_name_484},${source_dir2}/${save_data_name_485},${source_dir2}/${save_data_name_486},${source_dir2}/${save_data_name_487},${source_dir2}/${save_data_name_488},${source_dir2}/${save_data_name_489},${source_dir2}/${save_data_name_490},${source_dir2}/${save_data_name_491},${source_dir2}/${save_data_name_492},${source_dir2}/${save_data_name_493},${source_dir2}/${save_data_name_494},${source_dir2}/${save_data_name_495},${source_dir2}/${save_data_name_496},${source_dir2}/${save_data_name_497},${source_dir2}/${save_data_name_498},${source_dir2}/${save_data_name_499},${source_dir2}/${save_data_name_500},${source_dir2}/${save_data_name_501},${source_dir2}/${save_data_name_502},${source_dir2}/${save_data_name_503},${source_dir2}/${save_data_name_504},${source_dir2}/${save_data_name_505},${source_dir2}/${save_data_name_506},${source_dir2}/${save_data_name_507},${source_dir2}/${save_data_name_508},${source_dir2}/${save_data_name_509},${source_dir2}/${save_data_name_510},${source_dir2}/${save_data_name_511},${source_dir2}/${save_data_name_512},${source_dir2}/${save_data_name_513},${source_dir2}/${save_data_name_514},${source_dir2}/${save_data_name_515},${source_dir2}/${save_data_name_516},${source_dir2}/${save_data_name_517},${source_dir2}/${save_data_name_518},${source_dir2}/${save_data_name_519},${source_dir2}/${save_data_name_520},${source_dir2}/${save_data_name_521},${source_dir2}/${save_data_name_522},${source_dir2}/${save_data_name_523},${source_dir2}/${save_data_name_524},${source_dir2}/${save_data_name_525},${source_dir2}/${save_data_name_526},${source_dir2}/${save_data_name_527},${source_dir2}/${save_data_name_528},${source_dir2}/${save_data_name_529},${source_dir2}/${save_data_name_530},${source_dir2}/${save_data_name_531},${source_dir2}/${save_data_name_532},${source_dir2}/${save_data_name_533},${source_dir2}/${save_data_name_534},${source_dir2}/${save_data_name_535},${source_dir2}/${save_data_name_536},${source_dir2}/${save_data_name_537},${source_dir2}/${save_data_name_538},${source_dir2}/${save_data_name_539},${source_dir2}/${save_data_name_540},${source_dir2}/${save_data_name_541},${source_dir2}/${save_data_name_542},${source_dir2}/${save_data_name_543},${source_dir2}/${save_data_name_544},${source_dir2}/${save_data_name_545},${source_dir2}/${save_data_name_546},${source_dir2}/${save_data_name_547},${source_dir2}/${save_data_name_548},${source_dir2}/${save_data_name_549},${source_dir2}/${save_data_name_550},${source_dir2}/${save_data_name_551},${source_dir2}/${save_data_name_552},${source_dir2}/${save_data_name_553},${source_dir2}/${save_data_name_554},${source_dir2}/${save_data_name_555},${source_dir2}/${save_data_name_556},${source_dir2}/${save_data_name_557},${source_dir2}/${save_data_name_558},${source_dir2}/${save_data_name_559},${source_dir2}/${save_data_name_560},${source_dir2}/${save_data_name_561},${source_dir2}/${save_data_name_562},${source_dir2}/${save_data_name_563},${source_dir2}/${save_data_name_564},${source_dir2}/${save_data_name_565},${source_dir2}/${save_data_name_566},${source_dir2}/${save_data_name_567},${source_dir2}/${save_data_name_568},${source_dir2}/${save_data_name_569}\





    
