
task=square_d2
exp_name="articubot_${task}"
test_start_seed=100000

high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-23-15-53_square_d2_abs

# goal conditioned
# low_level_dir=data/outputs/2025.06.25/01.47.02_diff_c_square_d2
# python eval.py --config-name articubot_gc_diffusion_unet \
#     task=articubot_gc_diffpo \
#     task_name=$task \
#     low_level_dir=$low_level_dir \
#     high_level_dir=$high_level_dir \
#     exp_name=$exp_name \
#     test_start_seed=$test_start_seed \

# flow conditioned
low_level_dir=data/outputs/2025.06.27/15.35.07_diff_c_square_d2
python eval.py --config-name articubot_gc_diffusion_unet \
    task=articubot_flow_diffpo \
    task_name=$task \
    low_level_dir=$low_level_dir \
    high_level_dir=$high_level_dir \
    exp_name=$exp_name \
    test_start_seed=$test_start_seed \
    policy.conditioning_type=3d_flow_world_frame
