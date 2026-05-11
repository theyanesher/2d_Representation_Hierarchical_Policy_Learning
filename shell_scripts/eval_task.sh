#!/usr/bin/env bash

# If $1 is unset or empty, default to "square_d2"
# task="${1:-square_d2}"

# task=stack_d1
# low_level_dir="data/outputs/2025.06.05/22.10.21_train_dp3_stack_d1"
# high_level_dir="third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-10_use_all_data_stack_D1_abs-obj_stack_D1_abs"

# task="coffee_d2"
# low_level_dir=data/outputs/2025.06.05/21.37.45_train_dp3_coffee_d2 # 86%
# high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-06_use_all_data_coffee_D2_abs-obj_coffee_D2_abs

# task=square_d2
# low_level_dir=data/outputs/2025.05.21/12.46.41_train_dp3_square_d2
# high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-05-20_use_all_data_square_D2_abs-obj_square_D2_abs/

task=stack_three_d1
# high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-22-05-54_stack_three_d1_abs # 84%
high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-24-15-12_stack_three_d1_abs
low_level_dir=data/outputs/2025.06.23/15.13.19_train_dp3_stack_three_d1
/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/
/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/test_PointNet2/exps/2026-03-15finetune_mimicgen_mug_cleanup_d1_one_hot/model_97501.pth
# task=mug_cleanup_d1
# high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-22-13-48_mug_cleanup_d1_abs
# low_level_dir=data/outputs/2025.06.24/10.51.16_train_dp3_mug_cleanup_d1

# task=square_d2
# high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-23-15-53_square_d2_abs
# low_level_dir=data/outputs/2025.06.23/15.55.58_train_dp3_square_d2

# task=threading_d2
# high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-21-16-06_threading_d2_abs
# low_level_dir=data/outputs/2025.06.23/22.00.50_train_dp3_threading_d2


exp_name="articubot_${task}"
test_start_seed=100000

python eval.py --config-name articubot_mimic_eval \
    task_name=$task \
    low_level_dir=$low_level_dir \
    high_level_dir=$high_level_dir \
    exp_name=$exp_name \
    test_start_seed=$test_start_seed    # policy.pointnet_type=pointne