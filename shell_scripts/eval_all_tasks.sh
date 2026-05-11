#!/usr/bin/env bash

# Define your list of tasks…
tasks=(
  # stack_d1
  # stack_three_d1
  # square_d2
  # threading_d2
  # coffee_d2
  # three_piece_assembly_d2
  # hammer_cleanup_d1
  # mug_cleanup_d1
  # kitchen_d1
  # nut_assembly_d0
  # pick_place_d0
  # coffee_preparation_d1
)

high_level_paths=(
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-10_use_all_data_stack_D1_abs-obj_stack_D1_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_06-22-05-54_stack_three_d1_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_06-18-00-00_square_d2_abs # instance norm
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_06-21-15-17_square_d2_abs # group norm
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_06-23-15-53_square_d2_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-05-20_use_all_data_square_D2_abs-obj_square_D2_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-03_use_all_data_threading_D2_abs-obj_threading_D2_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-06_use_all_data_coffee_D2_abs-obj_coffee_D2_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-05-23_use_all_data_three_piece_assembly_D2_abs-obj_three_piece_assembly_D2_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-08_use_all_data_hammer_cleanup_D1_abs-obj_hammer_cleanup_D1_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_06-16_mug_cleanup_d1_abs
  # kitchen
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_06-13_nut_assembly_d0_abs
  # third_party/robogen/test_PointNet2/exps/pointnet2_super_06-15_pick_place_d0_abs
  # coffee_preparation
)

low_level_dirs=(
  # data/outputs/2025.06.05/22.10.21_train_dp3_stack_d1
  # data/outputs/2025.06.23/15.13.19_train_dp3_stack_three_d1
  # data/outputs/2025.05.20/20.50.52_train_dp3_square_d2
  # data/outputs/2025.06.21/20.57.15_train_dp3_square_d2
  # data/outputs/2025.06.03/17.13.26_train_dp3_threading_d2
  # data/outputs/2025.06.05/21.37.45_train_dp3_coffee_d2
  # data/outputs/2025.06.02/21.08.37_train_dp3_three_piece_assembly_d2
  # data/outputs/2025.06.05/22.00.46_train_dp3_hammer_cleanup_d1
  # data/outputs/2025.06.13/01.45.23_train_dp3_mug_cleanup_d1
  # kitchen
  # data/outputs/2025.06.10/19.04.13_train_dp3_nut_assembly_d0
  # data/outputs/2025.06.09/16.13.28_train_dp3_pick_place_d0
  # data/outputs/2025.06.13/18.12.53_train_dp3_coffee_preparation_d1
)

test_start_seed=100000

# Loop over indices of the arrays
for i in "${!tasks[@]}"; do
  task="${tasks[i]}"
  high_level="${high_level_paths[i]}"
  low_level="${low_level_dirs[i]}"
  exp_name="articubot_${task}"

  echo "evaluating ${task}"

  python eval.py \
    --config-name articubot_mimic_eval \
    task_name="${task}" \
    exp_name="${exp_name}" \
    high_level_dir="${high_level}" \
    low_level_dir="${low_level}" \
    test_start_seed=$test_start_seed
done
