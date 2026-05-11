#!/usr/bin/env bash

task=square_d2
high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-23-15-53_square_d2_abs
low_level_dir=data/outputs/2025.06.23/15.55.58_train_dp3_square_d2

exp_name="articubot_${task}"
test_start_seed=100000

python eval.py --config-name articubot_mimic_eval \
    task='articubot_pc_abs_eval_seeded' \ 
    dataset_path=data/robomimic/datasets/square_d2/for_oracle/square_d2_150_abs.hdf5 \
    task_name=$task \
    low_level_dir=$low_level_dir \
    high_level_dir=$high_level_dir \
    exp_name=$exp_name \
    test_start_seed=$test_start_seed