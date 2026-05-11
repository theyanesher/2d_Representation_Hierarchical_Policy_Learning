#!/usr/bin/env bash

# If $1 is unset or empty, default to "square_d2"
task="${1:-square_d2}"
exp_name="articubot_${task}"

echo "training ${task}"

python mimicgen_train.py \
    --config-name articubot_transformer_mimic_train \
    task_name="${task}" \
    exp_name="${exp_name}" \
    high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-30-15-55_square_d2_abs \
    n_demo=1000