#!/usr/bin/env bash

# If $1 is unset or empty, default to "square_d2"
task="${1:-square_d2}"
exp_name="articubot_${task}"

echo "training ${task}"

torchrun --standalone --nproc_per_node=4 mimicgen_train_ddp.py \
    --config-name articubot_mimic_ddp \
    task_name="${task}" \
    exp_name="${exp_name}" \
    policy.pointnet_type=pn2act3d \
    dataloader.batch_size=25 \
    val_dataloader.batch_size=25 \
    high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-30-15-55_square_d2_abs