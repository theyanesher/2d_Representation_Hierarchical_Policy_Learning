#!/usr/bin/env bash

# Define your list of tasks…
tasks=(
  # stack_three_d1
  # square_d2
  # three_piece_assembly_d2
  # nut_assembly_d0
  # threading_d2
  # hammer_cleanup_d1
  mug_cleanup_d1
  # coffee_d2
  # kitchen_d1
  # pick_place_d0
  # coffee_preparation_d1
  # stack_d1
)

# Loop over indices of the arrays
for i in "${!tasks[@]}"; do
  task="${tasks[i]}"
  exp_name="articubot_${task}"

  echo "training ${task}"

  python mimicgen_train.py \
    --config-name articubot_mimic_train \
    task_name="${task}" \
    exp_name="${exp_name}"
done

# python mimicgen_train.py --config-name articubot_mimic_train task_name=square_d2 exp_name=articubot_square_d2 policy.pointnet_type=pointnet2 dataloader.batch_size=20 val_dataloader.batch_size=20

