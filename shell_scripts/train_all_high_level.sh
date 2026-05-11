#!/usr/bin/env bash
set -euo pipefail

cd /project_data/held/mnakuraf/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2

# Define your list of tasks…
tasks=(
  # stack_three_d1
  # stack_d1
  threading_d2
  # coffee_d2
  # three_piece_assembly_d2
  # hammer_cleanup_d1
  # mug_cleanup_d1
  # kitchen_d1
  # nut_assembly_d0
  # pick_place_d0
  # square_d2
  # coffee_preparation_d1
)

for i in "${!tasks[@]}"; do
    task="${tasks[i]}"
    exp_name="articubot_${task}"

    torchrun --standalone --nproc_per_node=8 train_ddp_weighted_displacement.py \
        --batch_size 110 \
        --num_epochs 100 \
        --model_type pointnet2_super --model_invariant \
        --exp_path /project_data/held/mnakuraf/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2/exps \
        --num_train_objects ${task}_abs \
        --dataset_prefix /scratch/minon/${task}_abs/ \
        --exp_name _${task}_abs \
        --use_color \
        --use_all_data # \
        # --use_group_norm
done