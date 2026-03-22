#!/bin/bash

# Define the range and step
START=5
END=95
STEP=5
MAX_PARALLEL=5

# Generate the sequence of epochs and run them in parallel
seq $START $STEP $END | xargs -I {} -P $MAX_PARALLEL sh -c "
    echo 'Starting evaluation for epoch_{}...'
    pixi run python diffusion_policy/eval_diffpo_single_object.py \
      --low_level_exp_dir outputs/2026.01.14/18.20.02_train_diffusion_unet_hybrid_articubot_image \
      --low_level_ckpt_name epoch_{}.ckpt \
      --eval_exp_name diffpo \
      --folder_name data/rgb_eval
    echo 'Finished epoch_{}.'
"