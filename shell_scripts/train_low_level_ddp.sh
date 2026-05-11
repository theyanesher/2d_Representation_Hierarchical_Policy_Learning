#!/usr/bin/env bash

# If $1 is unset or empty, default to "square_d2"
task="${1:-square_d2}"

# Build the Hydra config name from the task
config_name="articubot_${task}_ddp"

# Launch
torchrun --standalone --nproc_per_node=2 mimicgen_train_ddp.py \
    --config-name "${config_name}" \
    n_demo=1000