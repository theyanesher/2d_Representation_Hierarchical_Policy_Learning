#!/usr/bin/env bash

# If $1 is unset or empty, default to "square_d2"
task="${1:-square_d2}"

# Build the Hydra config name from the task
config_name="articubot_${task}_train"

python mimicgen_train.py --config-name "${config_name}" \
    n_demo=1000 \
    training.add_noise_to_goal=true