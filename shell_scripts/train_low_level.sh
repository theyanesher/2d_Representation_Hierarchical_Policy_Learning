#!/usr/bin/env bash

# If $1 is unset or empty, default to "square_d2"
task="${1:-square_d2}"
exp_name="articubot_${task}"

echo "training ${task}"

python mimicgen_train.py \
    --config-name articubot_mimic_train \
    task_name="${task}" \
    exp_name="${exp_name}"