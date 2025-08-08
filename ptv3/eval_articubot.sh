#!/bin/bash

list=(40147 44817 44962 45132 45219 45243 45332 45378 45384 45463)
# list=(40147 44817)

gpu_count=8
i=0

for id in "${list[@]}"; do
    gpu_id=$((i % gpu_count))

    echo "Launching ID=$id on GPU=$gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_gmm_cleaned_old_data_format.py \
        --low_level_exp_dir /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/low-level-ckpt/ \
        --low_level_ckpt_name low-level.ckpt \
        --high_level_ckpt_name /project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/ptv3/data/articubot_cgn_ptv3-first-try/2025.08.02/18.10.23/model_122500.pth \
        --eval_exp_name eval_ptv3_articubot_50_cgn \
        --exp_dir data/diverse_objects/open_the_door_${id}/task_open_the_door_of_the_storagefurniture_by_its_handle \
        --model_type ptv3 \
        > data/logs/eval_${id}.log 2>&1 &

    ((i++))
done

wait  # Wait for all background jobs to finish
echo "All jobs finished."
