#!/bin/bash

cd /mnt/RoboGen_sim2real
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/rl_games:$PYTHONPATH
export PYTHONPATH=${PWD}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy:$PYTHONPATH
export PROJECT_DIR=${PWD}
source prepare.sh

python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_save_state.py \
    --low_level_exp_dir  ckpt/low_level_165-obj_scratch/ \
    --low_level_ckpt_name epoch-96.ckpt \
    --high_level_ckpt_name /data/chenyuah/RoboGen-sim2real/ckpt/high_level_165-obj_scratch/model_60.pth \
    --pointnet_class PointNet2_super \
    --model_invariant true \
    --output_obj_pcd_only \
    --eval_exp_name 0529_test_rollout \
    --exp_dir ${1}${2} \
