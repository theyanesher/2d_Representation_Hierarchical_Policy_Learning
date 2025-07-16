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

demo_name=165-obj
observation_mode=act3d_goal_displacement_gripper_to_object
pointcloud_num=4500

### pass in args of save_data_name and exp_folder
folder_name=${1}
exp_folder=${2}
exp_name=0630_weighted_full_rollout

python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_robogen_with_goal_PointNet_save_state.py \
    --low_level_exp_dir  ckpt/low_level_165-obj_scratch/ \
    --low_level_ckpt_name epoch-96.ckpt \
    --high_level_ckpt_name /data/chenyuah/RoboGen-sim2real/ckpt/0613_weighted_full/model_80.pth \
    --pointnet_class PointNet2_super \
    --model_invariant true \
    --output_obj_pcd_only \
    --eval_exp_name ${exp_name} \
    --exp_dir ${1}${2} \

python 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/gen_from_failure.py \
    --data_dir /mnt/RoboGen_sim2real/data/${exp_name}/rollout/ \
    --output_dir /mnt/RoboGen_sim2real/data/${exp_name}/gen/ \
    --exp_dir ${1}${2} 

python 3d_diffusion_policy/extract_data_from_states_2.py \
    --folder_name ${folder_name} \
    --save_path "/mnt/RoboGen_sim2real/data/${exp_name}/dp3_demo/${exp_folder}"\
    --exp_name "${demo_name}" \
    --pointcloud_num "${pointcloud_num}" \
    --num_experiment 1000 \
    --observation_mode "${observation_mode}" \
    --parallel 0 \
    --extract_name "${exp_folder}" \
    --noise_real_world_pcd 1 \
    --randomize_camera 1 \
