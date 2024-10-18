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

# demo_name=0725-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first
demo_name=${3}
observation_mode=act3d_goal_displacement_gripper_to_object
pointcloud_num=4500

### pass in args of save_data_name and exp_folder
# folder_name=data/diverse_objects_2/
folder_name=${4}
exp_folder=${1}
save_data_name=${2}

# python manipulation/gen_demo/gen_demo.py --root_dir "${folder_name}" --exp_name "${demo_name}" --extract_name "${exp_folder}" 
python 3d_diffusion_policy/extract_data_from_states_2.py \
    --folder_name ${folder_name} \
    --object_name storagefurniture \
    --save_path "data/dp3_demo/${save_data_name}" --exp_name "${demo_name}" \
    --pointcloud_num "${pointcloud_num}" \
    --num_experiment 1000 \
    --observation_mode "${observation_mode}" \
    --parallel 0 \
    --exp_folder "${exp_folder}" \





