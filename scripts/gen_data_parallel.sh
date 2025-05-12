#!/bin/bash

cd /mnt/RoboGen_sim2real
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/rl_games:$PYTHONPATH
export PYTHONPATH=${PWD}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy:$PYTHONPATH
export PROJECT_DIR=${PWD}
export YUFEI_OPENAI_API_KEY=xxx
source prepare.sh

demo_name=seuss_gen_random
observation_mode=act3d_goal_displacement_gripper_to_object
pointcloud_num=4500

### pass in args of save_data_name and exp_folder
folder_name=${1}
exp_folder=${2}
# save_data_name=${2}

# python manipulation/gen_demo.py --root_dir "${folder_name}" --exp_name "${demo_name}" --extract_name "${exp_folder}" 
# python manipulation/gen_demo/gen_demo.py --root_dir "${folder_name}" --exp_name "${demo_name}" --extract_name "${exp_folder}" --use_augmented_handle 1 --num_augmented_handle 5 --max_try_times 15
# python manipulation/gen_demo/gen_demo.py --root_dir "${folder_name}" --exp_name "${demo_name}" --extract_name "${exp_folder}" --use_augmented_handle 1 --num_augmented_handle 5 --max_try_times 5
python 3d_diffusion_policy/extract_data_from_states_2.py \
    --folder_name ${folder_name} \
    --save_path "data/dp3_demo/${demo_name}/${exp_folder}"\
    --exp_name "${demo_name}" \
    --pointcloud_num "${pointcloud_num}" \
    --num_experiment 1000 \
    --observation_mode "${observation_mode}" \
    --parallel 0 \
    --extract_name "${exp_folder}" \




