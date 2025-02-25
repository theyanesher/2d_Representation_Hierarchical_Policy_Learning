### 50 objects train with random camera
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-50-obj-camera-random-0105/2025.01.05/17.51.33_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_50-train-random-camera_random_camera \
#     --randomize_camera 1

# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-50-obj-camera-random-0105/2025.01.05/17.51.33_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_50-train-random-camera_random_camera_trial_2 \
#     --randomize_camera 1
    
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-50-obj-camera-random-0105/2025.01.05/17.51.33_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_50-train-random-camera_random_camera_trial_3 \
#     --randomize_camera 1


# ### 100 obj randomized cam
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-100-obj-camera-random-0107/2025.01.07/16.09.48_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_100-train-random-camera_random_camera \
#     --randomize_camera 1

# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-100-obj-camera-random-0107/2025.01.07/16.09.48_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_100-train-random-camera_random_camera_trial_2 \
#     --randomize_camera 1
    
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-100-obj-camera-random-0107/2025.01.07/16.09.48_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_100-train-random-camera_random_camera_trial_3 \
#     --randomize_camera 1

### 200 obj random data
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-200-obj-camera-random-0111/2025.01.11/18.29.52_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_200-train-random-camera_random_camera \
#     --randomize_camera 1


# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-200-obj-camera-random-0111/2025.01.11/18.29.52_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_200-train-random-camera_random_camera_trial_2 \
#     --randomize_camera 1

    
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-200-obj-camera-random-0111/2025.01.11/18.29.52_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_unet_diffusion_200-train-random-camera_random_camera_trial_3 \
#     --randomize_camera 1

### 10 obj radom cam
### train on 10 random cameras
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-10-obj-camera-random-0123/2025.01.23/01.39.05_train_dp3_robogen_open_door/ \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_unet_diffusion_10-train-random-camera-random-cam \
    --randomize_camera 1

python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-10-obj-camera-random-0123/2025.01.23/01.39.05_train_dp3_robogen_open_door/ \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_unet_diffusion_10-train-random-camera-random-cam_trial_2 \
    --randomize_camera 1
    
python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-10-obj-camera-random-0123/2025.01.23/01.39.05_train_dp3_robogen_open_door/ \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_unet_diffusion_10-train-random-camera-random-cam_trial_3 \
    --randomize_camera 1


