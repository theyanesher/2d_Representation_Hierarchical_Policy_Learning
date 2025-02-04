### 300 obj randomized cam
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-500-obj-camera-random-0105/2025.01.05/17.50.21_train_dp3_robogen_open_door \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_unet_diffusion_300-train-random-camera \

python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-500-obj-camera-random-0105/2025.01.05/17.50.21_train_dp3_robogen_open_door \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_unet_diffusion_300-train-random-camera_trial_2 \
    
python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-unet-diffusion-high-level-500-obj-camera-random-0105/2025.01.05/17.50.21_train_dp3_robogen_open_door \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_unet_diffusion_300-train-random-camera_trial_3 \


