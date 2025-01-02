
## 500 objects (200 + camera randomizations), eval with real-world camera placement
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1115-high-level-transformer-diffusion-500-obj-pred-goal-gripper/2024.11.16/00.44.40_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-58.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj_real_world_camera \
#     --real_world_camera 1

### 500 objects (200 + camera randomizations), eval with camera randomization
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-300-obj-1211/2024.12.13/00.40.14_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-56.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj_correct_real_world_camera_epoch_56 \
#     --real_world_camera 1

# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-300-obj-1211/2024.12.13/00.40.14_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-56.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj_correct_real_world_camera_epoch_56_trial_2 \
#     --real_world_camera 1

# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-300-obj-1211/2024.12.13/00.40.14_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-56.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj_correct_real_world_camera_epoch_56_trial_3 \
#     --real_world_camera 1


### 200 objects + camera randomizations
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-200-obj-camera-rand-1211/2024.12.13/02.02.40_train_dp3_robogen_open_door \
#     --high_level_ckpt_name epoch-32.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_200_obj_train_random_camera_real_camera_trial_2 \
#     --real_world_camera 1

# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-200-obj-camera-rand-1211/2024.12.13/02.02.40_train_dp3_robogen_open_door \
#     --high_level_ckpt_name epoch-32.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_200_obj_train_random_camera_real_camera_trial_3 \
#     --real_world_camera 1


### 50 objects + camera randomizations
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-50-obj-camera-random-1217/2024.12.17/00.51.54_train_dp3_robogen_open_door \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_50_obj_train_random_camera_real_world_camera \
#     --real_world_camera 1

# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-50-obj-camera-random-1217/2024.12.17/00.51.54_train_dp3_robogen_open_door \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_50_obj_train_random_camera_real_world_camera_trial_2 \
#     --real_world_camera 1

# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-50-obj-camera-random-1217/2024.12.17/00.51.54_train_dp3_robogen_open_door \
#     --high_level_ckpt_name epoch-60.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_50_obj_train_random_camera_real_world_camera_trial_3 \
#     --real_world_camera 1

### 100 obj + random cameras
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-100-obj-camera-random-1212/2024.12.13/01.31.47_train_dp3_robogen_open_door \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_transformer_diffusion_100_obj_train_random_camera_real_world_camera \
    --real_world_camera 1


python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-100-obj-camera-random-1212/2024.12.13/01.31.47_train_dp3_robogen_open_door \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_transformer_diffusion_100_obj_train_random_camera_real_world_camera_trial_2 \
    --real_world_camera 1


python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-100-obj-camera-random-1212/2024.12.13/01.31.47_train_dp3_robogen_open_door \
    --high_level_ckpt_name epoch-60.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_transformer_diffusion_100_obj_train_random_camera_real_world_camera_trial_3 \
    --real_world_camera 1

