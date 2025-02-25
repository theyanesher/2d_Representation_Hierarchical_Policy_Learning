### 500 objects (200 + camera randomizations)
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1115-high-level-transformer-diffusion-500-obj-pred-goal-gripper/2024.11.16/00.44.40_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-58.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj

### 500 objects (200 + camera randomizations) correct one
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-300-obj-1211/2024.12.13/00.40.14_train_dp3_robogen_open_door/ \
    --high_level_ckpt_name epoch-56.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj_correct_epoch_56

python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-300-obj-1211/2024.12.13/00.40.14_train_dp3_robogen_open_door/ \
    --high_level_ckpt_name epoch-56.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj_correct_epoch_56_trial_2

python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-transformer-diffusion-high-level-300-obj-1211/2024.12.13/00.40.14_train_dp3_robogen_open_door/ \
    --high_level_ckpt_name epoch-56.ckpt \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj_correct_epoch_56_trial_3



