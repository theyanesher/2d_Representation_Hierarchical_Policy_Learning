cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_heuristic_goal_switching \
#     --use_high_level 0 \
#     --heuristic_goal_switching 1

python eval_robogen_with_goal_prediction.py \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_heuristic_goal_switching_trial_2 \
    --use_high_level 0 \
    --heuristic_goal_switching 1

python eval_robogen_with_goal_prediction.py \
    --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
    --low_level_ckpt_name epoch-96.ckpt  \
    --eval_exp_name paper_eval_heuristic_goal_switching_trial_3 \
    --use_high_level 0 \
    --heuristic_goal_switching 1