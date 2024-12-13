### 50 objects
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0723-50-obj-pred-goal-gripper-mlp-self-attn-backbone-transformer-diffusion/2024.07.23/10.04.29_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-30.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_50_obj


# ### 200 objects
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/pratik/run_sample_basic_experiments/Robogen_Pratik_Branch/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/test-200-obj-pred-goal-gripper-pointnet-backbone-unet-diffusion-epsilon-attn_head-3_CondTrans_Non_Mean_cent/2024.09.24/20.06.46_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-26.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_200_obj


### 500 objects (200 + camera randomizations)
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_prediction.py \
#     --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1115-high-level-transformer-diffusion-500-obj-pred-goal-gripper/2024.11.16/00.44.40_train_dp3_robogen_open_door/ \
#     --high_level_ckpt_name epoch-58.ckpt \
#     --low_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door \
#     --low_level_ckpt_name epoch-96.ckpt  \
#     --eval_exp_name paper_eval_high_level_transformer_diffusion_500_obj



