### 200 objects
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_prediction.py \
    --high_level_exp_dir /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1115-high-level-transformer-diffusion-500-obj-pred-goal-gripper/2024.11.16/00.44.40_train_dp3_robogen_open_door/ \
    --high_level_ckpt_name epoch-58.ckpt \
    --low_level_exp_dir /project_data/held/mnakuraf/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-hierarchical-low-level-transformer-diffusion-200-training-objs-1215/2024.12.15/15.30.47_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-60.ckpt  \
    --eval_exp_name paper_eval_low_level_rotary_attention_unet_diffusion \
    --randomize_camera 1


