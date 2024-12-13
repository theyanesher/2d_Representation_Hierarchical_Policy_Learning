### 200 objects
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_prediction.py \
    --low_level_exp_dir /project_data/held/mnakuraf/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-dp3-200-training-objs-1211/2024.12.12/01.44.03_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-48.ckpt  \
    --eval_exp_name paper_eval_dp3_200_object \
    --use_high_level 0