trial_number=$1
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_prediction.py \
    --low_level_exp_dir /home/mino/Software/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-no-hierrachical-100-training-objs-1230/2024.12.30/17.41.32_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-60.ckpt  \
    --eval_exp_name paper_eval_no_hierachry_rotary_unet_diffusion_100_object_epoch60_trial${trial_number} \
    --use_high_level 0