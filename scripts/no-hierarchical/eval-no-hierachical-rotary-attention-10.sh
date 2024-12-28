cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_prediction.py \
    --low_level_exp_dir /home/mino/Software/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-no-hierrachical-10-training-objs-1216/2024.12.17/01.17.25_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-60.ckpt  \
    --eval_exp_name paper_eval_no_hierachry_rotary_unet_diffusion_10_object_epoch60_trial3 \
    --use_high_level 0