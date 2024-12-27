### 200 objects
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy
python eval_robogen_with_goal_PointNet.py \
    --low_level_exp_dir /home/mino/Software/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/paper-hierarchical-low-level-transformer-diffusion-200-training-objs-1215/2024.12.15/15.30.47_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-60.ckpt \
    --high_level_ckpt_name  /home/mino/Software/RoboGen-sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-10-09_use_75_episodes_500-obj/model_57.pth \
    --pointnet_class PointNet2_super \
    --eval_exp_name paper_eval_low_level_rotary_attention_unet_diffusion_200 \
    --model_invariant true \
    --output_obj_pcd_only
