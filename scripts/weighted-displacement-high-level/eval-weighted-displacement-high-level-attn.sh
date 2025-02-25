

### trained on 50 objects
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_PointNet.py \
    --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-96.ckpt \
    --high_level_ckpt_name  /project_data/held/yufeiw2/robogen_sim2real_ziyu_branch/RoboGen-sim2real/test_PointNet2/exps/pointnet2_super_add_attn_model_invariant_2025-01-08_use_75_episodes_50-obj/model_60.pth \
    --pointnet_class pointnet2_super_add_attn \
    --model_invariant true \
    --output_obj_pcd_only \
    --eval_exp_name  paper_eval_high_level_weighted-displacement_attn-50_48_obj \

python eval_robogen_with_goal_PointNet.py \
    --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-96.ckpt \
    --high_level_ckpt_name  /project_data/held/yufeiw2/robogen_sim2real_ziyu_branch/RoboGen-sim2real/test_PointNet2/exps/pointnet2_super_add_attn_model_invariant_2025-01-08_use_75_episodes_50-obj/model_60.pth \
    --pointnet_class pointnet2_super_add_attn \
    --model_invariant true \
    --output_obj_pcd_only \
    --eval_exp_name  paper_eval_high_level_weighted-displacement_attn-50_48_obj_trial_2 \

python eval_robogen_with_goal_PointNet.py \
    --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-96.ckpt \
    --high_level_ckpt_name  /project_data/held/yufeiw2/robogen_sim2real_ziyu_branch/RoboGen-sim2real/test_PointNet2/exps/pointnet2_super_add_attn_model_invariant_2025-01-08_use_75_episodes_50-obj/model_60.pth \
    --pointnet_class pointnet2_super_add_attn \
    --model_invariant true \
    --output_obj_pcd_only \
    --eval_exp_name  paper_eval_high_level_weighted-displacement_attn-50_48_obj_trial_3 \






