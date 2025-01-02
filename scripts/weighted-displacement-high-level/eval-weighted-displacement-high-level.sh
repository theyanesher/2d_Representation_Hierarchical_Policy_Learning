### trained on 500 objects
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-10-09_use_75_episodes_500-obj/model_57.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_500_obj_model_trial_2 \

# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-10-09_use_75_episodes_500-obj/model_57.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_500_obj_model_trial_3 \

    # default one
    # --high_level_ckpt_name /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-10-17_use_75_episodes_500-obj/model_57.pth \
    # model 2: epoch 60


### trained on 200 objects
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-11_use_75_episodes_200-obj_paper_1211/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_200_obj_trial_2 \

# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-11_use_75_episodes_200-obj_paper_1211/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_200_obj_trial_3 \


### trained on 300 objects
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-11_use_75_episodes_300-obj_paper_1211/model_57.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_300_obj_trial_2 \

# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-11_use_75_episodes_300-obj_paper_1211/model_57.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_300_obj_trial_3 \


### trained on 100 objects
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-15_use_75_episodes_100-obj_paper_1215/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_100_obj_trial_2 \

# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-15_use_75_episodes_100-obj_paper_1215/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_100_obj_trial_3 \

### trained on 100 objs with randomized cameras
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-13_use_75_episodes_camera_random_100_obj_high_level-obj_paper_1211/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_100_obj_train_random_camera_trial_2 \

# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-13_use_75_episodes_camera_random_100_obj_high_level-obj_paper_1211/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_100_obj_train_random_camera_trial_3 \

### trained on 50 objects with randomized cameras
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-15_use_75_episodes_camera_random_50_obj_high_level-obj_paper_1215/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_50_obj_train_random_camera_trial_2 \

# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-15_use_75_episodes_camera_random_50_obj_high_level-obj_paper_1215/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_50_obj_train_random_camera_trial_3 \

### trained on 200 objects with randomized cameras
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-13_use_75_episodes_camera_random_200_obj_high_level-obj_paper_1213/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_200_obj_train_random_camera_trial_2 \


# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-13_use_75_episodes_camera_random_200_obj_high_level-obj_paper_1213/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_200_obj_train_random_camera_trial_3 \

### trained on 50 objects
# cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-19_use_75_episodes_50-obj_paper_1219/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_50_obj \

# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-19_use_75_episodes_50-obj_paper_1219/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_50_obj_trial_2 \


# python eval_robogen_with_goal_PointNet.py \
#     --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
#     --low_level_ckpt_name epoch-96.ckpt \
#     --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-19_use_75_episodes_50-obj_paper_1219/model_60.pth \
#     --pointnet_class PointNet2_super \
#     --model_invariant true \
#     --output_obj_pcd_only \
#     --eval_exp_name  paper_eval_high_level_weighted-displacement_50_obj_trial_3 \


### trained on 10 objects
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/ 
python eval_robogen_with_goal_PointNet.py \
    --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-96.ckpt \
    --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-29_use_75_episodes_10-obj_paper_1229/model_60.pth \
    --pointnet_class PointNet2_super \
    --model_invariant true \
    --output_obj_pcd_only \
    --eval_exp_name  paper_eval_high_level_weighted-displacement_10_obj \

python eval_robogen_with_goal_PointNet.py \
    --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-96.ckpt \
    --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-29_use_75_episodes_10-obj_paper_1229/model_60.pth \
    --pointnet_class PointNet2_super \
    --model_invariant true \
    --output_obj_pcd_only \
    --eval_exp_name  paper_eval_high_level_weighted-displacement_10_obj_trial_2 \


python eval_robogen_with_goal_PointNet.py \
    --low_level_exp_dir  /project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1121-50-combined-low-level-unet-diffusion-no-dense-step-around-goal/2024.11.21/02.55.41_train_dp3_robogen_open_door/ \
    --low_level_ckpt_name epoch-96.ckpt \
    --high_level_ckpt_name  /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-29_use_75_episodes_10-obj_paper_1229/model_60.pth \
    --pointnet_class PointNet2_super \
    --model_invariant true \
    --output_obj_pcd_only \
    --eval_exp_name  paper_eval_high_level_weighted-displacement_10_obj_trial_3 \








