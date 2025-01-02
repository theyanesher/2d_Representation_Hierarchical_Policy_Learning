cd test_PointNet2
torchrun --standalone --nproc_per_node=8 train_ddp_weighted_displacement.py --batch_size 100 \
    --num_epochs 81 --model_type pointnet2_super --model_invariant \
    --exp_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps \
    --num_train_objects 500_plus_all_real_world \
    --dataset_prefix /scratch/yufeiw2/dp3_demo \
    --keep_gripper_in_fps 1 \
    --add_one_hot_encoding 1 \
    --output_obj_pcd_only \
    --load_model_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-11-28_use_75_episodes_500-obj_output_obj_only_keep_gripper_in_fps_one_hot/model_96.pth
