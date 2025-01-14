cd test_PointNet2
torchrun --standalone --nproc_per_node=8 train_ddp_weighted_displacement.py --batch_size 110 \
    --num_epochs 81 --model_type pointnet2_super --model_invariant \
    --exp_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps \
    --num_train_objects 500_plus_all_real_world_clean_distorted_goal \
    --dataset_prefix /scratch/yufeiw2/dp3_demo \
    --load_model_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-12-29_use_75_episodes_500_plus_all_real_world_clean_distorted_goal-obj/model_4.pth \
    --exp_name _2

