torchrun --standalone --nproc_per_node=5 train_ddp_weighted_displacement.py --batch_size 100 --num_epochs 100 --model_type pointnet2_super --model_invariant --exp_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps --dataset_prefix /scratch/yufeiw2/dp3_demo --load_model_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-10-09_use_75_episodes_500-obj/model_60.pth --num_train_objects 500


torchrun --standalone --nproc_per_node=8 train_ddp_weighted_displacement.py --batch_size 110 \
    --num_epochs 80 --model_type pointnet2_super --model_invariant \
    --exp_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps \
    --num_train_objects mixed_old_and_real_world_noisy_1119 --load_model_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-10-09_use_75_episodes_500-obj/model_60.pth