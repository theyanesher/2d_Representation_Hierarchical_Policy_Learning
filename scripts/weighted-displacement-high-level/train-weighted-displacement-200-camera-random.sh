cd test_PointNet2
torchrun --standalone --nproc_per_node=1 train_ddp_weighted_displacement.py --batch_size 50 \
    --num_epochs 60 --model_type pointnet2_super --model_invariant \
    --exp_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps \
    --num_train_objects camera_random_200_obj_high_level \
    --dataset_prefix /scratch/yufeiw2/dp3_demo \
    --exp_name _paper_1211