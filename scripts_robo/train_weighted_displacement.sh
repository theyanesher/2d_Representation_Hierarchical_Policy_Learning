torchrun --standalone --nproc_per_node=4 train_ddp_weighted_displacement.py --batch_size 60 \
    --num_epochs 60 --model_type pointnet2 --model_invariant \
    --exp_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps