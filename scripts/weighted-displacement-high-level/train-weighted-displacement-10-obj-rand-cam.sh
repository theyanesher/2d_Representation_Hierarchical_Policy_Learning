cd test_PointNet2
torchrun --standalone --nproc_per_node=4 train_ddp_weighted_displacement.py --batch_size 110 \
    --num_epochs 61 --model_type pointnet2_super --model_invariant \
    --exp_path /home/yufei/projects/RoboGen-sim2real/test_PointNet2/exps \
    --num_train_objects camera_random_10_obj_high_level \
    --dataset_prefix /home/yufei/projects/RoboGen-sim2real/data/dp3_demo \
    --exp_name _paper_0121