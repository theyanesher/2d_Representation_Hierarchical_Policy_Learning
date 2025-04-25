cd test_PointNet2
torchrun --standalone --nproc_per_node=2 train_ddp_weighted_displacement.py --batch_size 50 \
    --num_epochs 60 --model_type pointnet2_super --model_invariant \
    --exp_path /home/chenyuan/RoboGen-sim2real/exps \
    --num_train_objects articulated \
    --dataset_prefix /home/chenyuan/RoboGen-sim2real/data/dp3_demo/seuss_gen \
    --exp_name test

