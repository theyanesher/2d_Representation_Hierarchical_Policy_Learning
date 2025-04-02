cd test_PointNet2
torchrun --standalone --nproc_per_node=2 train_ddp_weighted_displacement.py --batch_size 50 \
    --num_epochs 60 --model_type pointnet2_super --model_invariant \
    --exp_path ./high_level \
    --num_train_objects 200 \
    --dataset_prefix /media/chenyuan/7e1e609b-387d-4a95-9219-2535fbbecfe9/articubot_demo/dp3_demo \
    --exp_name _paper_1211