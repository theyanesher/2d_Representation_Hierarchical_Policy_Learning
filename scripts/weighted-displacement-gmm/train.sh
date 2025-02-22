torchrun --standalone --nproc_per_node=1 train_ddp_weighted_displacement_gmm.py --batch_size 50    \
 --num_epochs 61 --model_type pointnet2_super --model_invariant     \
 --exp_path /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/GMM/     \
 --num_train_objects 10     \
 --dataset_prefix /scratch/yufeiw2/dp3_demo     \
 --conditioning_on_demo 1 \
 --demo_use_attn 0 \
 --demo_use_cur_obs 0 \
 --exp_name _debug
#  --dataset_prefix /scratch/yufeiw2/dp3_demo     \
#   --exp_name _fixed_variance_0.001 \
#  --fixed_variance 0.001 \
