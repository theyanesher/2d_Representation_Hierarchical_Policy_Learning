python mimicgen_train.py \
    --config-name articubot_gc_diffusion_unet \
    task=articubot_nogoal_diffpo \
    task_name=square_d2 \
    dataset_path=data/robomimic/datasets/square_d2/square_d2_pcd_abs_images.hdf5 \
    n_demo=1000 \
    high_level_dir=third_party/robogen/test_PointNet2/exps/pointnet2_super_06-30-15-55_square_d2_abs \
    conditioning_type=null \
    goal_conditioning=False \
    task.env_runner.use_fixed_seeds=False