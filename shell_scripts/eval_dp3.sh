python eval.py \
    --config-name dp3 \
    task=mimicgen_pc_abs_eval \
    task_name=square_d2 \
    policy.pointnet_type=act3d \
    low_level_dir=data/outputs/2025.06.23/14.17.34_train_dp3_square_d2 # act3d
    # low_level_dir=data/outputs/2025.06.21/18.36.45_train_dp3_square_d2 # pointnet++
    # low_level_dir=data/outputs/2025.06.20/15.09.31_train_dp3_square_d2 \ pointnet
