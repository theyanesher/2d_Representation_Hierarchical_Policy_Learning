# MimicGen Training Commands

## Dataset Generation

```bash
PYTHONPATH=/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_MimicGen/SMITH_on_mimicgen/external/robomimic:$PYTHONPATH MUJOCO_GL=egl DISPLAY=:99 python external/mimicgen/mimicgen/scripts/convert_dataset.py --input /scratch/pbhowal/Uncertainty_Dataset/Original_Dataset/coffee_preparation_d0.hdf5 --output_dir /scratch/pbhowal/Uncertainty_Dataset/test_COFFEE_PREPERATION_output_bocpd_use_only_jerk_and_curve  --camera_height 256 --camera_width 256 --num_workers 1 --pool_size 1 --use_bayesian_decomp --bocpd_config third_party/robogen/bocpd_config_COFFEE_PREPERATION.yaml
```

## Low-Level Policy Training

### Coffee - Goal Gripper (DIT + DINOv2)
- Dataset: `test_COFFEE_output_bocpd_use_only_jerk_and_curve_CHECK_NO_GMM` (146 demos, 2 cameras, no wrist cam)
- Task config: `MimicGen_Tasks/coffee_goal_gripper`

```bash
pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=MimicGen_Tasks/coffee_goal_gripper logging.project=mimicgen_tasks logging.name=coffee_goal_gripper_DIT name=coffee_goal_gripper_DIT dataloader.batch_size=22 visual_encoder=dinov2
```

DATA GENERATION CODE 


COFFEE PREPERATION D0 


[pbhowal@autobot-0-13 lfd3d]$ export CUDA_VISIBLE_DEVICES=6                                                                                                                                                
[pbhowal@autobot-0-13 lfd3d]$ WANDB_CACHE_DIR=/project_data/held/pratik/wandb_cache WANDB_DATA_DIR=/project_data/held/pratik/wandb_data PYTHONNOUSERSITE=1 PIXI_CACHE_DIR=/project_data/held/pratik/pixi_ca
che pixi run python scripts/run_gmm_on_dataset.py --dataset_dir /project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/LOW_LEVEL_TRA
IN_DATASET_SMITH_STYLE_DATASET/test_COFFEE_PREPERATION_output_bocpd_use_only_jerk_and_curve/ --ckpt_path ../../SMITH_High_Level_FineTune/HIGH_LEVEL_POLICIES/Coffee_Preperation_d0/GHOST_High_Level/rmse_an
d_std_combi\=0.024.ckpt