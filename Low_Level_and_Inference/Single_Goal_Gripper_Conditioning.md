MIMICGEN COMMANDS =>


Single GOAL Gripper Conditioning:



pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=articubot_goal_gripper task.dataset.data_dir=outputs/All_Heatmap_Dataset/ logging.name=groot_dinov2_goal_gripper_DIT name=groot_dinov2_goal_gripper_DIT dataloader.batch_size=22 visual_encoder=dinov2








