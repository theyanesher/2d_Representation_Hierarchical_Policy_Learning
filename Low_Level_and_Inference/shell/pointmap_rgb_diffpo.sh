pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=pointmap_articubot \
  task.dataset.data_dir=data/rgb_camera_randomized/41510/ \
  policy.encoder_backbone=single_pointmap_shared_crop \
  dataloader.batch_size=40 \
  action_mode=delta \
  task.dataset.pointmap_frame=gripper_frame \
  task.dataset.max_train_episodes=1