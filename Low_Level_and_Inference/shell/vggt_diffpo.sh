pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task.dataset.data_dir=data/rgb_camera_randomized/41510 \
  policy.encoder_backbone=vggt \
  dataloader.batch_size=100 \
  policy.crop_shape='[224,224]' \
  task.dataset.max_train_episodes=5


# pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
#   task.dataset.data_dir=data/rgb_camera_randomized/41510 \
#   policy.crop_shape='[224,224]' \
#   task.dataset.max_train_episodes=1 
