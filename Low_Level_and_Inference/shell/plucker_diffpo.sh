pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=plucker_articubot \
  task.dataset.data_dir=data/rgb_camera_randomized/41510/ \
  policy.encoder_backbone=shared_crop \
  dataloader.batch_size=32
