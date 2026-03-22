pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=pointmap_only_articubot \
  dataloader.batch_size=48 \
  task.dataset.data_dir=data/rgb_camera_randomized/41510/ \
  policy.encoder_backbone=shared_crop