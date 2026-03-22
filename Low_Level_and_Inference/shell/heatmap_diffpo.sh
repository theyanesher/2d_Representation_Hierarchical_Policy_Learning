pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=heatmap_articubot \
  dataloader.batch_size=48 \
  policy.encoder_backbone=shared_crop 
  # task.dataset.max_train_episodes=1
