pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=prope_articubot \
  task.dataset.data_dir=data/rgb/41510 \
  policy.encoder_backbone=prope \
  policy.crop_shape='[224, 224]' \
  # task.dataset.max_train_episodes=1 \
  # training.num_epochs=10000 \
  # training.checkpoint_every=1000
