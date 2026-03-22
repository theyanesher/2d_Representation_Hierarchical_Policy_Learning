pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=articubot \
  task.dataset.data_dir=data/rgb/41510/ \
  action_mode=relative