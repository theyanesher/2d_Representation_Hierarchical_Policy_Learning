pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task.dataset.data_dir=data/rgb/41510 \
  task.dataset.max_train_episodes=1 \
  action_mode=relative