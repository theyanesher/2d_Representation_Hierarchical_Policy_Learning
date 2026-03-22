pixi run torchrun --nproc_per_node=4 diffusion_policy/train_ddp.py --config-name=train_ddp_diffusion_unet_hybrid_workspace.yaml \
  task.dataset.data_dir=data/rgb_camera_randomized/41510 \
  policy.encoder_backbone=vggt \
  dataloader.batch_size=100 \
  policy.crop_shape='[224,224]' \
  training.num_epochs=100