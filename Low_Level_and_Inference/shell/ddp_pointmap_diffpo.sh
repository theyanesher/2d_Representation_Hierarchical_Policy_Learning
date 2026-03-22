pixi run torchrun --master_port=$(( RANDOM % 10000 + 20000 )) --nproc_per_node=2 \
  diffusion_policy/train_ddp.py --config-name=train_ddp_diffusion_unet_hybrid_workspace.yaml \
  task=pointmap_articubot \
  task.dataset.data_dir=data/rgb_camera_randomized/41510 \
  policy.encoder_backbone=shared_crop \
  action_mode=hybrid_delta \
  task.dataset.pointmap_frame=robot_frame \
  dataloader.batch_size=32