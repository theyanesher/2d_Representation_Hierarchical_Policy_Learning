pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.31/21.31.57_diffusion_unet_hybrid_depth_only \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name depth_only \
  --folder_name data/rgb_eval