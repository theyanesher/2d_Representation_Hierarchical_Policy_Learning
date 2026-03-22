pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.02.18/13.26.46_diffusion_unet_hybrid_pointmap/ \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name pointmap \
  --folder_name data/rgb_eval