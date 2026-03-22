pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.02.23/01.53.09_diffusion_unet_ddp_hybrid/ \
  --low_level_ckpt_name epoch_35.ckpt --eval_exp_name rgb_eval \
  --folder_name data/rgb_eval