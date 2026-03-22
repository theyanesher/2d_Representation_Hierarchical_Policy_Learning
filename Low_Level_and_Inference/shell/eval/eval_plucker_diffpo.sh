pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.02.24/11.37.04_diffusion_unet_ddp_plucker/ \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name plucker_late_fusion \
  --folder_name data/rgb_eval