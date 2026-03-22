pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.02.12/03.44.58_dit_block_hybrid \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name dit_block \
  --model_mode image \
  --folder_name data/rgb_eval