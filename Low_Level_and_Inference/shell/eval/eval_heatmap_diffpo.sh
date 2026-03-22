pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.02.09/03.20.18_diffusion_unet_hybrid_heatmap \
  --high_level_ckpt_name weighted_displacement_model/exps/2026-02-06_01-04-41_use_all_data_only_41510-obj_test_cleaned_code/model_60.pth \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name heatmap \
  --folder_name data/rgb_eval \
  --randomize_camera 2