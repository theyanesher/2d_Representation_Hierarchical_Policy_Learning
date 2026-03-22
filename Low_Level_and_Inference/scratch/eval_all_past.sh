pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.14/18.20.02_train_diffusion_unet_hybrid_articubot_image \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name diffpo \
  --folder_name data/rgb_eval

pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.18/22.40.52_diffusion_unet_hybrid_articubot_image \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name resnet_34 \
  --folder_name data/rgb_eval

pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.18/23.03.19_diffusion_unet_hybrid_articubot_image \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name left_right_randomization \
  --folder_name data/rgb_eval \
  --randomize_camera 1

pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.21/00.49.26_diffusion_unet_hybrid_articubot_image \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name diffpo_compression \
  --folder_name data/rgb_eval

pixi run python diffusion_policy/eval_diffpo_single_object.py \
    --low_level_exp_dir outputs/2026.01.20/01.41.40_diffusion_unet_hybrid_plucker \
    --low_level_ckpt_name epoch_60.ckpt --eval_exp_name diffpo_plucker \
    --folder_name data/rgb_eval \
    --randomize_camera 1 \
    --model_mode plucker_early_fusion

pixi run python diffusion_policy/eval_diffpo_single_object.py \
    --low_level_exp_dir outputs/2026.01.21/00.07.44_diffusion_unet_hybrid_plucker \
    --low_level_ckpt_name epoch_60.ckpt --eval_exp_name diffpo_plucker \
    --folder_name data/rgb_eval \
    --randomize_camera 1 \
    --model_mode plucker_late_fusion

pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.18/01.23.42_diffusion_unet_hybrid_articubot_image \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name full_randomization \
  --folder_name data/rgb_eval \
  --randomize_camera 2