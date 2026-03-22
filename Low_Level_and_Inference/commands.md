# Commands

## Data Generation
```
single traj data generation:
pixi run python manipulation/gen_demo.py --root_dir data/diverse_objects_all/ --extract_name 41510 --exp_name test_gen_demo --num_to_generate 1100 --max_try_times 5000

single traj data rendering: 
pixi run python manipulation/extract_images_from_states.py --folder_name data/diverse_objects_all --exp_name "test_gen_demo" --extract_name 41510 --traj_idx 0 --randomize_camera 0 --observation_mode image_plucker_pointmap

data generation:
pixi run python manipulation/parallel_extract_images.py --extract_name 41510 --exp_name test_gen_demo --randomize_camera 0 --observation_mode image
pixi run python manipulation/parallel_extract_images.py --extract_name 41510 --exp_name test_gen_demo --randomize_camera 1 --observation_mode image_plucker_pointmap


pixi run python manipulation/parallel_extract_images.py --extract_name 41510 --exp_name evals --randomize_camera 1 --observation_mode image --save_path data/scrap

# uploading to gcloud
gsutil cp data/data.tar gs://cmu-gpucloud-mnakuraf/rgb-articubot
```

## Visualization
```
pixi run python scripts/visualization_scripts/visualize_hdf5_dataset.py explore data/rgb_camera_randomized/41510/2025-10-30-21-05-53.h5

pixi run python scripts/visualization_scripts/visualize_hdf5_dataset.py multi data/rgb_camera_left_right_randomized/41510/
```

## training:
```
# regular diffpo
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task.dataset.data_dir=data/rgb/41510 \
  task.dataset.max_train_episodes=1

#DP + Heatmap
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task=rgb_heatmap_articubot.yaml \task.dataset.data_dir=/scratch/pbhowal/Articubot_Data_For_DP_and_Groot/Heatmap_Articubot_Dataset/

pixi run torchrun --standalone --nproc_per_node=1 diffusion_policy/train_ddp.py --config-name=train_ddp_diffusion_unet_hybrid_workspace.yaml task=rgb_heatmap_articubot.yaml \task.dataset.data_dir=/scratch/pbhowal/Articubot_Data_For_DP_and_Groot/Heatmap_Articubot_Dataset/

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml \
task=rgb_heatmap_articubot \
task.dataset.data_dir=/scratch/pbhowal/Articubot_Data_For_DP_and_Groot/Heatmap_Articubot_Dataset/ \
logging.name=groot_dinov2_rgb_resnet_heatmap_single_object \
name=groot_dinov2_rgb_resnet_heatmap_single_object \
dataloader.batch_size=22 \
visual_encoder=dinov2_rgb_resnet_heatmap

pixi run torchrun --standalone --nproc_per_node=8 diffusion_policy/train_ddp.py --config-name=train_flow_matching_dit_workspace_ddp.yaml task=rgb_heatmap_articubot task.dataset.data_dir=/scratch/pbhowal/Articubot_Data_For_DP_and_Groot/Heatmap_Articubot_Dataset/ logging.name=groot_dinov2_rgb_resnet_heatmap_single_object name=groot_dinov2_rgb_resnet_heatmap_single_object dataloader.batch_size=72 visual_encoder=dinov2_rgb_resnet_heatmap


pixi run python diffusion_policy/eval_hierarchical_diffpo_single_object.py --low_level_exp_dir outputs/2026.01.24/14.51.48_diffusion_unet_hybrid_image --low_level_ckpt_name epoch_60.ckpt --high_level_ckpt_name path/to/pointnet2.pth --update_goal_freq 5 --folder_name data/rgb_eval


# image-only diffpo
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task=image_articubot task.dataset.data_dir=data/rgb/41510 action_mode=relative

# rgbd diffpo
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=depth_articubot \
  dataloader.batch_size=48 \
  task.dataset.data_dir=data/rgb_camera_left_right_randomized/41510/ \
  policy.shared_crop=true \
  task.dataset.max_train_episodes=1

# depth-only
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=depth_only_articubot \
  dataloader.batch_size=48 \
  task.dataset.data_dir=data/rgb/41510/ \
  task.dataset.max_train_episodes=1


pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml   task=depth_articubot   task.dataset.data_dir=data/rgb_camera_left_right_randomized/41510/ dataloader.batch_size=48 hydra.run.dir=outputs/2026.01.22/14.39.31_diffusion_unet_hybrid_articubot_image

# plucker diffpo
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=plucker_articubot \
  task.dataset.data_dir=data/rgb_camera_left_right_randomized/41510/ \
  policy.shared_crop=true \
  dataloader.batch_size=48 \
  task.dataset.max_train_episodes=1

# pointmap diffpo
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=pointmap_articubot \
  task.dataset.data_dir=data/rgb_camera_left_right_randomized/41510/ \
  policy.shared_crop=true \
  task.dataset.max_train_episodes=1

# relative control
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=articubot \
  task.dataset.data_dir=data/rgb/41510/ \
  action_mode=relative \
  task.dataset.max_train_episodes=1 \
  dataloader.shuffle=True \
  training.num_epochs=10000 \
  training.checkpoint_every=100

# plucker diffpo debugging stuff
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task=plucker_articubot task.dataset.data_dir=data/rgb_camera_left_right_randomized/41510/ dataloader.batch_size=48 observation_mode='plucker_early_fusion' task.dataset.max_train_episodes=1  dataloader.shuffle=False 
```

## evaluation:
```
# diffpo_hybrid
pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.23/18.04.21_diffusion_unet_hybrid_articubot_image \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name diffpo_low_obj_variation \
  --folder_name data/min_obj_rand_eval

# plucker late fusion
pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.29/21.24.00_diffusion_unet_hybrid_plucker/ \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name plucker_late_fusion \
  --folder_name data/rgb_eval \
  --randomize_camera 1 \
  --model_mode plucker_late_fusion

# relative
pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.24/04.47.37_diffusion_unet_hybrid_hybrid \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name relative \
  --folder_name data/rgb_eval

# depth
pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.28/02.34.15_diffusion_unet_hybrid_articubot_image \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name depth \
  --model_mode diffpo_hybrid_depth \
  --folder_name data/rgb_eval \
  --randomize_camera 1

pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.28/20.40.02_diffusion_unet_hybrid_depth_only \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name depth \
  --model_mode depth_only \
  --folder_name data/rgb_eval

# image only
pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.24/14.51.48_diffusion_unet_hybrid_image \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name image_only \
  --model_mode diffpo_image \
  --folder_name data/rgb_eval

# pointmap evaluation
pixi run python diffusion_policy/eval_diffpo_single_object.py \
  --low_level_exp_dir outputs/2026.01.30/13.27.59_diffusion_unet_hybrid_pointmap \
  --low_level_ckpt_name epoch_60.ckpt --eval_exp_name pmlf \
  --model_mode pointmap_late_fusion \
  --folder_name data/rgb_eval \
  --randomize_camera 1

# print eval results
pixi run python scripts/print_eval_results.py --num_objs 1 --d outputs_eval/2026.02.03/04.06.51_diffusion_unet_hybrid_pointmap/epoch_60.ckpt/2026-02-04_00-07
```

