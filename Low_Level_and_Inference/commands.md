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
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task.dataset.data_dir=data/rgb/41510 task.dataset.max_train_episodes=1

#### FOR LOCAL ######

pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task.dataset.data_dir=/home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/outputs/Heatmap_Articubot_Dataset/

#DP + Heatmap
pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task=rgb_heatmap_articubot.yaml \task.dataset.data_dir=/scratch/pbhowal/Articubot_Data_For_DP_and_Groot/Heatmap_Articubot_Dataset/

pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task=rgb_heatmap_articubot.yaml \task.dataset.data_dir=/home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/outputs/Heatmap_Articubot_Dataset/


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

pixi run python diffusion_policy/eval_hierarchical_diffpo_single_object.py --low_level_exp_dir outputs/2026.03.25/15.30.38_diffusion_unet_hybrid_hybrid/ --low_level_ckpt_name epoch_40.ckpt --high_level_ckpt_name ../../../outputs/High_Level_Model/model_8.pth --update_goal_freq 8 --folder_name ../../../data/rgb_eval/



####### DP + GHOST HEATMAP

pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task=rgb_heatmap_articubot_ghost.yaml

######### DP + GHOST HEATMAP + HEATMAP_ROPE_VIT_ENCODER + MINSNR

pixi run python diffusion_policy/train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task=rgb_heatmap_articubot_ghost policy.encoder_backbone=vit_heatmap_rope task.dataset.data_dir=/home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/data/rgb_mino_data_ghost_heatmap_dataset/ logging.name=diffusion_vit_heatmap_rope_minsnr name=diffusion_vit_heatmap_rope_minsnr dataloader.batch_size=24 policy.use_min_snr=true policy.min_snr_gamma=5.0

pixi run python diffusion_policy/eval_hierarchical_diffpo_single_object.py --low_level_exp_dir outputs/2026.03.27/00.59.09_diffusion_vit_heatmap_rope_minsnr_hybrid/ --low_level_ckpt_name epoch_40.ckpt --high_level_ckpt_name ../../../outputs/High_Level_Model/model_8.pth --update_goal_freq 8 --folder_name ../../../data/rgb_eval/

####### GROOT + GHOST HEATMAP + HEATMAP ROPE STYLE POSITION EMBEDDING

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost task.dataset.data_dir=../../../../data/ logging.name=groot_vit_rope_heatmap_single_object name=groot_vit_rope_heatmap_single_object dataloader.batch_size=64 visual_encoder=vit_heatmap_rope

pixi run python diffusion_policy/eval_hierarchical_diffpo_single_object.py --low_level_exp_dir outputs/RoPE_Unscaled_GROOT/ --low_level_ckpt_name epoch_50.ckpt --high_level_ckpt_name ../../../outputs/High_Level_Model/model_8.pth --update_goal_freq 8 --folder_name ../../../data/rgb_eval/



####### GROOT + GHOST HEATMAP + HEATMAP ROPE STYLE POSITION EMBEDDING + FLOW TIME EMBEDDING

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost task.dataset.data_dir=../../../data/rgb_mino_data_ghost_heatmap_dataset/ logging.name=groot_vit_rope_heatmap_single_object_timestep_flow name=groot_vit_rope_heatmap_single_object_timestep_flow dataloader.batch_size=64 visual_encoder=vit_heatmap_rope policy.visual_encoder_cfg.use_flow_timestep_rope=true







####### GROOT + GHOST HEATMAP + HOMMI STYLE HEATMAPS + CHUNKS USED AS POSITIONAL EMBEDDINGS

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap
_articubot_ghost task.dataset.data_dir=/home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/data/rgb_mino_data_
ghost_heatmap_dataset/ logging.name=groot_dinov2_hommi_rgb_resnet_heatmap_single_object name=groot_dinov2_hommi_rgb_resnet_
heatmap_single_object dataloader.batch_size=22 visual_encoder=dinov2_hommi_style_heatmap 


pixi run python diffusion_policy/eval_hierarchical_diffpo_single_object.py --low_level_exp_dir outputs/2026.03.27/02.01.52_groot_dinov2_hommi_rgb_resnet_heatmap_single_object_hybrid/ --low_level_ckpt_name epoch_40.ckpt --high_level_ckpt_name ../../../outputs/High_Level_Model/model_8.pth --update_goal_freq 8 --folder_name ../../../data/rgb_eval/

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost task.dataset.data_dir=/home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/data/rgb_mino_data_ghost_heatmap_dataset/ logging.name=groot_dinov2_hommi_rgb_resnet_heatmap_single_object name=groot_dinov2_hommi_rgb_resnet_heatmap_single_object dataloader.batch_size=22 visual_encoder=dinov2_hommi_style_heatmap

####### GROOT + GHOST HEATMAP + HOMMI STYLE HEATMAPS + CHUNKS USED AS POSITIONAL EMBEDDINGS + SINGLE HEATMAP FOR GHOST HEATMAP

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost_single_channel task.dataset.data_dir=/home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/data/rgb_mino_data_ghost_heatmap_dataset/ logging.name=groot_dinov2_hommi_rgb_resnet_heatmap_single_object_single_channel name=groot_dinov2_hommi_rgb_resnet_heatmap_single_object_single_channel dataloader.batch_size=22 visual_encoder=dinov2_hommi_style_heatmap policy.visual_encoder_cfg.heatmap_channels=1



####### GROOT + GHOST HEATMAP + HOMMI STYLE HEATMAPS + ENTIRE IMAGE AND RESNET AS POSITIONAL EMBEDDINGS

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost task.dataset.data_dir=/home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/data/rgb_mino_data_ghost_heatmap_dataset/ logging.name=groot_dinov2_hommi_single_conv_rgb_resnet_heatmap_single_object name=groot_dinov2_hommi_single_conv_rgb_resnet_heatmap_single_object dataloader.batch_size=22 visual_encoder=dinov2_hommi_style_heatmap policy.visual_encoder_cfg.use_single_conv=true

####### GROOT + GHOST HEATMAP + VIT WITH HEATMAP AS POSITIONAL EMBEDDINGS (NOT ROPE)


pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost task.dataset.data_dir=/home/pratik_final/Downloads/Bimanual/Articubot_Data_Experiments/data/rgb_mino_data_ghost_heatmap_dataset/ logging.name=groot_VIT_Heatmap_positional_embedding_single_object name=groot_VIT_Heatmap_positional_embedding_single_object dataloader.batch_size=22 visual_encoder=vit_heatmap_pos_embedding



























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
  --low_level_exp_dir outputs/2026.03.23/15.38.45_diffusion_unet_hybrid_hybrid/ \
  --low_level_ckpt_name epoch_40.ckpt --eval_exp_name Just_DP_No_Heatmaps \
  --folder_name ../../../data/rgb_eval/


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


Commands Final

GROOT + ROPE POSITION EMBEDDING + 1 CHANNEL GHOST HEATMAP

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost_single_channel task.dataset.data_dir=outputs/one_object_4_point_sqrt_heatmap/ logging.name=groot_vit_rope_heatmap_single_object_single_channel name=groot_vit_rope_heatmap_single_object_single_channel dataloader.batch_size=4 visual_encoder=vit_heatmap_rope policy.visual_encoder_cfg.heatmap_channels=1


GROOT + ROPE POSITION EMBEDDING + 4 CHANNEL EXPONENTIAL HEATMAP

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_four_channels task.dataset.data_dir=outputs/All_Heatmap_Dataset/ logging.name=groot_vit_rope_exp_heatmap_single_object_four_channel name=groot_vit_rope_exp_heatmap_single_object_four_channel dataloader.batch_size=4 visual_encoder=vit_heatmap_rope policy.visual_encoder_cfg.heatmap_channels=4

GROOT + ROPE POSITION EMBEDDING + 4 CHANNEL GHOST HEATMAP + 4 CHANNEL CURRENT GRIPPER HEATMAPS

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost_current_heatmaps_added task.dataset.data_dir=outputs/All_Heatmap_Dataset/ logging.name=groot_vit_rope_ghost_heatmap_single_object_eight_channel name=groot_vit_rope_ghost_heatmap_single_object_eight_channel dataloader.batch_size=4 visual_encoder=vit_heatmap_rope policy.visual_encoder_cfg.heatmap_channels=8


pixi run python diffusion_policy/eval_hierarchical_diffpo_single_object.py --low_level_exp_dir outputs/groot_vit_rope_ghost_heatmap_single_object_eight_channel/ --low_level_ckpt_name epoch_45.ckpt --high_level_ckpt_name ../../../ArticuBot/outputs/High_Level_Policy/model_8.pth --update_goal_freq 8 --folder_name ../../../ArticuBot/data/rgb_eval/



GROOT + ROPE POSITION EMBEDDING + 4 CHANNEL EXPONENTIAL HEATMAP + 4 CHANNEL CURRENT GRIPPER HEATMAPS

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_current_heatmaps_added task.dataset.data_dir=outputs/All_Heatmap_Dataset/ logging.name=groot_vit_rope_exp_heatmap_single_object_eight_channel name=groot_vit_rope_exp_heatmap_single_object_eight_channel dataloader.batch_size=4 visual_encoder=vit_heatmap_rope policy.visual_encoder_cfg.heatmap_channels=8


ADD ROTATION AUGMENTATION TO STANDARD HEATMAP ROPE

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost task.dataset.data_dir=../../../../data/ logging.name=groot_vit_rope_heatmap_single_object name=groot_vit_rope_heatmap_single_object dataloader.batch_size=64 visual_encoder=vit_heatmap_rope task.dataset.heatmap_augmentation.enabled=true task.dataset.heatmap_augmentation.rot_sigma=3.0 task.dataset.heatmap_augmentation.rot_max=10.0 task.dataset.heatmap_augmentation.p=0.5

ADD ANGULAR COMPONENT TO STANDARD ROPE 

pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=rgb_heatmap_articubot_ghost task.dataset.data_dir=outputs/All_Heatmap_Dataset/ logging.name=groot_vit_rope_heatmap_single_object name=groot_vit_rope_heatmap_single_object dataloader.batch_size=1 visual_encoder=vit_heatmap_rope policy.visual_encoder_cfg.use_direction_axes_in_rope=true policy.hidden_size=1152 policy.input_embedding_dim=1152 policy.diffusion_model_cfg.output_dim=1152



pixi run python diffusion_policy/train.py --config-name=train_flow_matching_dit_workspace.yaml task=articubot_goal_gripper task.dataset.data_dir=outputs/All_Heatmap_Dataset/ logging.name=groot_dinov2_goal_gripper_DIT name=groot_dinov2_goal_gripper_DIT dataloader.batch_size=22 visual_encoder=dinov2




