#!/bin/bash
set -x

export PATH="$HOME/.pixi/bin:$PATH"

cd /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference

# Copy h5 data to local scratch in parallel
SRC_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/Articubot_Data/one_object_4_point_sqrt_heatmap"
DST_DIR="/local/slurm-${SLURM_JOB_ID}/local/pbhowal/one_object_4_point_sqrt_heatmap"
mkdir -p "$DST_DIR"

SECONDS=0
for f in "$SRC_DIR"/*.h5; do
    rsync -az "$f" "$DST_DIR/" &
done
wait
echo "Data copy done in ${SECONDS}s"

# Training run 1 - cross attention (resume) - 2 epochs only for testing
pixi run python diffusion_policy/train.py \
  --config-name=train_flow_matching_dit_workspace.yaml \
  task=rgb_heatmap_articubot_ghost \
  task.dataset.data_dir=/local/slurm-${SLURM_JOB_ID}/local/pbhowal/one_object_4_point_sqrt_heatmap/ \
  logging.name=groot_Dinov2_rgb_resnet_heatmap_cross_attention \
  name=groot_Dinov2_rgb_resnet_heatmap_cross_attention \
  dataloader.batch_size=40 \
  visual_encoder=dinov2_rgb_resnet_heatmap \
  policy.visual_encoder_cfg.add_cross_attention_mixing=true \
  policy.visual_encoder_cfg.heatmap_channels=4 \
  training.resume=true \
  +training.resume_ckpt_path=/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/groot_Dinov2_rgb_resnet_heatmap_cross_attention/checkpoints/latest.ckpt \
  training.num_epochs=2 \
  hydra.run.dir=outputs/groot_Dinov2_rgb_resnet_heatmap_cross_attention &

# Training run 2 - self attention (resume) - 2 epochs only for testing
pixi run python diffusion_policy/train.py \
  --config-name=train_flow_matching_dit_workspace.yaml \
  task=rgb_heatmap_articubot_ghost \
  task.dataset.data_dir=/local/slurm-${SLURM_JOB_ID}/local/pbhowal/one_object_4_point_sqrt_heatmap/ \
  logging.name=groot_Dinov2_rgb_resnet_heatmap_self_attention \
  name=groot_Dinov2_rgb_resnet_heatmap_self_attention \
  dataloader.batch_size=48 \
  visual_encoder=dinov2_rgb_resnet_heatmap \
  policy.visual_encoder_cfg.add_self_attention_mixing=true \
  policy.visual_encoder_cfg.heatmap_channels=4 \
  training.resume=true \
  +training.resume_ckpt_path=/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/groot_Dinov2_rgb_resnet_heatmap_self_attention/checkpoints/latest.ckpt \
  training.num_epochs=2 \
  hydra.run.dir=outputs/groot_Dinov2_rgb_resnet_heatmap_self_attention &

# Wait for all background jobs to finish
wait
