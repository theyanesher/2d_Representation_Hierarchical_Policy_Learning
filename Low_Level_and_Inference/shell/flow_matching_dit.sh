pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task.dataset.data_dir=data/rgb_camera_randomized/41510 \
    policy.visual_encoder_type=resnet \
    policy.use_separate_wrist_encoder=True \
    policy.visual_encoder_cfg='{backbone: resnet18, pretrained: true, use_group_norm: true}' \
    ~policy.visual_encoder_cfg.num_prope_layers \
    ~policy.visual_encoder_cfg.num_heads \
    task.dataset.max_train_episodes=1
