pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=prope_articubot \
    visual_encoder=resnet_prope \
    task.dataset.max_train_episodes=1
