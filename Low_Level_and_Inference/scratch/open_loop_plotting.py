import h5py
import matplotlib.pyplot as plt
import numpy as np
import hydra
from diffusion_policy.diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from omegaconf import OmegaConf, open_dict
from copy import deepcopy
import torch

if __name__ == "__main__":
    exp_dir = 'outputs/2026.01.17/16.27.51_diffusion_unet_hybrid_articubot_image'
    checkpoint_name = 'epoch_95.pth'
    with hydra.initialize(config_path='diffusion_policy/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="train_diffusion_unet_hybrid_workspace.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
    cfg = recomposed_config
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir, )
    low_level_policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        low_level_policy = deepcopy(workspace.ema_model)
    low_level_policy.eval()
    low_level_policy.reset()
    low_level_policy = low_level_policy.to('cuda')


    data = h5py.File('data/rgb_30_debug/41510/2025-10-30-21-05-53.h5', 'r')
    observations = data['obs']
    actions = data['action']
    total_steps = observations.shape[0]
    pred_actions = []
    for t in range(0, total_steps - 1):
        obs_t = {}
        for key in ['rgb', 'state']:
            if key == 'rgb':
                for i in range(3):
                    obs_t[f'cam{i}_image'] = torch.tensor(observations[key][t:t+2, i], dtype=torch.float32).permute(0,  3, 1, 2).to('cuda') / 255.0
            else:
                obs_t[key] = torch.tensor(observations[key][t:t+2], dtype=torch.float32).to('cuda')
        action_t = torch.tensor(actions[t:t+2], dtype=torch.float32).to('cuda')
        with torch.no_grad():
            pred_action = low_level_policy.predict_action(obs_t)
        print(f"Step {t}:")
        print(f"  Ground Truth Action: {action_t}")
        print(f"  Predicted Action:    {pred_action.cpu().numpy()}")
        pred_actions.append(pred_action.cpu().numpy())