import hydra
from omegaconf import OmegaConf
from pathlib import Path
from copy import deepcopy
from train import TrainDP3Workspace

ROOT_DIR = Path(__file__).parent.parent.parent

def load_policy(policy_dir, checkpoint_name,
                config_path: str = f'{ROOT_DIR}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):
    with hydra.initialize(config_path=config_path):  # same config_path as used by @hydra.main
        high_level_cfg = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(policy_dir)),
        )
    highlevel_workspace = TrainDP3Workspace(high_level_cfg)
    highlevel_checkpoint_dir = "{}/checkpoints/{}".format(policy_dir, checkpoint_name)
    highlevel_workspace.load_checkpoint(path=highlevel_checkpoint_dir)
    goal_policy = deepcopy(highlevel_workspace.model)
    if highlevel_workspace.cfg.training.use_ema:
        goal_policy = deepcopy(highlevel_workspace.ema_model)
    goal_policy.eval()
    goal_policy.reset()
    goal_policy = goal_policy.to('cuda:0')
    return goal_policy
