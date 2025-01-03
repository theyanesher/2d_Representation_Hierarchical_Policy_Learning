"""
To run this file, you actually need to set change
    action_pred_backup = copy.deepcopy(action_pred)
to
    action_pred_backup = copy.deepcopy(action_pred).detach()
in dp3.py, but I'm not doing that because I'm scared of breaking some piece of code.
"""

from copy import deepcopy
from diffusion_policy_3d.common.pytorch_util import dict_apply
# from diffusion_policy_3d.train_ddp import ddp_setup
from diffusion_policy_3d.dataset.robogen_dataset import RobogenDataset
from torch.utils.data.distributed import DistributedSampler
import hydra
import numpy as np
from omegaconf import OmegaConf
import os 
from pathlib import Path
import random
from torch.utils.data import DataLoader
from train import TrainDP3Workspace
import torch
from torch.distributed import init_process_group, destroy_process_group
import datetime

ROOT_DIR = Path(__file__).parent.parent.parent

def load_low_level_policy():
    checkpoint_name = 'latest.ckpt'
    exp_dir = f"{ROOT_DIR}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door"
    with hydra.initialize(config_path='../../3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        low_level_cfg = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
    lowlevel_workspace =TrainDP3Workspace(low_level_cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    lowlevel_workspace.load_checkpoint(path=checkpoint_dir)

    lowlevel_policy = deepcopy(lowlevel_workspace.model)
    if lowlevel_workspace.cfg.training.use_ema:
        lowlevel_policy = deepcopy(lowlevel_workspace.ema_model)
    lowlevel_policy.eval()
    lowlevel_policy.reset()
    lowlevel_policy = lowlevel_policy.to('cuda:0')
    return lowlevel_policy

def load_high_level_policy():
    goal_checkpoint_name = 'epoch-30.ckpt'
    goal_exp_dir = f"{ROOT_DIR}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0807-200-obj-pred-goal-gripper-PointNet2-backbone-UNet-diffusion-ep-75-epsilon/2024.08.07/14.03.40_train_dp3_robogen_open_door"
    with hydra.initialize(config_path='../../3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        high_level_cfg = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(goal_exp_dir)),
        )
    highlevel_workspace = TrainDP3Workspace(high_level_cfg)
    highlevel_checkpoint_dir = "{}/checkpoints/{}".format(goal_exp_dir, goal_checkpoint_name)
    highlevel_workspace.load_checkpoint(path=highlevel_checkpoint_dir)

    goal_policy = deepcopy(highlevel_workspace.model)
    if highlevel_workspace.cfg.training.use_ema:
        goal_policy = deepcopy(highlevel_workspace.ema_model)
    goal_policy.eval()
    goal_policy.reset()
    goal_policy = goal_policy.to('cuda:0')
    return goal_policy

def get_eval_dataset_paths():
    experiment_folder = ROOT_DIR / 'data/diverse_objects/'
    experiment_name = '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first'
    task_name = 'task_open_the_door_of_the_storagefurniture_by_its_handle'
    experiment_folder = list(map(lambda x: x / task_name / 'experiment' / experiment_name, experiment_folder.iterdir()))
    
    return experiment_folder 

def load_train_dataset():
    dataset_paths = Path('/data/minon/dp3_demo_combined_2_step_0/')
    dataset_paths = list(dataset_paths.iterdir())
    dataset_paths = list(filter(lambda x: x.is_dir(), dataset_paths))
    dataset_paths = dataset_paths[:5] # don't need all data
    train_dataset = RobogenDataset(dataset_paths, enumerate=True, horizon=2,
                                   observation_mode="act3d_goal_displacement_gripper_to_object",
                                   kept_in_disk=True, load_per_step=True, num_load_episodes=50,
                                   prediction_target='action', is_pickle=True, pad_before=1, pad_after=3,
                                   dataset_keys=['state', 'action', 'point_cloud', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd'])
    return train_dataset

def get_dataloader(dataset_object, shuffle=False, batch_size=2):
    dataloader = DataLoader(dataset_object, 
                            shuffle=shuffle,
                            # sampler=DistributedSampler(dataset_object),
                            batch_size=batch_size,
                            num_workers=5,
                            pin_memory=True,
                            )
    return dataloader

def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def copy_batch(batch):
    copy_batch = dict()
    for key in batch:
        copy_batch[key] = batch[key].clone()
    return batch

def translate_batch(batch, translation=1.0): 
    new_batch = copy_batch(batch) # too mem expensive
    new_batch = batch
    for key in batch:
        if key in ('point_cloud', 'gripper_pcd', 'goal_gripper_pcd'):
            new_batch[key] += translation
        elif key == 'agent_pos':
            new_batch[key][:,:,:3] += translation
    return new_batch

def test_translation_invariance(policy, dataloader, is_high=True):
    print("Calculating difference between outputs")
    norms = []
    device = torch.device(0)
    for i, batch in enumerate(dataloader):
        if i == 10:
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))['obs']
        seed = random.randint(0, 100)

        # run policy
        set_random_seed(seed, using_cuda=True)
        model_output = policy.predict_action(batch)['action']

        # translate input batch
        translation_vector = torch.ones(3).to(device)
        translated_batch = translate_batch(batch, translation_vector)

        # run policy with translated batch
        set_random_seed(seed, using_cuda=True)
        translated_model_output = policy.predict_action(translated_batch)['action']

        #calculate difference between translated vs non-translated vector
        if is_high:
            model_output = model_output[:, :2, :].view(-1, 2, 4, 3)
            translated_model_output = translated_model_output[:, :2, :].view(-1, 2, 4, 3)
        diff = torch.linalg.norm(model_output - translated_model_output, axis=-1)
        norms.append(diff)
    average_diff = torch.mean(torch.stack(norms))
    std_dev = torch.std(torch.stack(norms))
    print(f"diff mean: {average_diff} std_dev: {std_dev}")
    return average_diff

def main():
    lowlevel_policy = load_low_level_policy()
    highlevel_policy = load_high_level_policy()
    train_dataset = load_train_dataset()
    train_dataloader = get_dataloader(train_dataset, shuffle=True)

    test_translation_invariance(highlevel_policy, train_dataloader, is_high=True)
    test_translation_invariance(lowlevel_policy, train_dataloader, is_high=False)
    
if __name__ == '__main__':
    main()