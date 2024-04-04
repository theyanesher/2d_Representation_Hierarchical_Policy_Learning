import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif, get_pc, take_round_images_around_object
from manipulation.gpt_reward_api import get_handle_pos
import pybullet as p
import numpy as np
from copy import deepcopy
import pytorch3d.ops as torch3d_ops
import sys
from termcolor import cprint
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from manipulation.gpt_reward_api import get_joint_state
sys.path.append("/home/ziyu/Desktop/workspace/RoboGen-sim2real")

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    lastest_ckpt_path = workspace.get_checkpoint_path(tag="latest")
    if lastest_ckpt_path.is_file():
        cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
        workspace.load_checkpoint(path=lastest_ckpt_path)
    # workspace.load_checkpoint(path="/home/ziyu/Desktop/workspace/RoboGen-sim2real/3d_diffusion_policy/dp3_data/outputs/robogen_open_door-dp3-1458_seed0/checkpoints/latest.ckpt")
    policy = deepcopy(workspace.model)
    policy.eval()

    env, _ = build_up_env(
        "/home/ziyu/Desktop/workspace/RoboGen-sim2real/data/storagefurniture_48700/storagefurniture_48700_sac/open_the_door_of_the_storagefurniture_by_its_handle_The_robotic_arm_will_open_the_door_of_the_storage_furniture_by_its_handle.yaml",
        "data/storagefurniture_48700/storagefurniture_48700_sac/task_open_the_door_of_the_storagefurniture_by_its_handle",
        "open_the_storage_furniture_door",
        # "/home/ziyu/Desktop/workspace/RoboGen-sim2real/data/sac_storagefurniture/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-16-01-21/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/2024-03-27-16-02-33/grasp_the_door_handle_primitive/states/state_0.pkl",
        # "/home/ziyu/Desktop/workspace/RoboGen-sim2real/data/sac_storagefurniture/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-16-01-21/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/2024-03-27-16-02-33/open_the_door/states/state_0.pkl",
        None,
        render=True, 
        randomize=False,
        obj_id=0,
    )
    object_name = "StorageFurniture"
    env.reset()
    env_ = RobogenPointCloudWrapper(env, object_name)
    env = MultiStepWrapper(env_, n_obs_steps=2, n_action_steps=4, max_episode_steps=600, reward_agg_method='sum')
    
    obs = env.reset()
    final_rgbs = []
    episode_reward = 0
    for _ in range(150):
        # input("take action")
        np_obs_dict = dict(obs)
        obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to('cuda'))

        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
            action_dict = policy.predict_action(obs_dict_input)
        
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        action = np_action_dict['action'].squeeze(0)
        obs, reward, done, info = env.step(action)

        done = np.all(done)
        episode_reward += reward
        final_rgbs.append(env.env.render())
        if done:
            break

    
    joint_angle = get_joint_state(env_._env, "StorageFurniture", "joint_0")
    print("episode reward: ", episode_reward)
    print("joint angle: ", joint_angle)
    save_numpy_as_gif(np.array(final_rgbs), "test.gif")
        

if __name__ == "__main__":
    main()




