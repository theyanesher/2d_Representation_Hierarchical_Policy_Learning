import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif
import pybullet as p
import numpy as np
from copy import deepcopy
import pytorch3d.ops as torch3d_ops
import sys
from termcolor import cprint
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from manipulation.gpt_reward_api import get_joint_state
import tqdm
import json

def eval(cfg):
    for i in range(12):
        rgbs = main(cfg, i)
        save_numpy_as_gif(np.array(rgbs), f"open_door_{i}.gif")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = cfg.load_checkpoint_path
    # workspace.load_checkpoint(path=checkpoint_dir)
    # workspace.load_checkpoint(path="/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/autobot/vary_init_joint_angle_gripper/vary_init_joint_angle_gripper/2024.04.17/04.46.46_train_dp3_robogen_open_door/checkpoints/epoch=2000-test_mean_score=0.015.ckpt")
    # workspace.load_checkpoint(path="/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/vary_init_angle/2024.04.16/16.04.17_train_dp3_robogen_open_door/checkpoints/epoch=1000-test_mean_score=0.304.ckpt")
    checkpoint_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/vary_robot_init_joint_near_handle/2024.04.19/14.26.08_train_dp3_robogen_open_door/checkpoints/latest.ckpt"
    workspace.load_checkpoint(path=checkpoint_dir)

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()

    experiment_folder = "{}/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle".format(os.environ['PROJECT_DIR'])
    experiment_name = "vary_robot_init_joint_near_handle"
    experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
    all_experiments = os.listdir(experiment_path)
    all_experiments = sorted(all_experiments)

    after_reaching_init_state_file = []
    config_files = []
    for experiment in all_experiments:
        first_step_folder = "grasp_the_door_handle_primitive"
        first_stage_states_path = os.path.join(experiment_path, experiment, first_step_folder, "states")
        stage_lengths = os.path.join(experiment_path, experiment, first_step_folder, "stage_lengths.json")
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)
        reaching_phase = stage_lengths['reach_handle']
        init_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(reaching_phase))
        after_reaching_init_state_file.append(init_state_file)
        meta_info = os.path.join(experiment_path, experiment, "meta_info.json")
        with open(meta_info, "r") as f:
            meta_info = json.load(f)
        config_files.append(meta_info['config_path'])
    
    starting_from_pregrasp = True

    opened_joint_angles = {}
    for idx in tqdm.tqdm(range(len(config_files))):
        print("config path: ", config_files[idx])
        config_path = config_files[idx]
        env, _ = build_up_env(
            "{}/{}".format(os.environ['PROJECT_DIR'], config_path),
            "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle",
            "grasp_the_door_handle",
            after_reaching_init_state_file[idx] if starting_from_pregrasp else None,
            render=False, 
            randomize=False,
            obj_id=0,
            horizon=400,
        )
        
        in_gripper_frame = False
        
        object_name = "StorageFurniture"
        env.reset()
        env_ = RobogenPointCloudWrapper(env, object_name, 
                                        in_gripper_frame=in_gripper_frame, 
                                        gripper_num_points=0)
        env = MultiStepWrapper(env_, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                            max_episode_steps=600, reward_agg_method='sum')
        obs = env.reset()
        policy.reset()
        final_rgbs = []
        episode_reward = 0
        horizon = 60 if starting_from_pregrasp else 120
        for _ in tqdm.tqdm(range(horizon)):
            np_obs_dict = dict(obs)
            # change the point cloud to be in the gripper frame
            # np_obs_dict['point_cloud'] = env_._transfer_point_cloud_to_gripper_frame(np_obs_dict['point_cloud'])
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
        # print("episode reward: ", episode_reward)
        cprint(f"joint angle: {joint_angle}", "blue")
        env_._env.close()
        opened_joint_angles[config_path] = float(joint_angle)
    
        save_path = "data/eval_results/{}".format(checkpoint_dir.replace("/", "_"))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open("{}/opened_joint_angles_pregrasped_{}.json".format(save_path, starting_from_pregrasp), "w") as f:
            json.dump(opened_joint_angles, f, indent=4)
        save_numpy_as_gif(np.array(final_rgbs), "{}/open_door_{}_{:.3f}_pregrasp_{}.gif".format(save_path, idx, joint_angle, starting_from_pregrasp))

if __name__ == "__main__":
    # import cProfile, pstats, io
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    # ps.print_stats(50)
    # print(s.getvalue())
    # ps = pstats.Stats(pr, stream=s).sort_stats('time')
    # ps.print_stats(50)
    # print(s.getvalue())