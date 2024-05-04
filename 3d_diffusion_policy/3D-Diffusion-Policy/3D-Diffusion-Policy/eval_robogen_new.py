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
import time

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
    # checkpoint_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/vary_robot_init_joint_near_handle_after_reaching/2024.04.20/15.48.13_train_dp3_robogen_open_door/checkpoints/latest.ckpt"
    checkpoint_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0502-vary-obj-init-angle-robot-init-joint-near-handle-larger/2024.05.03/01.50.11_train_dp3_robogen_open_door/checkpoints/epoch-2700-test_mean_score=0.767.ckpt"
    workspace.load_checkpoint(path=checkpoint_dir)

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()

    experiment_folder = "{}/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle".format(os.environ['PROJECT_DIR'])
    experiment_name = "0502-vary-obj-init-angle-robot-init-joint-near-handle-larger"
    experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
    all_experiments = os.listdir(experiment_path)
    all_experiments = sorted(all_experiments)
    
    after_reaching_after_reaching_init_state_files = []
    init_state_files = []
    config_files = []
    for experiment in all_experiments:
        cprint("experiment: {}".format(experiment), "red")
        first_step_folder = "grasp_the_door_handle_primitive"
        first_stage_states_path = os.path.join(experiment_path, experiment, first_step_folder, "states")
        
        if os.path.exists(os.path.join(experiment_path, experiment, first_step_folder, "label.json")):
            with open(os.path.join(experiment_path, experiment, first_step_folder, "label.json"), 'r') as f:
                label = json.load(f)
            if not label['good_traj']: continue
        
        expert_opened_angle_file = os.path.join(experiment_path, experiment, first_step_folder, "opened_angle.txt")
        with open(expert_opened_angle_file, "r") as f:
            expert_opened_angle = f.readlines()
            expert_opened_angle = float(expert_opened_angle[0].lstrip().rstrip())
        if expert_opened_angle < 0.1:
            continue
        
        stage_lengths = os.path.join(experiment_path, experiment, first_step_folder, "stage_lengths.json")
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)
            
        reaching_phase = stage_lengths['reach_handle']
        after_reaching_init_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(reaching_phase))
        after_reaching_after_reaching_init_state_files.append(after_reaching_init_state_file)
        init_state_file = os.path.join(first_stage_states_path, "state_0.pkl")
        init_state_files.append(init_state_file)
        config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
        config_files.append(config_file)
    
    starting_from_pregrasp = False

    opened_joint_angles = {}
    for idx in tqdm.tqdm(range(1, len(config_files))):
        print("config path: ", config_files[idx])
        config_path = config_files[idx]
        env, _ = build_up_env(
            config_files[idx],
            "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle",
            "grasp_the_door_handle",
            after_reaching_after_reaching_init_state_files[idx] if starting_from_pregrasp else init_state_files[idx],
            render=False, 
            randomize=False,
            obj_id=0,
            horizon=400,
        )
        
        in_gripper_frame = False
        
        object_name = "StorageFurniture"
        env.reset()
        pointcloud_env = RobogenPointCloudWrapper(env, object_name, 
                                        in_gripper_frame=in_gripper_frame, 
                                        gripper_num_points=0)
        env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                            max_episode_steps=600, reward_agg_method='sum')
        obs = env.reset()
        policy.reset()
        final_rgbs = []
        episode_reward = 0
        horizon = 60 if starting_from_pregrasp else 120
        for _ in tqdm.tqdm(range(horizon)):
            np_obs_dict = dict(obs)
            # change the point cloud to be in the gripper frame
            # np_obs_dict['point_cloud'] = pointcloud_env._transfer_point_cloud_to_gripper_frame(np_obs_dict['point_cloud'])
            obs_dict = dict_apply(np_obs_dict,
                                        lambda x: torch.from_numpy(x).to('cuda'))
            # run policy
            beg = time.time()
            with torch.no_grad():
                obs_dict_input = {}  # flush unused keys
                obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                action_dict = policy.predict_action(obs_dict_input)
            end = time.time()
            # cprint("policy time: {}".format(end - beg), "blue")
            np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
            action = np_action_dict['action'].squeeze(0)
            beg = time.time()
            obs, reward, done, info = env.step(action)
            end = time.time()
            # cprint("step time: {}".format(end - beg), "blue")
            done = np.all(done)
            episode_reward += reward
            beg = time.time()
            final_rgbs.append(env.env.render())
            end = time.time()
            # cprint("render time: {}".format(end - beg), "blue")
            if done:
                break

        joint_angle = float(info["improved_joint_angle"][-1])
        cprint(f"improved joint angle: {joint_angle}", "blue")
        max_improvement = expert_opened_angle - env.env._env.init_joint_angle
        pointcloud_env._env.close()
        opened_joint_angles[config_path] = [joint_angle, max_improvement, joint_angle / max_improvement]
    
        checkpoint_name_start_idx = checkpoint_dir.find("3D-Diffusion-Policy/data/")  + len("3D-Diffusion-Policy/data/")
        save_path = "data/eval_results/{}".format(checkpoint_dir[checkpoint_name_start_idx:].replace("/", "_"))
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