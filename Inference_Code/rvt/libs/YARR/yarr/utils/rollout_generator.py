# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult

class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0'):
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False, timesteps_low_level = 4):
        # import pdb;pdb.set_trace();
        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()
        agent.reset()
        # import pdb; pdb.set_trace();
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

        # obs_history_low_level = {k: [np.array(v, dtype=self._get_type(v))] * timesteps_low_level for k, v in obs.items()}
        obs_history_low_level = {
            k: [np.array(v, dtype=self._get_type(v)).copy() for _ in range(timesteps_low_level)]
            for k, v in obs.items()
        }


        for step in range(episode_length): # 
            # import pdb; pdb.set_trace();
            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            # import pdb; pdb.set_trace();
            if not replay_ground_truth:
                act_result = agent.act(step_signal.value, prepped_data,
                                    deterministic=eval, low_level_obs_dict = obs_history_low_level, episodes = eval_demo_seed, timestep = step) # , scene_points, pos_debug, ori_debug, gripper_open_close PUT IN AGAIN WHEN RUNNING OUR POLICY
                debug = False
                if debug:
                    import pickle
                    # Example point clouds
                    # scene = torch.randn(1000, 3)       # Replace with your actual point cloud
                    # my_pred = torch.randn(1000, 3)
                    # their_pred = torch.randn(1000, 3)
                    # import pdb; pdb.set_trace();
                    from rvt.models.robogen_utils import get_4_points_from_gripper_pos_orient_torch
                    gripper_open_close[gripper_open_close == 1] = 0.04
                    gripper_points_arm = get_4_points_from_gripper_pos_orient_torch(gripper_pos = pos_debug, gripper_quat = torch.tensor(ori_debug), cur_joint_angle = gripper_open_close)
                    # step = 0 
                    # Create the dictionary
                    data = {
                        "scene": scene_points.cpu().squeeze().numpy(),
                        "my_pred": gripper_points_arm.cpu().squeeze().numpy(),
                        # "their_pred": pred_wpt
                    }

                    # Save to pickle file
                    with open(f"/home/pratik_final/Downloads/Bimanual/RVT2_GMM_Inference/Final_Push/Bimanual_Manipulation/runs/rvt_with_heatmap/eval/test/1/pointclouds_{step}.pkl", "wb") as f:
                        pickle.dump(data, f)
            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            # import pdb; pdb.set_trace();
            transition = env.step(act_result, obs_dict = obs_history_low_level)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)
            # import pdb; pdb.set_trace();
            for k in obs_history_low_level.keys():
                if k in ["gripper_pose_low_level", "reverse_trans_low_level", "gripper_open_low_level", "pcd_low_level", "pointcloud_low_level", "rgb_low_level", "depth_low_level", "heatmap_low_level", "lang_goal_tokens"]:
                    continue
                # import pdb; pdb.set_trace();
                obs_history_low_level[k].append(np.array(transition.observation[k]).copy())
                obs_history_low_level[k].pop(0)
            # import pdb; pdb.set_trace();
            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition
            # import pdb; pdb.set_trace();
            if transition.info.get("needs_reset", transition.terminal):
                return
