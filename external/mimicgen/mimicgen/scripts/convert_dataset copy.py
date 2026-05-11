"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # (space saving option) extract 84x84 image observations with compression and without 
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
import multiprocessing
from mimicgen.utils.articubot_util import rotation_transfer_matrix_to_6D
from articubot_on_mimicgen.third_party.robogen.robogen_utils import (
    compute_new_goal_gripper_pcd
)

try:
    from robosuite.utils import transform_utils as T
except ImportError:
    T = None
multiprocessing.set_start_method('spawn', force=True)


def extract_trajectory(
    env_meta,
    args, 
    initial_state, 
    states, 
    actions,
    robot0_gripper_qpos,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    done_mode = args.done_mode
    if env_meta['env_name'].startswith('PickPlace'):
        camera_names=['birdview', 'agentview', 'robot0_eye_in_hand']
    else:
        camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'agentview_full', 'robot0_robotview', 'robot0_eye_in_hand'], 
        camera_names=camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    # import pdb; pdb.set_trace()
    env.reset()
    # import pdb; pdb.set_trace()
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):
        # print(f"Processing timestep {t} of {traj_len}")

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    np_gripper_pcd = np.asarray(traj["obs"]["gripper_pcd"])
    gripper_action = actions[..., -1]
    # switch_indices = list(np.where(np.sign(gripper_action[:-1]) != np.sign(gripper_action[1:]))[0])
    # switch_indices.append(len(gripper_action) - 1)
    # switch_indices = np.array(switch_indices, dtype=np.int16)
    # repeat_count = np.insert(np.diff(switch_indices), 0, switch_indices[0])
    # repeat_count[-1] += 1
    # goal_gripper_pcd = np_gripper_pcd[switch_indices]
    # goal_gripper_pcd = np.repeat(goal_gripper_pcd, repeat_count, axis=0)
    # traj["obs"]["goal_gripper_pcd"] = goal_gripper_pcd

    goal_gripper_pcd, switch_idxs = compute_new_goal_gripper_pcd(np_gripper_pcd, 
                                                                 robot0_gripper_qpos,
                                                                 actions,
                                                                 return_switch_idxs=True)
    traj["obs"]["goal_gripper_pcd"] = goal_gripper_pcd
    # import pdb; pdb.set_trace()
    

    # Unnormalize actions and recompute rotation as 6D in gripper local frame (waypoint = delta_rotation * cur_orientation, then delta_local s.t. waypoint = cur * delta_local)
    if T is not None and EnvUtils.is_robosuite_env(env_meta=env_meta):
        inner = env.env if hasattr(env, "env") else env
        controller = inner.robots[0].controller
        max_dpos = float(controller.output_max[0])
        max_drot = float(controller.output_max[3])
        traj_len = actions.shape[0]
        action_arrays = []
        for t in range(traj_len):
            # Unnormalize translation
            delta_pos = np.array(actions[t, :3], dtype=np.float64) * max_dpos
            # Compute waypoint from stored action: waypoint_rot = delta_rotation @ cur_rot (world-frame delta)
            delta_rot_axisangle = np.array(actions[t, 3:6], dtype=np.float64) * max_drot
            delta_rot = T.quat2mat(T.axisangle2quat(delta_rot_axisangle))
            cur_rot = T.quat2mat(traj["obs"]["robot0_eef_quat"][t])
            waypoint_rot = delta_rot @ cur_rot
            # Local delta: waypoint = cur @ delta_local  =>  delta_local_rot = cur_rot.T @ waypoint_rot
            delta_local_rot = cur_rot.T @ waypoint_rot
            delta_rot_6d = rotation_transfer_matrix_to_6D(delta_local_rot).reshape(-1)
            gripper = actions[t, -1]
            gripper *= -0.01 ### this is the panda gripper speed. In mimicgen it is 1 for closing and -1 for opening so we need to flip it. 
            action_arrays.append(np.concatenate([delta_pos, delta_rot_6d, [gripper]]))

        traj["actions"] = np.array(action_arrays)

    gripper_actions = traj["actions"][:, -1]
    ### plot the gripper actions
    # import matplotlib.pyplot as plt
    # plt.plot(gripper_actions)
    # plt.show()

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj

def worker(x):
    env_meta, args, initial_state, states, actions, robot0_gripper_qpos, j = x
    print("extracing trajectory {}".format(j))
    traj = extract_trajectory(
        env_meta=env_meta,
        args=args,
        initial_state=initial_state, 
        states=states, 
        actions=actions,
        robot0_gripper_qpos=robot0_gripper_qpos,
    )
    return traj

def dataset_states_to_obs(args):
    num_workers = args.num_workers
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.input)
    if env_meta['env_name'].startswith('PickPlace'):
        camera_names=['birdview', 'agentview', 'robot0_eye_in_hand']
    else:
        camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'agentview_full', 'robot0_robotview', 'robot0_eye_in_hand'], 
        camera_names=camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.input, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]
    # demos = demos[50: 100]

    # output file in same directory as input file
    print("input file: {}".format(args.input))

    total_samples = 0
    for i in range(0, len(demos), num_workers):
        end = min(i + num_workers, len(demos))
        initial_state_list = []
        states_list = []
        actions_list = []
        robot0_gripper_qpos_list = []
        for j in range(i, end):
            ep = demos[j]
            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            actions = f["data/{}/actions".format(ep)][()]
            robot0_gripper_qpos = f["data/{}/obs/robot0_gripper_qpos".format(ep)][()]

            initial_state_list.append(initial_state)
            states_list.append(states)
            actions_list.append(actions)
            robot0_gripper_qpos_list.append(robot0_gripper_qpos)

        # parallel version
        with multiprocessing.Pool(num_workers) as pool:
            trajs = pool.map(worker, [[env_meta, args, initial_state_list[j], states_list[j], actions_list[j], robot0_gripper_qpos_list[j], j + i] for j in range(len(initial_state_list))]) 

        # non-parallel versions
        # trajs = []
        # for j in range(len(initial_state_list)):
        #     trajs.append(extract_trajectory(
        #         env_meta=env_meta,
        #         args=args,
        #         initial_state=initial_state_list[j], 
        #         states=states_list[j], 
        #         actions=actions_list[j],
        #         robot0_gripper_qpos=robot0_gripper_qpos_list[j],
        #     ))


        for j, ind in enumerate(range(i, end)):
            ep = demos[ind]
            traj = trajs[j]
            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            # optional: write per-timestep npz (all keys from traj["obs"] + action)
            if getattr(args, "output_dir", None) is not None:
                traj_dir = os.path.join(args.output_dir, ep)
                os.makedirs(traj_dir, exist_ok=True)
                n_steps = len(traj["actions"])
                for t_idx in range(n_steps):
                    step_path = os.path.join(traj_dir, "{}.npz".format(t_idx))
                    np.savez_compressed(
                        step_path,
                        state=traj["obs"]["state"][t_idx][None, :],
                        point_cloud=traj["obs"]["point_cloud"][t_idx][None, :],
                        action=traj["actions"][t_idx][None, :],
                        gripper_pcd=traj["obs"]["gripper_pcd"][t_idx][None, :],
                        goal_gripper_pcd=traj["obs"]["goal_gripper_pcd"][t_idx][None, :],
                    )

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            # ep_data_grp = data_grp.create_group(ep)
            # ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            # ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            # ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            # ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            # for k in traj["obs"]:
            #     if args.compress:
            #         ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
            #     else:
            #         ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            #     if not args.exclude_next_obs:
            #         if args.compress:
            #             ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
            #         else:
            #             ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # # episode metadata
            # if is_robosuite_env:
            #     ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            # ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
            # print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))
        
        del trajs

    # # copy over all filter keys that exist in the original hdf5
    # if "mask" in f:
    #     f.copy("mask", f_out)

    # global metadata
    # data_grp.attrs["total"] = total_samples
    # data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    # print("Wrote {} trajectories to {}".format(len(demos), output_path))

    # f.close()
    # f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    
    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=512,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=2,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        type=bool,
        default=True,
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        type=bool,
        default=True,
        help="(optional) compress observations with gzip option in hdf5",
    )

    # optional directory to write per-timestep npz files (one subdir per episode, 0.npz, 1.npz, ...)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="(optional) directory to save per-timestep npz files; keys: state, point_cloud, action, gripper_pcd, goal_gripper_pcd",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)