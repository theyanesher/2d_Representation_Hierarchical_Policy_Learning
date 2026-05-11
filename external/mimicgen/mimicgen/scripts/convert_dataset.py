"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

This is the parallelized version of the SMITH-style converter.
The output format (per-timestep .npz files under --output_dir) is identical to the
legacy version. Speedups come from:
  1. Building the simulator env ONCE per worker process (Pool initializer), not per demo.
  2. Workers write their own .npz files — eliminates pickling huge image arrays back to parent.
  3. pool.imap_unordered for streaming progress (no lockstep batches).

Args:
    input (str): path to input hdf5 dataset
    output_dir (str): directory to save per-timestep npz files
    n (int): if provided, stop after n trajectories are processed
    pool_size (int): number of parallel worker processes (recommended: 2 per GPU)
    num_workers (int): legacy alias for pool_size when pool_size not provided
    use_bayesian_decomp (bool): use BOCPD-based subgoal decomposition
    bocpd_config (str): path to BOCPD hyperparameter yaml
    camera_height/camera_width (int): rendered observation size

Example usage:
    python convert_dataset.py --input demo.hdf5 --output_dir out/ \
        --camera_height 256 --camera_width 256 \
        --pool_size 2 \
        --use_bayesian_decomp --bocpd_config third_party/robogen/bocpd_config.yaml
"""
import os
import json
import time
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
from third_party.robogen.robogen_utils import (
    compute_new_goal_gripper_pcd
)
from third_party.robogen.subgoal_decomp import compute_subgoal_gripper_pcd
from third_party.robogen.bayesian_subgoal_decomp import compute_bayesian_subgoal_gripper_pcd

try:
    from robosuite.utils import transform_utils as T
except ImportError:
    T = None
multiprocessing.set_start_method('spawn', force=True)


# ---------------------------------------------------------------------------
# Worker-process globals — populated once per child by _init_worker.
# ---------------------------------------------------------------------------
_WORKER_ENV = None
_WORKER_ENV_META = None
_WORKER_ARGS = None


def _camera_names_for(env_meta):
    if env_meta['env_name'].startswith('PickPlace'):
        return ['birdview', 'agentview', 'robot0_eye_in_hand']
    return ['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']


def _init_worker(env_meta, args):
    """Build the simulator env ONCE per worker process and stash in module globals."""
    global _WORKER_ENV, _WORKER_ENV_META, _WORKER_ARGS
    camera_names = _camera_names_for(env_meta)
    _WORKER_ENV = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
        save_image_keys=['agentview', 'robot0_eye_in_hand'],
    )
    _WORKER_ENV_META = env_meta
    _WORKER_ARGS = args


def extract_trajectory(
    env,
    env_meta,
    args,
    initial_state,
    states,
    actions,
    robot0_gripper_qpos,
):
    """
    Replay a demo's logged states in `env`, gather observations + subgoal labels,
    and reformat actions. Pure function — does NOT create or destroy `env`.
    """
    done_mode = args.done_mode
    assert states.shape[0] == actions.shape[0]

    env.reset()
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
    for t in range(1, traj_len + 1):
        if t == traj_len:
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            next_obs = env.reset_to({"states": states[t]})

        r = env.get_reward()

        done = False
        if (done_mode == 1) or (done_mode == 2):
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            done = done or env.is_success()["task"]
        done = int(done)

        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        obs = deepcopy(next_obs)

    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    np_gripper_pcd = np.asarray(traj["obs"]["gripper_pcd"])

    if args.gripper_only_subgoal:
        goal_gripper_pcd, switch_idxs = compute_new_goal_gripper_pcd(
            np_gripper_pcd, robot0_gripper_qpos, actions, return_switch_idxs=True)
    elif getattr(args, 'use_bayesian_decomp', False):
        eef_vel_lin = np.asarray(traj["obs"]["robot0_eef_vel_lin"])
        eef_pos     = np.asarray(traj["obs"]["robot0_eef_pos"])
        eef_quat    = np.asarray(traj["obs"]["robot0_eef_quat"])
        goal_gripper_pcd, switch_idxs = compute_bayesian_subgoal_gripper_pcd(
            gripper_pcd=np_gripper_pcd,
            eef_qpos=robot0_gripper_qpos,
            actions=actions,
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            eef_vel_lin=eef_vel_lin,
            config=args.bocpd_config_dict,
            return_switch_idxs=True,
        )
    else:
        eef_vel_lin = np.asarray(traj["obs"]["robot0_eef_vel_lin"])
        goal_gripper_pcd, switch_idxs = compute_subgoal_gripper_pcd(
            gripper_pcd=np_gripper_pcd,
            eef_qpos=robot0_gripper_qpos,
            actions=actions,
            eef_vel_lin=eef_vel_lin,
            curvature_threshold=args.curvature_threshold,
            min_segment_len=args.min_segment_len,
            warmup_steps=args.warmup_steps,
            return_switch_idxs=True,
        )
    traj["obs"]["goal_gripper_pcd"] = goal_gripper_pcd

    # Unnormalize actions; recompute rotation as gripper-local 6D.
    if T is not None and EnvUtils.is_robosuite_env(env_meta=env_meta):
        inner = env.env if hasattr(env, "env") else env
        controller = inner.robots[0].controller
        max_dpos = float(controller.output_max[0])
        max_drot = float(controller.output_max[3])
        action_traj_len = actions.shape[0]
        action_arrays = []
        for t in range(action_traj_len):
            delta_pos = np.array(actions[t, :3], dtype=np.float64) * max_dpos
            delta_rot_axisangle = np.array(actions[t, 3:6], dtype=np.float64) * max_drot
            delta_rot = T.quat2mat(T.axisangle2quat(delta_rot_axisangle))
            cur_rot = T.quat2mat(traj["obs"]["robot0_eef_quat"][t])
            waypoint_rot = delta_rot @ cur_rot
            delta_local_rot = cur_rot.T @ waypoint_rot
            delta_rot_6d = rotation_transfer_matrix_to_6D(delta_local_rot).reshape(-1)
            gripper = actions[t, -1]
            gripper *= -0.01  # MimicGen +1=close, -1=open → flip + scale to Panda gripper-velocity
            action_arrays.append(np.concatenate([delta_pos, delta_rot_6d, [gripper]]))
        traj["actions"] = np.array(action_arrays)

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def _write_traj_to_npz(traj, output_dir, ep_name):
    """Write per-timestep compressed .npz files. Same key set & compression as legacy."""
    traj_dir = os.path.join(output_dir, ep_name)
    os.makedirs(traj_dir, exist_ok=True)
    n_steps = len(traj["actions"])
    for t_idx in range(n_steps):
        step_path = os.path.join(traj_dir, "{}.npz".format(t_idx))
        np.savez_compressed(
            step_path,
            # high-level policy keys
            state=traj["obs"]["state"][t_idx][None, :],
            point_cloud=traj["obs"]["point_cloud"][t_idx][None, :],
            action=traj["actions"][t_idx][None, :],
            gripper_pcd=traj["obs"]["gripper_pcd"][t_idx][None, :],
            goal_gripper_pcd=traj["obs"]["goal_gripper_pcd"][t_idx][None, :],
            # low-level policy keys
            rgb_agentview=traj["obs"]["agentview_image"][t_idx][None, :],
            depth_agentview=traj["obs"]["agentview_depth"][t_idx][None, :],
            rgb_wrist=traj["obs"]["robot0_eye_in_hand_image"][t_idx][None, :],
            depth_wrist=traj["obs"]["robot0_eye_in_hand_depth"][t_idx][None, :],
            agentview_extrinsics=traj["obs"]["agentview_extrinsics"][t_idx][None, :],
            agentview_intrinsics=traj["obs"]["agentview_intrinsics"][t_idx][None, :],
            wrist_extrinsics=traj["obs"]["robot0_eye_in_hand_extrinsics"][t_idx][None, :],
            wrist_intrinsics=traj["obs"]["robot0_eye_in_hand_intrinsics"][t_idx][None, :],
            eef_pos=traj["obs"]["robot0_eef_pos"][t_idx][None, :],
            eef_quat=traj["obs"]["robot0_eef_quat"][t_idx][None, :],
            gripper_qpos=traj["obs"]["robot0_gripper_qpos"][t_idx][None, :],
        )
    return n_steps


def _worker_run(task):
    """Process one demo end-to-end inside a worker process. Returns small tuple only."""
    ep_name, initial_state, states, actions, robot0_gripper_qpos = task
    t0 = time.time()
    traj = extract_trajectory(
        env=_WORKER_ENV,
        env_meta=_WORKER_ENV_META,
        args=_WORKER_ARGS,
        initial_state=initial_state,
        states=states,
        actions=actions,
        robot0_gripper_qpos=robot0_gripper_qpos,
    )
    t_extract = time.time() - t0
    n_steps = _write_traj_to_npz(traj, _WORKER_ARGS.output_dir, ep_name)
    t_total = time.time() - t0
    return ep_name, n_steps, t_extract, t_total


def dataset_states_to_obs(args):
    pool_size = getattr(args, 'pool_size', None) or args.num_workers

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.input)
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env_meta, indent=4))
    print("")
    print("input file: {}".format(args.input))

    # Enumerate demos
    f = h5py.File(args.input, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    if args.n is not None:
        demos = demos[:args.n]

    # Resume: classify each demo by output-dir state.
    # A demo is "complete" iff the LAST expected step file exists.
    # Checking only 0.npz (the legacy behavior) wrongly marks crashed-mid-demo
    # outputs as complete; we read the expected length from the source HDF5
    # (states.shape[0] is deterministic) and probe <n-1>.npz instead.
    if getattr(args, 'output_dir', None) is not None:
        complete = []
        partial = []
        fresh = []
        for ep in demos:
            n_expected = int(f["data/{}/states".format(ep)].shape[0])
            ep_dir = os.path.join(args.output_dir, ep)
            if n_expected <= 0:
                fresh.append(ep)
                continue
            last_path = os.path.join(ep_dir, "{}.npz".format(n_expected - 1))
            if os.path.exists(last_path):
                complete.append(ep)
            elif os.path.exists(os.path.join(ep_dir, "0.npz")):
                partial.append(ep)
            else:
                fresh.append(ep)
        demos = partial + fresh  # re-run partial demos first; they overwrite step files in-place
        print("Resume: {} complete (skip), {} partial (re-run), {} fresh — total to do: {}".format(
            len(complete), len(partial), len(fresh), len(demos)))
        if args.list_partial and partial:
            print("  partial demos to re-run: {}".format(partial[:20] + (["..."] if len(partial) > 20 else [])))

    # Pre-load all task tuples (states + actions are small; closes the HDF5 before fork-out)
    tasks = []
    for ep in demos:
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        actions = f["data/{}/actions".format(ep)][()]
        robot0_gripper_qpos = f["data/{}/obs/robot0_gripper_qpos".format(ep)][()]
        tasks.append((ep, initial_state, states, actions, robot0_gripper_qpos))
    f.close()

    if not tasks:
        print("Nothing to do.")
        return

    print("[parallel] pool_size={} | demos to process: {}".format(pool_size, len(tasks)))

    from tqdm import tqdm
    total_samples = 0
    extract_times = []
    total_times = []

    with multiprocessing.Pool(
        processes=pool_size,
        initializer=_init_worker,
        initargs=(env_meta, args),
    ) as pool:
        for ep_name, n_steps, t_extract, t_total in tqdm(
            pool.imap_unordered(_worker_run, tasks),
            total=len(tasks),
            desc="demos",
            unit="demo",
        ):
            total_samples += n_steps
            extract_times.append(t_extract)
            total_times.append(t_total)
            tqdm.write("[done] {}: {} steps | extract {:.1f}s | total {:.1f}s".format(
                ep_name, n_steps, t_extract, t_total))

    if total_times:
        print("Wrote {} total samples across {} demos.".format(total_samples, len(tasks)))
        print("Per-demo wall-time: mean={:.1f}s, p50={:.1f}s, p95={:.1f}s, max={:.1f}s".format(
            float(np.mean(total_times)),
            float(np.percentile(total_times, 50)),
            float(np.percentile(total_times, 95)),
            float(np.max(total_times)),
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="path to input hdf5 dataset")
    parser.add_argument("--n", type=int, default=None,
                        help="(optional) stop after n trajectories are processed")

    # Parallelism. pool_size is the real knob; num_workers kept as a legacy alias.
    parser.add_argument("--num_workers", type=int, default=2,
                        help="legacy alias for --pool_size when --pool_size is not set")
    parser.add_argument("--pool_size", type=int, default=None,
                        help="number of parallel worker processes (recommended: 2 per GPU)")

    parser.add_argument("--shaped", action='store_true',
                        help="(optional) use shaped rewards")
    parser.add_argument("--camera_names", type=str, nargs='+', default=[],
                        help="(unused — camera set is fixed by env type)")
    parser.add_argument("--camera_height", type=int, default=512,
                        help="(optional) height of image observations")
    parser.add_argument("--camera_width", type=int, default=512,
                        help="(optional) width of image observations")
    parser.add_argument("--done_mode", type=int, default=2,
                        help="0: done at success state. 1: done at end. 2: both.")
    parser.add_argument("--copy_rewards", action='store_true',
                        help="(unused in npz output mode; kept for CLI compatibility)")
    parser.add_argument("--copy_dones", action='store_true',
                        help="(unused in npz output mode; kept for CLI compatibility)")
    parser.add_argument("--exclude-next-obs", type=bool, default=True,
                        help="(unused in npz output mode; kept for CLI compatibility)")
    parser.add_argument("--compress", type=bool, default=True,
                        help="(unused in npz output mode; npz is always compressed)")

    # Subgoal decomposition options
    parser.add_argument("--use_bayesian_decomp", action='store_true',
                        help="use BOCPD-based subgoal decomposition")
    parser.add_argument("--bocpd_config", type=str,
                        default="third_party/robogen/bocpd_config.yaml",
                        help="path to BOCPD hyperparameter yaml (with --use_bayesian_decomp)")
    parser.add_argument("--gripper_only_subgoal", action='store_true',
                        help="use original gripper open/close only subgoal decomposition")
    parser.add_argument("--curvature_threshold", type=float, default=0.5)
    parser.add_argument("--min_segment_len", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=20)

    parser.add_argument("--output_dir", type=str, default=None,
                        help="(required) directory to save per-timestep npz files")
    parser.add_argument("--list_partial", action='store_true',
                        help="print the names of demos being re-run because of incomplete output")

    args = parser.parse_args()

    if args.use_bayesian_decomp:
        import yaml
        with open(args.bocpd_config, 'r') as _f:
            args.bocpd_config_dict = yaml.safe_load(_f)
        print("[BOCPD] loaded config from {}: {}".format(args.bocpd_config, args.bocpd_config_dict))
    else:
        args.bocpd_config_dict = None

    if args.output_dir is None:
        raise SystemExit("--output_dir is required (npz output mode).")

    dataset_states_to_obs(args)
