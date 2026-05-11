"""
Verify that after unnormalizing the action stored in a robomimic-compatible HDF5
dataset, the relationship next_ee_pose ≈ cur_ee_pose + unnorm(action) holds in task space.

How actions are generated (MimicGen data generation)
--------------------------------------------------
See: mimicgen/datagen/waypoint.py execute(), mimicgen/env_interfaces/robosuite.py

1. At step t we have state[t] (stored) and waypoint target pose.
2. Current EE pose is read from the env (same as EE pose in state[t]).
3. action[t] = target_pose_to_action(waypoint.pose) = normalize(waypoint.pose - cur_ee_pose):
   - delta_pos = (target_pos - curr_pos);  normalized_pos = clip(delta_pos / max_dpos, -1, 1)
   - delta_rot (axis-angle) from (target_rot @ curr_rot.T);  normalized_rot = clip(delta_rot / max_drot, -1, 1)
   with max_dpos = controller.output_max[0], max_drot = controller.output_max[3] (scalars).
4. env.step(play_action) is called; then state[t+1] is the state after the step.
5. Stored transition: (states[t], actions[t], states[t+1]) with action[t] in [-1, 1].

So unnormalization must match action_to_target_pose: delta_pos = action[:3] * max_dpos,
delta_rot = action[3:6] * max_drot (same scalars). The controller then interprets this
as a goal and may not fully reach it in one step (interpolation/ramp), so tolerance
may be needed.

Usage:
  python -m mimicgen.scripts.verify_action_state_consistency --dataset datasets/core/square_d2.hdf5
  python -m mimicgen.scripts.verify_action_state_consistency --dataset /path/to/dataset.hdf5 --atol 1e-3 --max_demos 5
  # Plot action magnitude over time (expect decay toward waypoint, jump when switching waypoint):
  python -m mimicgen.scripts.verify_action_state_consistency --dataset datasets/core/square_d2.hdf5 --plot_action_magnitude --plot_output action_mag.png
"""
import argparse
import json
import sys

import h5py
import numpy as np

# Register MimicGen envs so robosuite.make can find them
try:
    import mimicgen
except ImportError:
    pass

import robosuite.utils.transform_utils as T
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

try:
    import matplotlib
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_action_magnitude(dataset_path, demos, control_dim, controller, output_path, max_demos_plot=10):
    """
    Plot action magnitude (L2 norm of arm action) over timesteps for each demo.
    When approaching the same waypoint over multiple steps, magnitude should tend to
    decrease; it may jump when switching to the next waypoint.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping action magnitude plot.")
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    demos_to_plot = demos[: min(len(demos), max_demos_plot)]
    max_dpos = float(controller.output_max[0])
    max_drot = float(controller.output_max[3])

    with h5py.File(dataset_path, "r") as f:
        for demo_key in demos_to_plot:
            actions = f["data/{}/actions".format(demo_key)][()]
            # Arm action magnitude (L2 norm of first control_dim dims, typically 6)
            mag = np.linalg.norm(actions[:, :3] * max_dpos, axis=1)
            # mag = np.linalg.norm(actions[:, 3:6] * max_drot, axis=1)
            ax.plot(np.arange(len(mag)), mag, alpha=0.7, label=demo_key)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Action magnitude (L2 norm of arm action)")
    ax.set_title("Action magnitude over time (smaller toward waypoint, jump at next waypoint)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.show()
    plt.close(fig)
    print("Saved action magnitude plot to {}".format(output_path))


def unnormalize_arm_action(action_arm, controller):
    """
    Unnormalize arm action from [-1, 1] to physical delta using the *exact* same
    formula as MimicGen env_interface action_to_target_pose() (robosuite.py):
    delta_position = action[:3] * max_dpos, delta_rotation = action[3:6] * max_drot,
    with max_dpos = controller.output_max[0], max_drot = controller.output_max[3].
    """
    max_dpos = float(controller.output_max[0])
    max_drot = float(controller.output_max[3])
    delta_pos = np.array(action_arm[:3], dtype=np.float64) * max_dpos
    delta_rot = np.array(action_arm[3:6], dtype=np.float64) * max_drot
    return np.concatenate([delta_pos, delta_rot])


def get_ee_pose_from_state(env, state_flattened):
    """
    Reset env to the given state using env.reset_to (same as convert_dataset.py),
    then return current end-effector pose.

    EE pose is read from the sim using the controller's eef_name site (same as
    MimicGen env_interface get_robot_eef_pose(): see mimicgen/env_interfaces/robosuite.py
    get_object_pose(obj_name=controller.eef_name, obj_type="site"), which uses
    sim.data.site_xpos / site_xmat). The controller's ee_pos/ee_ori_mat are
    populated the same way in robosuite's base_controller.update() (see
    base_controller.py lines 141-144). We read from the sim directly so we
    don't depend on the controller cache and match MimicGen's pattern.
    """
    env.reset_to({"states": state_flattened})
    inner = env.env if hasattr(env, "env") else env
    sim = inner.sim
    ctrl = inner.robots[0].controller
    site_id = sim.model.site_name2id(ctrl.eef_name)
    ee_pos = np.array(sim.data.site_xpos[site_id])
    ee_ori_mat = np.array(sim.data.site_xmat[site_id].reshape(3, 3))
    return ee_pos, ee_ori_mat


def orientation_error_from_matrices(rot_expected, rot_actual):
    """Rotation error in radians (angle of the difference rotation)."""
    diff = rot_expected.T @ rot_actual
    # angle = arccos((trace(R)-1)/2)
    trace = np.trace(diff)
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    return float(angle)


def verify_demo(env, states, actions, agent_pos, atol_pos, atol_ori_rad, controller, control_dim, debug_first=False):
    """
    Verify for one demo that next_ee_pose ≈ cur_ee_pose + unnorm_action (task space).
    Returns (passed, max_pos_error, max_ori_error_rad, n_checked).
    """
    n = len(states)
    if n < 2:
        return True, 0.0, 0.0, 0

    max_pos_err = 0.0
    max_ori_err = 0.0
    n_checked = 0

    for t in range(n - 1):
        cur_state = states[t]
        next_state = states[t + 1]
        action = actions[t]

        arm_action = np.array(action[:control_dim], dtype=np.float64)
        unnorm = unnormalize_arm_action(arm_action, controller)

        delta_pos = unnorm[:3]
        delta_axisangle = unnorm[3:6]

        cur_pos, cur_ori = get_ee_pose_from_state(env, cur_state)
        next_pos, next_ori = get_ee_pose_from_state(env, next_state)

        cur_pos_from_stored = agent_pos[t]
        next_pos_from_stored = agent_pos[t + 1]

        assert np.allclose(cur_pos_from_stored, cur_pos), f"cur_pos_from_stored: {cur_pos_from_stored} != cur_pos: {cur_pos}"
        assert np.allclose(next_pos_from_stored, next_pos), f"next_pos_from_stored: {next_pos_from_stored} != next_pos: {next_pos}"

        expected_next_pos = cur_pos + delta_pos
        pos_err = np.linalg.norm(next_pos - expected_next_pos)
        max_pos_err = max(max_pos_err, pos_err)

        if debug_first and t == 0:
            achieved_delta = next_pos - cur_pos
            print("[debug t=0] commanded delta_pos (unnorm): {}".format(delta_pos))
            print("[debug t=0] achieved delta_pos (next-cur): {}".format(achieved_delta))
            print("[debug t=0] cur_pos {} next_pos {}".format(cur_pos, next_pos))
            print("[debug t=0] pos_err {:.6f}".format(pos_err))

        delta_rot_mat = T.quat2mat(T.axisangle2quat(delta_axisangle))
        expected_next_ori = delta_rot_mat @ cur_ori
        ori_err = orientation_error_from_matrices(expected_next_ori, next_ori)
        max_ori_err = max(max_ori_err, ori_err)

        n_checked += 1

    passed = (max_pos_err <= atol_pos) and (max_ori_err <= atol_ori_rad)
    return passed, max_pos_err, max_ori_err, n_checked


def main():
    parser = argparse.ArgumentParser(
        description="Verify next_state ≈ cur_state + unnormalized(action) in task space."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/core/square_d2.hdf5",
        help="Path to HDF5 dataset",
    )
    parser.add_argument(
        "--atol_pos",
        type=float,
        default=2e-3,
        help="Absolute tolerance for position (meters)",
    )
    parser.add_argument(
        "--atol_ori",
        type=float,
        default=5e-2,
        help="Absolute tolerance for orientation (radians)",
    )
    parser.add_argument(
        "--max_demos",
        type=int,
        default=None,
        help="Max number of demos to check (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-demo errors",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print first transition commanded vs achieved delta (for debugging)",
    )
    parser.add_argument(
        "--plot_action_magnitude",
        action="store_true",
        help="Plot L2 norm of arm action over timesteps (expect decay toward waypoint, jump at next)",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default="action_magnitude.png",
        help="Path for action magnitude plot (default: action_magnitude.png)",
    )
    parser.add_argument(
        "--plot_max_demos",
        type=int,
        default=10,
        help="Max number of demos to show in action magnitude plot (default: 10)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset

    print("Loading env_meta from {}...".format(dataset_path))
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    if not EnvUtils.is_robosuite_env(env_meta=env_meta):
        print("This script supports robosuite datasets only.")
        sys.exit(1)

    print("Creating environment (env_name={})...".format(env_meta["env_name"]))
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )
    # Use reset_to() on the (possibly wrapped) env; controller lives on inner robosuite env
    inner = env.env if hasattr(env, "env") else env
    controller = inner.robots[0].controller
    control_dim = controller.control_dim
    print(controller)
    print("controller type: ", type(controller))
    print("controller control_dim: ", controller.control_dim)

    with h5py.File(dataset_path, "r") as f:
        demos = [k for k in f["data"].keys() if k.startswith("demo_")]
        demos.sort(key=lambda x: int(x.split("_")[1]))

    if args.max_demos is not None:
        demos = demos[: args.max_demos]

    if args.plot_action_magnitude:
        plot_action_magnitude(
            dataset_path,
            demos,
            control_dim,
            controller,
            output_path=args.plot_output,
            max_demos_plot=args.plot_max_demos,
        )

    print("Checking {} demos (atol_pos={}, atol_ori={} rad)...".format(
        len(demos), args.atol_pos, args.atol_ori
    ))

    all_passed = True
    for demo_key in demos:
        with h5py.File(dataset_path, "r") as f:
            states = f["data/{}/states".format(demo_key)][()]
            actions = f["data/{}/actions".format(demo_key)][()]
            # Load into memory; don't keep HDF5 dataset ref (file closes after with block)
            agent_pos = f["data/{}/obs/robot0_eef_pos".format(demo_key)][()]

        print("agent_pos shape: ", agent_pos.shape)

        passed, max_pos_err, max_ori_err, n_checked = verify_demo(
            env, states, actions, agent_pos,
            atol_pos=args.atol_pos,
            atol_ori_rad=args.atol_ori,
            controller=controller,
            control_dim=control_dim,
            debug_first=args.debug,
        )
        if not passed:
            all_passed = False
        if args.verbose or not passed:
            print("  {}: n_checked={} max_pos_err={:.6f} m max_ori_err={:.6f} rad  {}".format(
                demo_key, n_checked, max_pos_err, max_ori_err, "PASS" if passed else "FAIL"
            ))

    if all_passed:
        print("\nPASSED: All checked transitions satisfy next_ee_pose ≈ cur_ee_pose + unnorm(action).")
    else:
        print("\nFAILED: Some transitions exceeded tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
