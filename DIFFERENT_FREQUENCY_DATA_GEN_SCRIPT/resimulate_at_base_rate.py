#!/usr/bin/env python3
"""
STEP 1 of the sampling-rate ablation: re-simulate a 20 Hz MimicGen core dataset
at a higher BASE control frequency, producing states only (no rendering).

WHY THIS SCRIPT EXISTS
----------------------
convert_dataset.py cannot change the sampling rate. It never steps physics --
for every timestep it calls env.reset_to({"states": states[t]}), i.e. teleports
MuJoCo to a logged state and renders it. The source hdf5 therefore fixes the
rate: a 20 Hz demo holds 20 Hz states and no information in between.

Higher-rate data has to be produced by actually simulating. That is this script.
It writes a NEW core hdf5 in exactly the format convert_dataset.py consumes, so
everything downstream is unchanged.

Note we are not inventing data. MuJoCo already integrates at 500 Hz
(macros.SIMULATION_TIMESTEP = 0.002) no matter the control rate; at 20 Hz
robosuite runs 25 physics sub-steps per action and hands back only the state
after all 25. Re-simulating at 500 Hz records every sub-step instead of every
25th. The fine-grained states were always being computed -- just discarded.

METHOD: waypoint following
--------------------------
The stored actions are OSC_POSE *deltas* relative to the current pose, so they
cannot be repeated or subdivided: re-issuing the same delta k times moves the
arm k times too far, because each control tick re-applies "move d from wherever
you are now".

So for each source action we reconstruct the ABSOLUTE waypoint it was asking
for, then drive to that waypoint over k ticks at the higher rate, recomputing
the residual every tick:

    target_pos = cur_pos + a[:3] * max_dpos
    target_rot = R(a[3:6] * max_drot) @ cur_rot     <- convert_dataset.py:181-183
                                                       convention, kept verbatim
    repeat k times:
        cmd_pos  = clip((target_pos - cur_pos) / max_dpos, -1, 1)
        cmd_rot  = clip(axisangle(target_rot @ cur_rot.T) / max_drot, -1, 1)
        cmd_grip = a[-1]            # a command, not a delta: held across ticks
        env.step([cmd_pos, cmd_rot, cmd_grip])

Each source control step spanned 1/source_freq seconds and the k ticks span the
same wall-clock, so the arm converges to the same waypoint. Episode duration is
identical; only the observation/action rate changes.

RATE CONSTRAINTS
----------------
robosuite computes sub-steps as int(control_timestep / model_timestep) with
model_timestep = 0.002 s. That truncation means control_freq MUST divide 500
exactly or you silently get a different rate:

     40 Hz -> int(0.0250/0.002) = 12 -> 41.67 Hz   WRONG
     60 Hz -> int(0.0167/0.002) =  8 -> 62.50 Hz   WRONG
    200 Hz -> int(0.0050/0.002) =  2 -> 250.0 Hz   WRONG
    500 Hz -> int(0.0020/0.002) =  1 -> 500 Hz     ok (1 substep: no settling)

Valid: 500, 250, 125, 100, 50, 25, 20, 10, 5, 4, 2, 1. 500 Hz is the ceiling.
Since 500 = 2^2 * 5^3, any rate needing 2^3 or a factor of 3 is unreachable.

The base must also be an integer multiple of every target rate you intend to
stride down to. For the ladder {5, 10, 20, 50, 100, 250} the LCM is 500, so the
base must be 500 Hz -- a 250 Hz base cannot produce the 20 Hz arm (250/20=12.5).

OUTPUT (drop-in source for convert_dataset.py)
----------------------------------------------
    data.attrs["env_args"]                  env_args with control_freq updated
    data.attrs["total"]                     total samples written
    data/demo_i/states                      (T*k, 34)
    data/demo_i/actions                     (T*k, 7)   commands actually issued
    data/demo_i/rewards, dones              (T*k,)
    data/demo_i/obs/robot0_gripper_qpos     (T*k, 2)   read by convert_dataset
    data/demo_i/obs/robot0_eef_pos          (T*k, 3)   for validation / action recompute
    data/demo_i/obs/robot0_eef_quat         (T*k, 4)
    data/demo_i.attrs["model_file"], ["num_samples"], ["resim_success"]

Demos whose task success does not survive re-simulation are reported and, by
default, dropped (--keep_failures to retain them).

EXAMPLE
-------
    python resimulate_at_base_rate.py \
        --input  /data/theya/data/uncertainity_subgoal/D1/env_hdf5/core/hammer_cleanup_d1.hdf5 \
        --output /data/theya/data/rate_ablation/base_500hz/hammer_cleanup_d1.hdf5 \
        --control_freq 500 --n 10 --pool_size 16
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np

# Make `mimicgen` importable without requiring PYTHONPATH from the caller. This
# must also hold in spawned workers, so it runs at module import time.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "external", "mimicgen")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

SIMULATION_TIMESTEP = 0.002                       # robosuite macros.SIMULATION_TIMESTEP
VALID_CONTROL_FREQS = (1, 2, 4, 5, 10, 20, 25, 50, 100, 125, 250, 500)

# Populated once per worker process by _init_worker.
_ENV = None
_MAX_DPOS = None
_MAX_DROT = None
_K = None
_RESYNC = True


# --------------------------------------------------------------------------- #
# Rate validation
# --------------------------------------------------------------------------- #
def validate_rate(source_freq, target_freq):
    """Fail loudly rather than silently producing an off-by-truncation rate."""
    if target_freq not in VALID_CONTROL_FREQS:
        substeps = max(int((1.0 / target_freq) / SIMULATION_TIMESTEP), 1)
        actual = 1.0 / (substeps * SIMULATION_TIMESTEP)
        raise SystemExit(
            "[ERROR] control_freq={} does not divide 500 Hz evenly. robosuite "
            "truncates sub-steps with int(), so you would silently simulate at "
            "{:.2f} Hz.\n        Valid rates: {}".format(
                target_freq, actual, VALID_CONTROL_FREQS))
    if target_freq % source_freq != 0:
        raise SystemExit(
            "[ERROR] target ({} Hz) must be an integer multiple of the source "
            "rate ({} Hz) so sub-steps align with source actions.".format(
                target_freq, source_freq))
    substeps = int((1.0 / target_freq) / SIMULATION_TIMESTEP)
    if substeps == 1:
        print("[warn] control_freq={} gives exactly 1 physics sub-step per tick: "
              "the controller goal is re-set every integration step with no "
              "settling. This is expected for a 500 Hz base -- verify task "
              "success survives before scaling up.".format(target_freq))
    return target_freq // source_freq


# --------------------------------------------------------------------------- #
# Worker setup
# --------------------------------------------------------------------------- #
def _build_env(env_meta, control_freq):
    """Raw robosuite env at the target rate.

    Deliberately NOT robomimic's EnvRobosuite: this repo's patched wrapper fuses
    a four-camera Open3D point cloud on every get_observation(), which is very
    expensive and unnecessary here -- we only need sim states, eef pose and
    gripper qpos. Rendering happens later, in convert_dataset.py.

    With all renderers off no GL context is created at all, so this runs without
    EGL and parallelises cleanly across CPU cores.
    """
    import robosuite
    # Registers the MimicGen task classes (HammerCleanup_D1, CoffeePreparation_D1,
    # Kitchen_D1, ...) with robosuite's env registry. Without this robosuite.make
    # only knows the ~19 stock robosuite tasks and raises "Environment not found".
    import mimicgen  # noqa: F401  (side-effect import only)

    kwargs = dict(env_meta["env_kwargs"])
    kwargs.pop("env_name", None)
    kwargs.update(
        control_freq=int(control_freq),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        reward_shaping=False,
    )
    for key in ("camera_names", "camera_heights", "camera_widths",
                "camera_depths", "camera_segmentations", "render_gpu_device_id"):
        kwargs.pop(key, None)

    return robosuite.make(env_meta["env_name"], **kwargs)


def _init_worker(env_meta, control_freq, k, resync):
    global _ENV, _MAX_DPOS, _MAX_DROT, _K, _RESYNC
    _ENV = _build_env(env_meta, control_freq)
    controller = _ENV.robots[0].controller
    _MAX_DPOS = float(controller.output_max[0])
    _MAX_DROT = float(controller.output_max[3])
    _K = int(k)
    _RESYNC = bool(resync)


# --------------------------------------------------------------------------- #
# Core: replay one demo at the higher rate
# --------------------------------------------------------------------------- #
def _reset_to(env, model_xml, state_vector):
    """Mirror robomimic EnvRobosuite.reset_to for a raw robosuite env (v1.4)."""
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(np.asarray(state_vector))
    env.sim.forward()


def _eef_pose(env):
    """(pos, rotmat, obs) of the end effector, straight from the sim."""
    from robosuite.utils import transform_utils as T
    obs = env._get_observations(force_update=True)
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    rot = T.quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    return pos, rot, obs


def _resimulate(env, model_xml, init_state, actions, k, max_dpos, max_drot,
                resync_states=None):
    """Waypoint-follow one demo at k ticks per source action.

    Targets are reconstructed from the stored action deltas. Tracking the poses
    the original demo actually *reached* instead sounds more faithful but was
    measured at 0% task success on Hammer -- driving exactly onto the achieved
    pose removes the controller lag the original contact dynamics depended on.
    Do not "fix" this by switching to achieved poses.
    """
    from robosuite.utils import transform_utils as T

    _reset_to(env, model_xml, init_state)

    states, cmds, grip_qpos, eef_pos_log, eef_quat_log, rewards = [], [], [], [], [], []
    success = False

    # env.step() already returns the post-step observations (base.py:406), so we
    # never call _get_observations(force_update=True) inside the loop -- doing so
    # recomputes every observable a second time and roughly doubles the runtime.
    # The pose returned by step() is also what the next tick's residual is
    # computed from, so it is reused rather than re-queried.
    cur_pos, cur_rot, obs0 = _eef_pose(env)

    # Row 0 is the INITIAL state, before any tick. The source format has
    # states[0] = initial state (convert_dataset teleports to it), and striding
    # by k must land on 0, k, 2k, ... so every arm keeps that convention.
    states.append(np.asarray(env.sim.get_state().flatten()))
    cmds.append(np.zeros(7, dtype=np.float32))
    grip_qpos.append(np.asarray(obs0["robot0_gripper_qpos"], dtype=np.float32))
    eef_pos_log.append(cur_pos.astype(np.float32))
    eef_quat_log.append(np.asarray(obs0["robot0_eef_quat"], dtype=np.float32))
    rewards.append(0.0)

    for t, a in enumerate(np.asarray(actions, dtype=np.float64)):
        # RESYNC: snap back onto the original demo before each source interval.
        # Open-loop action replay diverges (measured: only 70% of Hammer demos
        # still succeed even at k=1, i.e. replaying at the ORIGINAL rate), so
        # without this the re-simulated dataset silently loses ~30% of demos and
        # keeps a success-biased subset. Re-syncing bounds divergence to a single
        # 1/20 s interval: every 20 Hz sample is then exactly the original state,
        # and only the k-1 intermediate states are newly simulated.
        if resync_states is not None:
            env.sim.set_state_from_flattened(np.asarray(resync_states[t]))
            env.sim.forward()
            cur_pos, cur_rot, _ = _eef_pose(env)

        # The absolute waypoint this source action was steering toward.
        target_pos = cur_pos + a[:3] * max_dpos
        delta_rot = T.quat2mat(T.axisangle2quat(a[3:6] * max_drot))
        target_rot = delta_rot @ cur_rot           # convert_dataset.py:183 convention

        for _ in range(k):
            cmd_pos = np.clip((target_pos - cur_pos) / max_dpos, -1.0, 1.0)
            # R_cmd such that R_cmd @ cur_rot == target_rot
            residual_rot = target_rot @ cur_rot.T
            cmd_rot = np.clip(
                T.quat2axisangle(T.mat2quat(residual_rot)) / max_drot, -1.0, 1.0)
            cmd = np.concatenate([cmd_pos, cmd_rot, [a[-1]]])

            post, reward, _, _ = env.step(cmd)

            cur_pos = np.asarray(post["robot0_eef_pos"], dtype=np.float64)
            cur_quat = np.asarray(post["robot0_eef_quat"], dtype=np.float64)
            cur_rot = T.quat2mat(cur_quat)

            states.append(np.asarray(env.sim.get_state().flatten()))
            cmds.append(cmd.astype(np.float32))
            grip_qpos.append(np.asarray(post["robot0_gripper_qpos"], dtype=np.float32))
            eef_pos_log.append(cur_pos.astype(np.float32))
            eef_quat_log.append(cur_quat.astype(np.float32))
            rewards.append(float(reward))

        # Success is sticky and only needs to be observed once; checking every
        # tick means k times more contact queries for no extra information.
        if not success and env._check_success():
            success = True

    if not states:
        return None

    n = len(states)
    dones = np.zeros(n, dtype=np.int64)
    dones[-1] = 1
    return dict(
        states=np.asarray(states, dtype=np.float64),
        actions=np.asarray(cmds, dtype=np.float32),
        robot0_gripper_qpos=np.asarray(grip_qpos, dtype=np.float32),
        robot0_eef_pos=np.asarray(eef_pos_log, dtype=np.float32),
        robot0_eef_quat=np.asarray(eef_quat_log, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=dones,
        success=bool(success),
    )


def _process_demo(task):
    """Pool worker entry point. Returns (ep, traj_or_None, error_or_None)."""
    ep, model_xml, states, actions = task
    try:
        resync = states if _RESYNC else None
        traj = _resimulate(_ENV, model_xml, states[0], actions,
                           _K, _MAX_DPOS, _MAX_DROT, resync)
        return ep, traj, None
    except Exception as exc:                       # keep one bad demo from killing the pool
        return ep, None, "{}: {}".format(type(exc).__name__, exc)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Re-simulate a MimicGen core dataset at a higher base control frequency")
    p.add_argument("--input", required=True, help="source 20 Hz core hdf5")
    p.add_argument("--output", required=True, help="destination hdf5 (states only, no images)")
    p.add_argument("--control_freq", type=int, default=500,
                   help="base control frequency in Hz; must divide 500 (default 500)")
    p.add_argument("--n", type=int, default=None, help="only process the first N demos")
    p.add_argument("--pool_size", type=int, default=16, help="worker processes (default 16)")
    p.add_argument("--no_resync", action="store_true",
                   help="disable per-source-step re-sync to the original states. "
                        "Re-sync is ON by default: open-loop action replay diverges "
                        "(only ~70%% of Hammer demos survive even at k=1), so without "
                        "it you lose ~30%% of demos to a success-biased filter.")
    p.add_argument("--keep_failures", action="store_true",
                   help="keep demos whose task success does not survive re-simulation")
    args = p.parse_args()

    with h5py.File(args.input, "r") as fin:
        env_meta = json.loads(fin["data"].attrs["env_args"])
        source_freq = int(env_meta["env_kwargs"]["control_freq"])
        k = validate_rate(source_freq, args.control_freq)

        demos = sorted(fin["data"].keys(), key=lambda s: int(s[5:]))
        if args.n is not None:
            demos = demos[: args.n]

        print("[rate] {} Hz -> {} Hz   (k={} ticks per source action)".format(
            source_freq, args.control_freq, k))
        print("[env ] {}   demos: {}   workers: {}   resync: {}".format(
            env_meta["env_name"], len(demos), args.pool_size, not args.no_resync))

        # states[0] + actions only; full states are not needed and pickle poorly.
        tasks = [(ep,
                  fin["data/{}".format(ep)].attrs["model_file"],
                  fin["data/{}/states".format(ep)][()],
                  fin["data/{}/actions".format(ep)][()])
                 for ep in demos]

    model_xml_by_ep = {t[0]: t[1] for t in tasks}

    out_meta = json.loads(json.dumps(env_meta))
    out_meta["env_kwargs"]["control_freq"] = int(args.control_freq)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    tmp_path = args.output + ".partial"

    kept, dropped, errored, total = 0, [], [], 0
    t0 = time.time()

    ctx = mp.get_context("spawn")                  # MuJoCo does not survive fork
    with h5py.File(tmp_path, "w") as fout:
        gout = fout.create_group("data")
        gout.attrs["env_args"] = json.dumps(out_meta)

        with ctx.Pool(args.pool_size, initializer=_init_worker,
                      initargs=(env_meta, args.control_freq, k, not args.no_resync)) as pool:
            for i, (ep, traj, err) in enumerate(
                    pool.imap_unordered(_process_demo, tasks), start=1):
                if err is not None:
                    errored.append((ep, err))
                elif traj is None:
                    errored.append((ep, "empty trajectory"))
                else:
                    if not traj["success"] and not args.keep_failures:
                        dropped.append(ep)
                    else:
                        if not traj["success"]:
                            dropped.append(ep)     # recorded but retained
                        g = gout.create_group(ep)
                        for key in ("states", "actions", "rewards", "dones"):
                            g.create_dataset(key, data=traj[key], compression="gzip")
                        og = g.create_group("obs")
                        for key in ("robot0_gripper_qpos", "robot0_eef_pos",
                                    "robot0_eef_quat"):
                            og.create_dataset(key, data=traj[key], compression="gzip")
                        g.attrs["model_file"] = model_xml_by_ep[ep]
                        g.attrs["num_samples"] = int(traj["states"].shape[0])
                        g.attrs["resim_success"] = bool(traj["success"])
                        kept += 1
                        total += int(traj["states"].shape[0])

                elapsed = time.time() - t0
                print("  [{}/{}] {}  kept={} dropped={} errors={}  {:.2f} demo/s"
                      .format(i, len(tasks), ep, kept, len(dropped), len(errored),
                              i / max(elapsed, 1e-9)), flush=True)

        gout.attrs["total"] = total

    os.replace(tmp_path, args.output)

    n_ok = len(tasks) - len(dropped) - len(errored)
    print("\n[done] {}".format(args.output))
    print("  demos written : {}/{}".format(kept, len(tasks)))
    print("  total samples : {:,}".format(total))
    print("  success rate  : {}/{} = {:.0%}".format(n_ok, len(tasks),
                                                    n_ok / max(len(tasks), 1)))
    print("  wall clock    : {:.1f} s".format(time.time() - t0))
    if dropped:
        print("  LOST SUCCESS  : {}{}".format(dropped[:10],
                                              " ..." if len(dropped) > 10 else ""))
    if errored:
        print("  ERRORED       : {}".format(errored[:5]))
    if dropped or errored:
        print("\n  Re-simulation is not bit-identical to the original rollout, so a\n"
              "  few contact-sensitive demos dropping out is expected. A large\n"
              "  fraction failing means waypoint tracking is not keeping up at this\n"
              "  rate -- try a lower base (e.g. 100 Hz) before scaling up.")


if __name__ == "__main__":
    sys.exit(main())
