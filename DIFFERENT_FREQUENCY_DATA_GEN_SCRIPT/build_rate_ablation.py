#!/usr/bin/env python3
"""
Sampling-rate ablation pipeline: one source dataset -> every rate arm, rendered.

    stage 1  resimulate   20 Hz core hdf5 ---------> 500 Hz states (no images)
    stage 2  stride       500 Hz states  ---------> per-arm states + actions
    stage 3  render       per-arm states ---------> per-arm npz trees (images)

Run it once per task; every stage is resumable and can be run in isolation with
--stages, so a crash during rendering never costs you the re-simulation.

WHY THE RATE LADDER IS WHAT IT IS
---------------------------------
robosuite derives physics sub-steps as int(control_timestep / model_timestep)
with model_timestep = 0.002 s, so a control rate MUST divide 500 exactly or it
silently becomes something else (40 -> 41.67, 60 -> 62.5, 200 -> 250). And since
500 = 2^2 * 5^3, any rate needing 2^3 or a factor of 3 is unreachable -- which
is why 2x/3x/10x of 20 Hz do not exist.

The base must additionally be an integer multiple of every arm. For the default
ladder {5, 10, 20, 50, 100, 250} the LCM is 500, so base = 500 Hz. A 250 Hz base
could NOT produce the 20 Hz arm (250/20 = 12.5).

STRIDING AND THE ACTION PROBLEM
-------------------------------
States stride trivially: arm sample j is base index j*k, k = base/rate. Row 0 is
the initial state in both, so the convention carries over.

Actions do NOT stride. They are OSC_POSE deltas meaning "move d from where you
are now", so actions[::k] is one k-th of the motion that actually occurred over
the interval -- train on that and the arm under-moves by a factor of k, with no
error anywhere to warn you. Nor can they simply be summed: position deltas add
only approximately, and rotations compose multiplicatively (R_{k-1}...R_1 R_0),
so summing axis-angles is wrong whenever the wrist changes axis mid-interval.

Nor may they be replaced by the ACHIEVED delta eef_pos[j+1] - eef_pos[j]
(action_mode=achieved, the original implementation). OSC_POSE here is a
critically-damped tracker (kp=150, no interpolator): a goal held for one
control period T is only reached to 1-(1+wT)e^{-wT}, w=sqrt(kp) -- ~13% at
20 Hz, ~1% at 100 Hz -- and the policy in the source data commands
accordingly (measured achieved/commanded = 0.23 at 20 Hz). Labelling the
achieved delta as the command under-commands by ~4x at 20 Hz and ~15x at
100 Hz; policies trained that way crawl and score ~0 (5 Hz survives at
~0.2 because it is ~1.3x). Same rate, same recipe, native data -> 0.59.

So each arm's actions are obtained by INVERTING THE CONTROLLER (action_mode=
invert, the default): reset the sim to the arm's true state_j, and find the
OSC command at that arm's control_freq which, held for one period, lands on
the arm's true pose_{j+1}. Two probes fit the local affine response (which
includes velocity carry-over), one Newton step refines it, one more step
verifies it; the residual is stored per demo. Exact per step, no calibration
constants, valid at any rate, and for the 20 Hz arm it recovers the source
commands (validate_arm_actions.py checks this and replays every arm).

Actions are written normalised to the [-1, 1] OSC convention, because
convert_dataset.py re-expands them by max_dpos / max_drot (lines 198-217). That
means stage 3 needs no modification at all -- it just reads each arm's hdf5.

KEYPOINTS
---------
Deliberately skipped here (--no_subgoal on convert_dataset). Keypoint detection
is run afterwards, per arm, so that BOCPD/RDP/etc. see each rate's own
trajectory -- which is the thing the ablation is measuring. Note the else-branch
in convert_dataset.py computes a curvature heuristic when --use_bayesian_decomp
is absent, so simply dropping that flag does NOT skip keypoints; --no_subgoal
must be applied to convert_dataset.py first.

EXAMPLE
-------
    python build_rate_ablation.py --task hammer_cleanup_d1 --n 140 --gpu 0
    python build_rate_ablation.py --task coffee_preparation_d1 --stages render --gpu 1
"""

import argparse
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "external", "mimicgen")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_SRC_DIR = "/data/theya/data/uncertainity_subgoal/D1/env_hdf5/core"
DEFAULT_OUT_ROOT = "/data/theya/data/rate_ablation"
DEFAULT_RATES = (5, 10, 20, 50, 100, 250)
DEFAULT_BASE_FREQ = 500
CONVERT_SCRIPT = "external/mimicgen/mimicgen/scripts/convert_dataset.py"
# third_party.robogen.robogen_utils does `from manipulation.utils import ...`,
# which lives in the low-level repo. Every eval script in shell_scripts/ adds
# this to PYTHONPATH for the same reason.
DEFAULT_LL_REPO = ("/home/theyanesh/Pratik_Low_Level/"
                   "2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference")


# --------------------------------------------------------------------------- #
# Stage 1 -- re-simulate at the base rate
# --------------------------------------------------------------------------- #
def stage_resimulate(src, base_hdf5, base_freq, n_demos, pool_size, python_bin):
    if os.path.exists(base_hdf5):
        print("[1/3] base exists, skipping: {}".format(base_hdf5))
        return
    print("[1/3] re-simulating {} -> {} Hz".format(os.path.basename(src), base_freq))
    cmd = [python_bin, os.path.join(_HERE, "resimulate_at_base_rate.py"),
           "--input", src, "--output", base_hdf5,
           "--control_freq", str(base_freq), "--no_resync",
           "--pool_size", str(pool_size)]
    if n_demos is not None:
        cmd += ["--n", str(n_demos)]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


# --------------------------------------------------------------------------- #
# Stage 2 -- stride to each arm, re-deriving actions
# --------------------------------------------------------------------------- #
def _actions_from_poses(eef_pos, eef_quat, gripper_cmd, max_dpos, max_drot):
    """LEGACY (action_mode=achieved): achieved per-step deltas labelled as commands.

    Kept only for ablation / comparison. Under-commands by the controller's
    tracking ratio (see module docstring) -- do not train on this.
    """
    from robosuite.utils import transform_utils as T

    n = len(eef_pos)
    out = np.zeros((n, 7), dtype=np.float32)
    rots = [T.quat2mat(np.asarray(q, dtype=np.float64)) for q in eef_quat]
    for j in range(n - 1):
        dpos = np.asarray(eef_pos[j + 1], dtype=np.float64) - np.asarray(
            eef_pos[j], dtype=np.float64)
        delta_world = rots[j + 1] @ rots[j].T
        daa = T.quat2axisangle(T.mat2quat(delta_world))
        out[j, :3] = np.clip(dpos / max_dpos, -1.0, 1.0)
        out[j, 3:6] = np.clip(daa / max_drot, -1.0, 1.0)
        out[j, 6] = gripper_cmd[j]
    if n:
        out[-1, 6] = gripper_cmd[-1]
    return out


# ---- controller inversion (action_mode=invert) ------------------------------
PROBE_EPS = 0.2     # probe command in normalised units; well inside [-1, 1]
ROT_TINY = 1e-6     # OSC_POSE.set_goal keeps a STALE goal_ori on an exactly-zero
                    # rotation delta (math.isclose check), so never send exact 0.
GAIN_FLOOR = 1e-3   # below this the axis is not responding (limit/singularity)

_STRIDE_ENV = None
_STRIDE_LIMITS = None


def _init_stride_worker(env_meta, rate):
    global _STRIDE_ENV, _STRIDE_LIMITS
    from resimulate_at_base_rate import _build_env
    _STRIDE_ENV = _build_env(env_meta, rate)
    ctrl = _STRIDE_ENV.robots[0].controller
    _STRIDE_LIMITS = (float(ctrl.output_max[0]), float(ctrl.output_max[3]))


def _world_aa(R_to, R_from):
    """Axis-angle of the world-frame rotation taking R_from onto R_to."""
    from robosuite.utils import transform_utils as T
    q = T.mat2quat(R_to @ R_from.T)
    if q[3] < 0:                      # canonical hemisphere, else angle -> 2pi - theta
        q = -q
    return T.quat2axisangle(q)


def _nonzero_rot(c):
    c = np.array(c, dtype=np.float64)
    c[c == 0.0] = ROT_TINY
    return c


def _probe(env, state, cmd6, grip, R_ref):
    """Reset to `state`, hold `cmd6` (normalised pos+rot) for one control period.

    Returns the achieved pose as a 6-vector: world-frame position (m) and the
    world-frame axis-angle (rad) taking R_ref onto the achieved orientation.
    """
    from robosuite.utils import transform_utils as T
    env.sim.set_state_from_flattened(np.asarray(state))
    env.sim.forward()
    cmd6 = np.asarray(cmd6, dtype=np.float64)
    post, _, _, _ = env.step(np.concatenate([cmd6[:3], _nonzero_rot(cmd6[3:6]), [grip]]))
    p = np.asarray(post["robot0_eef_pos"], dtype=np.float64)
    R = T.quat2mat(np.asarray(post["robot0_eef_quat"], dtype=np.float64))
    return np.concatenate([p, _world_aa(R, R_ref)])


def _solve(J, r, lo, hi):
    """Box-constrained least squares for J dc = r with lo <= dc <= hi.

    The controller clips commands to [-1, 1] itself, so the physical response
    is affine only inside the box; an unconstrained solve that is then clipped
    lands somewhere else. The source data proves a feasible in-box command
    exists for every step (it produced the pose), so the bounded solve finds
    it. Robust to a dead axis (joint limit / singularity) via lsq_linear.
    """
    from scipy.optimize import lsq_linear
    res = lsq_linear(J, r, bounds=(lo, hi), lsmr_tol="auto", max_iter=200)
    return res.x


MAX_ITERS = 6       # 20/100 Hz converge on the prior; 5 Hz (4 waypoints/step) needs a few
TOL_POS = 1e-4      # m   -- converged when the verified pose is within 0.1 mm ...
TOL_ROT = 1e-3      # rad -- ... and 0.06 deg of the arm's true next pose


def _pose_err(y_a, y_b, rot_weight=0.1):
    """Scalar pose error: metres + (rad * 0.1 m/rad lever) so rot counts comparably."""
    return np.linalg.norm(y_a[:3] - y_b[:3]) + rot_weight * np.linalg.norm(y_a[3:] - y_b[3:])


def _jacobian(env, s, c, g, R_ref, y_c):
    """Finite-difference response Jacobian around command c, one probe per DOF.

    Probing all DOFs at once lets the rotation transient swamp the ~1 mm
    position response at 20 Hz (it even flips the fitted sign), hence per-DOF.
    Probes step away from the box edge so they stay inside [-1, 1].
    """
    J = np.zeros((6, 6))
    for i in range(6):
        h = PROBE_EPS if c[i] + PROBE_EPS <= 1.0 else -PROBE_EPS
        e = np.zeros(6); e[i] = h
        J[:, i] = (_probe(env, s, c + e, g, R_ref) - y_c) / h
    return J


def _invert_demo(task):
    """Pool worker: for every strided step find the command that reaches the next pose.

    y(c) = pose reached by holding the 6-D normalised command c for one control
    period from the arm's true state_j; the target is the arm's true pose_{j+1}.
    Gauss-Newton with a finite-difference Jacobian, box-constrained to [-1, 1]:

      * start from the command the 500 Hz base actually issued at that tick
        (`prior`). For the 20 Hz arm that IS the source command, so almost every
        step converges on the first probe; for other rates it is the residual
        toward the demonstrator's waypoint -- the right intent, wrong magnitude.
        Starting from the demonstrator's command also resolves the non-uniqueness
        of contact-phase steps (pressing harder into a surface does not change the
        pose) toward what the demonstrator did, instead of a minimum-norm push.
      * re-estimate J at the current iterate (a linearisation around c = 0 is
        useless from rest: joint stiction makes the small-probe response tiny,
        the gains come out far too small and the solve saturates) and accept
        only improving steps (half step once, then stop) so contact
        nonlinearities cannot make it diverge.
      * commands the controller physically cannot execute in one period remain
        saturated, exactly as in the source data.

    Returns (ep, actions (N,7) float32, diag dict, error_or_None).
    """
    ep, model_xml, states, eef_pos, eef_quat, grip_cmd, prior = task
    env = _STRIDE_ENV
    try:
        from resimulate_at_base_rate import _reset_to
        from robosuite.utils import transform_utils as T

        _reset_to(env, model_xml, states[0])
        n = len(states)
        out = np.zeros((n, 7), dtype=np.float32)
        m = max(n - 1, 0)
        err_pos = np.zeros(m); err_rot = np.zeros(m)
        sat = np.zeros(m, dtype=bool); iters = np.zeros(m, dtype=np.int8)
        rots = [T.quat2mat(np.asarray(q, dtype=np.float64)) for q in eef_quat]
        one = np.ones(6)

        for j in range(n - 1):
            Rj = rots[j]
            s = states[j]
            g = float(grip_cmd[j])
            y_t = np.concatenate([np.asarray(eef_pos[j + 1], dtype=np.float64),
                                  _world_aa(rots[j + 1], Rj)])

            c = np.clip(np.asarray(prior[j], dtype=np.float64), -1.0, 1.0)
            y_c = _probe(env, s, c, g, Rj)
            for it in range(MAX_ITERS):
                r = y_t - y_c
                if np.linalg.norm(r[:3]) < TOL_POS and np.linalg.norm(r[3:]) < TOL_ROT:
                    break
                J = _jacobian(env, s, c, g, Rj, y_c)
                dc = _solve(J, r, -one - c, one - c)
                improved = False
                for scale in (1.0, 0.5):
                    c_new = np.clip(c + scale * dc, -1.0, 1.0)
                    y_new = _probe(env, s, c_new, g, Rj)
                    if _pose_err(y_new, y_t) < _pose_err(y_c, y_t):
                        c, y_c, improved = c_new, y_new, True
                        break
                iters[j] = it + 1
                if not improved:
                    break

            err_pos[j] = np.linalg.norm(y_c[:3] - y_t[:3])
            err_rot[j] = np.linalg.norm(_world_aa(
                T.quat2mat(T.axisangle2quat(y_c[3:])) @ Rj, rots[j + 1]))
            sat[j] = bool(np.any(np.abs(c) >= 1.0 - 1e-6))
            out[j, :6] = c
            out[j, 6] = g
        if n:
            out[-1, 6] = grip_cmd[-1]
        return ep, out, dict(err_pos=err_pos, err_rot=err_rot, sat=sat, iters=iters), None
    except Exception as exc:                       # keep one bad demo from killing the pool
        return ep, None, None, "{}: {}".format(type(exc).__name__, exc)


def _invert_arm(env_meta, rate, payload, pool_size):
    import multiprocessing as mp
    tasks = [(ep, xml, st, pos, quat, grip, prior)
             for (ep, xml, st, pos, quat, grip, prior, _, _) in payload]
    results, errors = {}, []
    t0 = time.time()
    ctx = mp.get_context("spawn")                  # MuJoCo does not survive fork
    with ctx.Pool(pool_size, initializer=_init_stride_worker,
                  initargs=(env_meta, rate)) as pool:
        for i, (ep, acts, diag, err) in enumerate(pool.imap_unordered(_invert_demo, tasks), 1):
            if err is not None:
                errors.append((ep, err))
            else:
                results[ep] = (acts, diag)
            if i % 10 == 0 or i == len(tasks):
                print("[2/3]     {:>4} Hz invert {}/{}  {:.1f} demo/s".format(
                    rate, i, len(tasks), i / max(time.time() - t0, 1e-9)), flush=True)
    if errors:
        raise SystemExit("[ERROR] inversion failed for {} demo(s) at {} Hz: {}".format(
            len(errors), rate, errors[:3]))
    return results


def stage_stride(base_hdf5, out_dir, base_freq, rates, task, pool_size=16,
                 action_mode="invert", force=False):
    if action_mode not in ("invert", "achieved"):
        raise SystemExit("[ERROR] unknown action_mode {!r}".format(action_mode))
    with h5py.File(base_hdf5, "r") as fin:
        base_meta = json.loads(fin["data"].attrs["env_args"])
        demos = sorted(fin["data"].keys(), key=lambda s: int(s[5:]))
        if not demos:
            raise SystemExit("[ERROR] {} contains no demos".format(base_hdf5))

        # max_dpos / max_drot are controller properties, identical across rates.
        from resimulate_at_base_rate import _build_env
        probe = _build_env(base_meta, base_freq)
        ctrl = probe.robots[0].controller
        max_dpos, max_drot = float(ctrl.output_max[0]), float(ctrl.output_max[3])
        probe.close()

        made = []
        for rate in rates:
            if base_freq % rate:
                raise SystemExit(
                    "[ERROR] base {} Hz is not an integer multiple of arm {} Hz; "
                    "that arm cannot be produced by striding.".format(base_freq, rate))
            k = base_freq // rate
            arm_path = os.path.join(out_dir, "{}_{}hz.hdf5".format(task, rate))
            if os.path.exists(arm_path) and not force:
                print("[2/3]   {:>4} Hz (k={:<3}) exists, skipping".format(rate, k))
                made.append(arm_path)
                continue

            arm_meta = json.loads(json.dumps(base_meta))
            arm_meta["env_kwargs"]["control_freq"] = int(rate)

            # Strided payload per demo (states are small; images come later).
            payload = []
            for ep in demos:
                src = fin["data/{}".format(ep)]
                n_base = src["states"].shape[0]
                idx = np.arange(0, n_base, k)
                # Base row t holds the command that PRODUCED state t (row 0 is an
                # all-zero placeholder before any tick). The gripper command in
                # force during arm interval j is therefore row j*k + 1, not j*k --
                # the latter is the last tick of the previous interval, which
                # lags every gripper toggle by one step and is 0 at j = 0.
                cmd_rows = np.minimum(idx + 1, n_base - 1)
                base_cmds = src["actions"][()][cmd_rows]
                payload.append((ep, src.attrs["model_file"],
                                src["states"][()][idx],
                                src["obs/robot0_eef_pos"][()][idx],
                                src["obs/robot0_eef_quat"][()][idx],
                                base_cmds[:, -1],           # gripper command in force
                                base_cmds[:, :6],           # arm command issued at that tick (prior)
                                src["rewards"][()][idx],
                                src["obs/robot0_gripper_qpos"][()][idx]))

            if action_mode == "invert":
                print("[2/3]   {:>4} Hz (k={:<3}) inverting controller, {} demos, {} workers"
                      .format(rate, k, len(demos), pool_size), flush=True)
                results = _invert_arm(base_meta, rate, payload, pool_size)
            else:
                results = {ep: (_actions_from_poses(pos, quat, grip, max_dpos, max_drot), None)
                           for (ep, _, _, pos, quat, grip, _, _, _) in payload}

            tmp = arm_path + ".partial"
            total = 0
            all_err_pos, all_err_rot, all_sat = [], [], []
            with h5py.File(tmp, "w") as fout:
                g = fout.create_group("data")
                g.attrs["env_args"] = json.dumps(arm_meta)
                g.attrs["action_mode"] = action_mode
                for (ep, xml, st, pos, quat, grip, _prior, rew, gq) in payload:
                    acts, diag = results[ep]
                    d = g.create_group(ep)
                    d.create_dataset("states", data=st, compression="gzip")
                    d.create_dataset("actions", data=acts, compression="gzip")
                    d.create_dataset("rewards", data=rew, compression="gzip")
                    dones = np.zeros(len(st), dtype=np.int64)
                    dones[-1] = 1
                    d.create_dataset("dones", data=dones, compression="gzip")
                    og = d.create_group("obs")
                    og.create_dataset("robot0_gripper_qpos", data=gq, compression="gzip")
                    og.create_dataset("robot0_eef_pos", data=pos, compression="gzip")
                    og.create_dataset("robot0_eef_quat", data=quat, compression="gzip")
                    d.attrs["model_file"] = xml
                    d.attrs["num_samples"] = int(len(st))
                    if diag is not None:
                        d.create_dataset("invert_err_pos", data=diag["err_pos"].astype(np.float32))
                        d.create_dataset("invert_err_rot", data=diag["err_rot"].astype(np.float32))
                        d.attrs["invert_err_pos_p95_mm"] = float(np.percentile(diag["err_pos"], 95) * 1e3) if len(diag["err_pos"]) else 0.0
                        d.attrs["invert_saturated_frac"] = float(diag["sat"].mean()) if len(diag["sat"]) else 0.0
                        d.create_dataset("invert_iters", data=diag["iters"])
                        d.create_dataset("invert_sat", data=diag["sat"])
                        all_err_pos.append(diag["err_pos"]); all_err_rot.append(diag["err_rot"]); all_sat.append(diag["sat"])
                    total += len(st)
                g.attrs["total"] = total
                if all_err_pos:
                    ep_all = np.concatenate(all_err_pos); er_all = np.concatenate(all_err_rot); sat_all = np.concatenate(all_sat)
                    g.attrs["invert_err_pos_median_mm"] = float(np.median(ep_all) * 1e3)
                    g.attrs["invert_err_pos_p95_mm"] = float(np.percentile(ep_all, 95) * 1e3)
                    g.attrs["invert_err_rot_p95_deg"] = float(np.degrees(np.percentile(er_all, 95)))
                    g.attrs["invert_saturated_frac"] = float(sat_all.mean())
            os.replace(tmp, arm_path)
            msg = "[2/3]   {:>4} Hz (k={:<3}) {} demos, {:>9,} samples -> {}".format(
                rate, k, len(demos), total, os.path.basename(arm_path))
            if all_err_pos:
                uns = ~sat_all
                msg += ("\n[2/3]          inversion residual: pos median {:.3f} mm, p95 {:.3f} mm; rot p95 {:.3f} deg; "
                        "saturated {:.1%} of steps; unsaturated-only pos p95 {:.3f} mm").format(
                    np.median(ep_all) * 1e3, np.percentile(ep_all, 95) * 1e3,
                    np.degrees(np.percentile(er_all, 95)), sat_all.mean(),
                    np.percentile(ep_all[uns], 95) * 1e3 if uns.any() else float("nan"))
            print(msg, flush=True)
            made.append(arm_path)
    return made


# --------------------------------------------------------------------------- #
# Stage 3 -- render each arm
# --------------------------------------------------------------------------- #
def stage_render(arm_hdf5s, out_root, task, rates, gpu, pool_size,
                 camera_h, camera_w, python_bin, ll_repo, force=False):
    convert = os.path.join(_REPO_ROOT, CONVERT_SCRIPT)
    if not os.path.isfile(convert):
        raise SystemExit("[ERROR] missing {}".format(convert))

    # convert_dataset.py imports third_party.robogen (repo root), mimicgen
    # (external/mimicgen) and robomimic. external/robomimic MUST precede the
    # site-packages copy: it is the only one emitting the base-frame `state` and
    # the fused point cloud the rest of the pipeline expects.
    if not os.path.isdir(ll_repo):
        print("[3/3][WARN] --ll_repo not found: {}\n"
              "            third_party.robogen needs manipulation.utils from there; "
              "conversion will fail.".format(ll_repo))
    pypath = os.pathsep.join([
        _REPO_ROOT,
        os.path.join(_REPO_ROOT, "external", "mimicgen"),
        os.path.join(_REPO_ROOT, "external", "robomimic"),
        ll_repo,
        os.environ.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               PYTHONPATH=pypath, PYTHONNOUSERSITE="1")
    env.setdefault("DISPLAY", ":99")

    probe = subprocess.run([python_bin, convert, "--help"], cwd=_REPO_ROOT,
                           env=env, capture_output=True, text=True)
    supports_no_subgoal = "--no_subgoal" in probe.stdout
    if not supports_no_subgoal:
        print("[3/3][WARN] convert_dataset.py did not report a --no_subgoal flag, so it\n"
              "            will run its curvature-heuristic keypoints (the else-branch)\n"
              "            on every arm: wasted time, and the wrong keypoints.")
        if probe.returncode != 0:
            print("[3/3][WARN] (--help itself failed, so the probe is unreliable:\n"
                  "            {})".format(probe.stderr.strip().splitlines()[-1:]))

    for arm, rate in zip(arm_hdf5s, rates):
        outdir = os.path.join(out_root, "npz", "{}_{}hz".format(task, rate))
        done_marker = os.path.join(outdir, ".render_complete")
        if os.path.exists(done_marker) and not force:
            print("[3/3]   {:>4} Hz already rendered, skipping".format(rate))
            continue
        os.makedirs(outdir, exist_ok=True)
        print("[3/3]   rendering {:>4} Hz -> {}".format(rate, outdir), flush=True)
        cmd = [python_bin, convert, "--input", arm, "--output_dir", outdir,
               "--camera_height", str(camera_h), "--camera_width", str(camera_w),
               "--pool_size", str(pool_size)]
        if supports_no_subgoal:
            cmd.append("--no_subgoal")
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=_REPO_ROOT, env=env)
        open(done_marker, "w").write("ok\n")
        print("[3/3]   {:>4} Hz done in {:.1f} min".format(rate, (time.time() - t0) / 60))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Build every sampling-rate arm for one task, end to end")
    p.add_argument("--task", required=True,
                   help="e.g. hammer_cleanup_d1 / coffee_preparation_d1 / kitchen_d1")
    p.add_argument("--src_dir", default=DEFAULT_SRC_DIR)
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--base_freq", type=int, default=DEFAULT_BASE_FREQ)
    p.add_argument("--rates", type=int, nargs="+", default=list(DEFAULT_RATES))
    p.add_argument("--n", type=int, default=140,
                   help="source demos to re-simulate; ~77%% survive replay, so 140 "
                        "yields roughly 108 (default 140)")
    p.add_argument("--pool_size", type=int, default=16)
    p.add_argument("--render_pool_size", type=int, default=6,
                   help="workers for convert_dataset; VRAM-bound, ~6 per 24 GB GPU")
    p.add_argument("--camera_h", type=int, default=256)
    p.add_argument("--camera_w", type=int, default=256)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--stages", default="resim,stride,render",
                   help="comma-separated subset of resim,stride,render")
    p.add_argument("--action_mode", default="invert", choices=("invert", "achieved"),
                   help="stride stage: 'invert' (default) recovers the OSC command that "
                        "reaches each next pose by probing the simulator; 'achieved' is "
                        "the legacy achieved-delta labelling, kept only for comparison")
    p.add_argument("--force", action="store_true", help="redo stages even if outputs exist")
    p.add_argument("--ll_repo", default=DEFAULT_LL_REPO,
                   help="Low_Level_and_Inference checkout providing manipulation.utils")
    p.add_argument("--python_bin",
                   default=os.path.join(_REPO_ROOT, ".pixi/envs/eval/bin/python"))
    args = p.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    src = os.path.join(args.src_dir, args.task + ".hdf5")
    if not os.path.isfile(src):
        raise SystemExit("[ERROR] no source dataset: {}".format(src))

    base_dir = os.path.join(args.out_root, "base_{}hz".format(args.base_freq))
    arm_dir = os.path.join(args.out_root, "arms")
    for d in (base_dir, arm_dir):
        os.makedirs(d, exist_ok=True)
    base_hdf5 = os.path.join(base_dir, args.task + ".hdf5")

    for rate in args.rates:
        if args.base_freq % rate:
            raise SystemExit(
                "[ERROR] arm {} Hz is not an integer divisor of the {} Hz base."
                .format(rate, args.base_freq))

    print("=" * 72)
    print("task  : {}".format(args.task))
    print("base  : {} Hz     arms: {}".format(args.base_freq, args.rates))
    print("stages: {}        gpu: {}".format(sorted(stages), args.gpu))
    print("=" * 72)
    t0 = time.time()

    if "resim" in stages:
        stage_resimulate(src, base_hdf5, args.base_freq, args.n,
                         args.pool_size, args.python_bin)
    if not os.path.exists(base_hdf5):
        raise SystemExit("[ERROR] {} missing -- run the resim stage first".format(base_hdf5))

    arms = [os.path.join(arm_dir, "{}_{}hz.hdf5".format(args.task, r)) for r in args.rates]
    if "stride" in stages:
        arms = stage_stride(base_hdf5, arm_dir, args.base_freq, args.rates,
                            args.task, pool_size=args.pool_size,
                            action_mode=args.action_mode, force=args.force)
    if "render" in stages:
        missing = [a for a in arms if not os.path.exists(a)]
        if missing:
            raise SystemExit("[ERROR] arm files missing, run the stride stage: {}"
                             .format(missing))
        stage_render(arms, args.out_root, args.task, args.rates, args.gpu,
                     args.render_pool_size, args.camera_h, args.camera_w,
                     args.python_bin, args.ll_repo, force=args.force)

    print("\n[done] {} in {:.1f} min".format(args.task, (time.time() - t0) / 60))
    print("  base : {}".format(base_hdf5))
    print("  arms : {}".format(arm_dir))
    print("  npz  : {}".format(os.path.join(args.out_root, "npz")))


if __name__ == "__main__":
    sys.exit(main())
