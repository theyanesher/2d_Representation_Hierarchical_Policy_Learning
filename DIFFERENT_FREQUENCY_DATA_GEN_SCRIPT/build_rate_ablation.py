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

So each arm's actions are RE-DERIVED from that arm's own strided eef poses:

    delta_pos   = eef_pos[j+1] - eef_pos[j]                  (world frame)
    delta_local = R_j^T @ R_{j+1}                            (gripper frame)

which is exact for any k, needs no small-angle assumption, and uses ACHIEVED
poses rather than commanded ones (OSC never fully reaches its setpoint, so the
sum of commands drifts from the real displacement).

They are written back normalised to the [-1, 1] OSC convention, because
convert_dataset.py re-expands them by max_dpos / max_drot (lines 178-189). That
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
    """Exact per-arm actions from that arm's achieved poses.

    Returns (N, 7) normalised to [-1, 1], matching what convert_dataset.py
    expects to re-expand by max_dpos / max_drot.
    """
    from robosuite.utils import transform_utils as T

    n = len(eef_pos)
    out = np.zeros((n, 7), dtype=np.float32)
    rots = [T.quat2mat(np.asarray(q, dtype=np.float64)) for q in eef_quat]
    for j in range(n - 1):
        dpos = np.asarray(eef_pos[j + 1], dtype=np.float64) - np.asarray(
            eef_pos[j], dtype=np.float64)
        # World-frame rotation taking pose j to pose j+1, as convert_dataset
        # reconstructs it: waypoint_rot = delta_rot @ cur_rot.
        delta_world = rots[j + 1] @ rots[j].T
        daa = T.quat2axisangle(T.mat2quat(delta_world))
        out[j, :3] = np.clip(dpos / max_dpos, -1.0, 1.0)
        out[j, 3:6] = np.clip(daa / max_drot, -1.0, 1.0)
        out[j, 6] = gripper_cmd[j]
    # Last row has no successor; hold the final gripper command, zero motion.
    if n:
        out[-1, 6] = gripper_cmd[-1]
    return out


def stage_stride(base_hdf5, out_dir, base_freq, rates, task, force=False):
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

            tmp = arm_path + ".partial"
            total = 0
            with h5py.File(tmp, "w") as fout:
                g = fout.create_group("data")
                g.attrs["env_args"] = json.dumps(arm_meta)
                for ep in demos:
                    src = fin["data/{}".format(ep)]
                    idx = np.arange(0, src["states"].shape[0], k)
                    pos = src["obs/robot0_eef_pos"][()][idx]
                    quat = src["obs/robot0_eef_quat"][()][idx]
                    grip_cmd = src["actions"][()][idx][:, -1]

                    d = g.create_group(ep)
                    d.create_dataset("states", data=src["states"][()][idx],
                                     compression="gzip")
                    d.create_dataset("actions",
                                     data=_actions_from_poses(pos, quat, grip_cmd,
                                                              max_dpos, max_drot),
                                     compression="gzip")
                    d.create_dataset("rewards", data=src["rewards"][()][idx],
                                     compression="gzip")
                    dones = np.zeros(len(idx), dtype=np.int64)
                    dones[-1] = 1
                    d.create_dataset("dones", data=dones, compression="gzip")
                    og = d.create_group("obs")
                    og.create_dataset("robot0_gripper_qpos",
                                      data=src["obs/robot0_gripper_qpos"][()][idx],
                                      compression="gzip")
                    og.create_dataset("robot0_eef_pos", data=pos, compression="gzip")
                    og.create_dataset("robot0_eef_quat", data=quat, compression="gzip")
                    d.attrs["model_file"] = src.attrs["model_file"]
                    d.attrs["num_samples"] = int(len(idx))
                    total += len(idx)
                g.attrs["total"] = total
            os.replace(tmp, arm_path)
            print("[2/3]   {:>4} Hz (k={:<3}) {} demos, {:>9,} samples -> {}".format(
                rate, k, len(demos), total, os.path.basename(arm_path)))
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
                            args.task, force=args.force)
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
