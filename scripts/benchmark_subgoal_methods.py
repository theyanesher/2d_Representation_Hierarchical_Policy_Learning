"""Measure per-demo inference time for the subgoal methods in
external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py.

Times ONLY the keypoint computation. Loading the per-frame .npz and writing
the mirror tree are excluded -- on the rendered datasets those dominate wall
time by orders of magnitude and say nothing about a method's cost. Every demo
is loaded once, up front, then each method is run --warmup times untimed and
--repeats times timed; the reported number is the MEDIAN over repeats (the
mean is dominated by GC/scheduler outliers on a loaded box).

ONE METHOD PER PROCESS
----------------------
Run this with a single --methods value and let the caller loop, or use
--isolate to have the script re-exec itself once per method. This is not
tidiness: generate_extra_keypoints.py:96-105 documents that importing
awe_subgoal_decomp pulls robosuite -> numba-jitted transform utils, which sets
the CPU flush-denormals-to-zero flag process-wide, which perturbs
scipy.interpolate.splprep's convergence inside bspline_subgoal_decomp. Timing
awe and bspline in one process therefore changes bspline's knot count and its
runtime. --isolate is the default for that reason.

COST CLASSES -- do not put these in one column
----------------------------------------------
  in-process CPU : rdp, rdp_gripper, random, fixed_interval, bspline,
                   bspline_greville, awe, gripper_heuristic,
                   fixed_interval_const, orientation_heuristic
  out-of-process : uvd -- each call spawns the py3.9 `uvd` pixi env and loads a
                   vision model on the GPU. Reported as end-to-end wall time;
                   the fixed startup cost is reported separately as the
                   difference between a 1-repeat and an n-repeat run.
  network        : vlm -- Qwen round-trips. Latency-bound, and the frame count
                   sent is set by --vlm_sample_every_n_frames, not by T.

TRAJECTORY LENGTH
-----------------
Cost scales with T (awe --awe_solver dp is O(T^3)). Point --dataset_dir at the
rate-ablation arms to get a T-sweep over identical trajectories:

    /data/theya/data/rate_ablation/npz/hammer_cleanup_d1_{5,10,20,50,100}hz

Example -- the ten CPU methods on one arm, one process each:

    .pixi/envs/eval/bin/python scripts/benchmark_subgoal_methods.py \\
        --dataset_dir /data/theya/data/rate_ablation/npz/hammer_cleanup_d1_20hz \\
        --methods cpu --demos 10 --repeats 5 \\
        --out logs/subgoal_timing/hammer_20hz.json
"""
import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_GEN = os.path.join(_ROOT, "external", "mimicgen", "mimicgen", "scripts",
                    "generate_extra_keypoints.py")


def _load_generator():
    """Import generate_extra_keypoints.py by path and reuse ITS helpers, so the
    timed code is the same code the pipeline runs. Only the dispatch switch is
    mirrored below (_time_one_method); if that switch changes upstream, this
    file must follow."""
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, os.path.join(_ROOT, "third_party", "robogen"))
    spec = importlib.util.spec_from_file_location("_gek", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Methods that are pure in-process CPU -- the only ones directly comparable.
CPU_METHODS = (
    "rdp", "rdp_gripper", "random", "fixed_interval",
    "bspline", "bspline_greville", "awe",
    "gripper_heuristic", "fixed_interval_const", "orientation_heuristic",
)


def _build_opts(args):
    """A namespace matching what process_demo reads off its `opts`. Defaults are
    generate_extra_keypoints.py's own -- keep them in sync."""
    return argparse.Namespace(
        gripperless=args.gripperless,
        epsilon=args.epsilon,
        interval=None,
        const_interval=args.const_interval,
        n_random=args.n_random,
        seed=args.seed,
        snap_window=args.snap_window,
        max_error=args.max_error,
        degree=args.degree,
        influence_threshold=None,
        awe_err_threshold=args.awe_err_threshold,
        awe_solver=args.awe_solver,
        awe_use_gripper=args.awe_use_gripper,
        orientation_threshold=args.orientation_threshold,
        uvd_camera=args.uvd_camera,
        uvd_preprocessor=args.uvd_preprocessor,
        uvd_device=None,
        uvd_pixi_env="uvd",
        uvd_pixi_manifest=os.path.join(_ROOT, "pixi.toml"),
        vlm_camera=args.vlm_camera,
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        vlm_qwen_base_url=args.vlm_qwen_base_url,
        vlm_sample_every_n_frames=args.vlm_sample_every_n_frames,
        vlm_refine=True,
        vlm_stop_after_sparse_annotation=False,
        vlm_refinement_radius=15,
        vlm_refinement_stride=1,
        vlm_min_boundary_distance_frames=0,
        vlm_frame_width=224,
        vlm_frames_per_sheet=20,
        vlm_columns=5,
        vlm_sheet_overlap_frames=2,
        vlm_instruction=None,
        vlm_logs_dir=os.path.join(_ROOT, "logs", "subtask_boundaries_bench"),
        mix_groups=[],
    )


def _time_one_method(gek, method, demo, opts):
    """Run `method` once on a pre-loaded demo, return (seconds, n_keypoints).

    Mirrors the dispatch in generate_extra_keypoints.process_demo. The clock
    brackets ONLY the compute call."""
    T = demo["T"]
    t0 = time.perf_counter()

    if method in gek.VLM_METHODS:
        boundaries = gek._compute_vlm_boundaries(
            demo["rgb"][opts.vlm_camera], opts,
            os.path.join(opts.vlm_logs_dir, demo["name"]))
        switch_idxs = gek._finalize_indices(np.asarray(boundaries, dtype=int), T)
    elif method in gek.UVD_METHODS:
        _, switch_idxs = gek._compute_uvd_via_subprocess(
            demo["gripper_pcd"], demo["rgb"][opts.uvd_camera], opts)
    elif method in gek.BSPLINE_METHODS:
        _, switch_idxs = gek.compute_bspline_subgoal_gripper_pcd(
            gripper_pcd=demo["gripper_pcd"], eef_pos=demo["eef_pos"],
            method=method, max_error=opts.max_error, degree=opts.degree,
            influence_threshold=opts.influence_threshold,
            return_switch_idxs=True)
    elif method in gek.HEURISTIC_METHODS:
        if method == "gripper_heuristic":
            idxs = gek._gripper_heuristic_keypoints(demo["gripper_qpos"], demo["action"])
        elif method == "orientation_heuristic":
            idxs = gek._orientation_heuristic_keypoints(
                demo["eef_quat"], opts.orientation_threshold)
        else:
            idxs = gek._fixed_interval_const_keypoints(T, opts.const_interval)
        switch_idxs = gek._finalize_indices(idxs, T)
    elif method in gek.AWE_METHODS:
        _, switch_idxs = gek._get_awe_fn()(
            gripper_pcd=demo["gripper_pcd"], eef_pos=demo["eef_pos"],
            eef_quat=demo["eef_quat"], actions=demo["action"],
            err_threshold=opts.awe_err_threshold, method=opts.awe_solver,
            pos_only=False, use_gripper_seeding=opts.awe_use_gripper,
            return_switch_idxs=True)
    else:
        _, switch_idxs = gek.compute_rdp_subgoal_gripper_pcd(
            gripper_pcd=demo["gripper_pcd"], eef_pos=demo["eef_pos"],
            method=method, eef_qpos=demo["gripper_qpos"], actions=demo["action"],
            epsilon=opts.epsilon, interval=opts.interval, n_random=opts.n_random,
            seed=opts.seed, snap_window=opts.snap_window, return_switch_idxs=True)

    return time.perf_counter() - t0, len(switch_idxs)


def _load_demos(gek, dataset_dir, n_demos, methods, opts):
    """Load arrays for the first n_demos demo_* dirs. Outside all timing."""
    names = sorted(
        [d for d in os.listdir(dataset_dir)
         if d.startswith("demo_") and os.path.isdir(os.path.join(dataset_dir, d))],
        key=lambda x: int(x.split("_")[1]))[:n_demos]
    need_vlm = any(m in gek.VLM_METHODS for m in methods)
    need_uvd = any(m in gek.UVD_METHODS for m in methods)

    demos = []
    for name in names:
        step_files = gek._sorted_step_files(os.path.join(dataset_dir, name))
        if not step_files:
            continue
        eef_pos, eef_quat, gripper_qpos, action, gripper_pcd = gek._load_demo_arrays(
            step_files, gripperless=opts.gripperless)
        rgb = {}
        if need_vlm:
            rgb[opts.vlm_camera] = gek._load_rgb_frames(step_files, opts.vlm_camera)
        if need_uvd and opts.uvd_camera not in rgb:
            rgb[opts.uvd_camera] = gek._load_rgb_frames(step_files, opts.uvd_camera)
        demos.append(dict(name=name, T=len(step_files), eef_pos=eef_pos,
                          eef_quat=eef_quat, gripper_qpos=gripper_qpos,
                          action=action, gripper_pcd=gripper_pcd, rgb=rgb))
        print("  loaded {} (T={})".format(name, len(step_files)), flush=True)
    return demos


def _bench_method(gek, method, demos, opts, warmup, repeats):
    per_demo = []
    for demo in demos:
        for _ in range(warmup):
            _time_one_method(gek, method, demo, opts)
        samples, n_kp = [], 0
        for _ in range(repeats):
            dt, n_kp = _time_one_method(gek, method, demo, opts)
            samples.append(dt)
        med = statistics.median(samples)
        per_demo.append(dict(demo=demo["name"], T=demo["T"],
                             median_s=med, min_s=min(samples),
                             us_per_frame=med / demo["T"] * 1e6,
                             n_keypoints=n_kp))
        print("    {:<22} {:<10} T={:<6} {:8.2f} ms  {:7.2f} us/frame  {:3d} kp".format(
            method, demo["name"], demo["T"], med * 1e3,
            med / demo["T"] * 1e6, n_kp), flush=True)

    meds = [d["median_s"] for d in per_demo]
    return dict(
        method=method,
        awe_solver=opts.awe_solver if method in gek.AWE_METHODS else None,
        n_demos=len(per_demo),
        median_ms_per_demo=statistics.median(meds) * 1e3,
        mean_ms_per_demo=statistics.mean(meds) * 1e3,
        max_ms_per_demo=max(meds) * 1e3,
        median_us_per_frame=statistics.median([d["us_per_frame"] for d in per_demo]),
        mean_keypoints=statistics.mean([d["n_keypoints"] for d in per_demo]),
        mean_T=statistics.mean([d["T"] for d in per_demo]),
        per_demo=per_demo,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", required=True,
                   help="directory holding demo_*/ of per-frame npz")
    p.add_argument("--methods", nargs="+", default=["cpu"],
                   help="method names, or 'cpu' (the 10 in-process methods) or 'all'")
    p.add_argument("--demos", type=int, default=10,
                   help="how many demo_* dirs to time (default 10)")
    p.add_argument("--repeats", type=int, default=5,
                   help="timed repetitions per demo; median is reported")
    p.add_argument("--warmup", type=int, default=2,
                   help="untimed repetitions first, to pay imports/JIT")
    p.add_argument("--isolate", dest="isolate", action="store_true", default=True,
                   help="re-exec one subprocess per method (default; see module "
                        "docstring -- importing awe perturbs bspline)")
    p.add_argument("--no_isolate", dest="isolate", action="store_false",
                   help="run every method in THIS process (bspline timings "
                        "become dependent on whether awe was imported)")
    p.add_argument("--out", default=None, help="write JSON results here")
    # method parameters -- generate_extra_keypoints.py defaults
    p.add_argument("--gripperless", action="store_true")
    p.add_argument("--epsilon", type=float, default=0.02)
    p.add_argument("--const_interval", type=int, default=50)
    p.add_argument("--n_random", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snap_window", type=int, default=5)
    p.add_argument("--max_error", type=float, default=0.08)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--awe_err_threshold", type=float, default=0.2)
    p.add_argument("--awe_solver", choices=["greedy", "dp"], default="greedy")
    p.add_argument("--awe_use_gripper", action="store_true")
    p.add_argument("--orientation_threshold", type=float, default=np.pi / 4)
    p.add_argument("--uvd_camera", default="agentview")
    p.add_argument("--uvd_preprocessor", default=None)
    p.add_argument("--vlm_camera", default="agentview")
    p.add_argument("--vlm_provider", default="qwen")
    p.add_argument("--vlm_model", default=None)
    p.add_argument("--vlm_qwen_base_url", default=None)
    p.add_argument("--vlm_sample_every_n_frames", type=int, default=15)
    p.add_argument("--_child_method", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    gek = _load_generator()
    if args.uvd_preprocessor is None:
        args.uvd_preprocessor = gek.UVD_DEFAULT_PREPROCESSOR

    if args.methods == ["all"]:
        methods = list(gek.VALID_METHODS)
    elif args.methods == ["cpu"]:
        methods = list(CPU_METHODS)
    else:
        methods = args.methods
    bad = [m for m in methods if m not in gek.VALID_METHODS]
    if bad:
        raise SystemExit("Unknown method(s) {}. Valid: {}".format(
            bad, list(gek.VALID_METHODS)))

    # Parent in --isolate mode: fan out one child per method, collect their JSON.
    if args.isolate and args._child_method is None and len(methods) > 1:
        results = []
        for m in methods:
            tmp = (args.out or "/tmp/subgoal_bench") + ".{}.part.json".format(m)
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--_child_method", m, "--methods", m, "--no_isolate",
                   "--dataset_dir", args.dataset_dir,
                   "--demos", str(args.demos), "--repeats", str(args.repeats),
                   "--warmup", str(args.warmup), "--out", tmp]
            for flag in ("epsilon", "const_interval", "n_random", "seed",
                         "snap_window", "max_error", "degree",
                         "awe_err_threshold", "awe_solver",
                         "orientation_threshold", "uvd_camera", "vlm_camera",
                         "vlm_provider", "vlm_sample_every_n_frames"):
                cmd += ["--" + flag, str(getattr(args, flag))]
            if args.awe_use_gripper:
                cmd.append("--awe_use_gripper")
            if args.gripperless:
                cmd.append("--gripperless")
            print("\n=== {} (isolated process) ===".format(m), flush=True)
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print("  [FAIL] {} exited {}".format(m, rc))
                continue
            with open(tmp) as fh:
                results.extend(json.load(fh)["results"])
            os.remove(tmp)
        _report(results, args)
        return

    print("loading {} demos from {}".format(args.demos, args.dataset_dir), flush=True)
    opts = _build_opts(args)
    demos = _load_demos(gek, args.dataset_dir, args.demos, methods, opts)
    if not demos:
        raise SystemExit("no demo_* directories found in " + args.dataset_dir)

    results = []
    for m in methods:
        print("\n  timing {} ...".format(m), flush=True)
        results.append(_bench_method(gek, m, demos, opts, args.warmup, args.repeats))
    _report(results, args)


def _report(results, args):
    results = sorted(results, key=lambda r: r["median_ms_per_demo"])
    print("\n" + "=" * 78)
    print("{:<22} {:>10} {:>12} {:>10} {:>9}".format(
        "method", "ms/demo", "us/frame", "keypoints", "mean T"))
    print("-" * 78)
    for r in results:
        name = r["method"] + ("[{}]".format(r["awe_solver"]) if r["awe_solver"] else "")
        print("{:<22} {:>10.3f} {:>12.2f} {:>10.1f} {:>9.0f}".format(
            name, r["median_ms_per_demo"], r["median_us_per_frame"],
            r["mean_keypoints"], r["mean_T"]))
    print("=" * 78)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(dict(dataset_dir=args.dataset_dir, demos=args.demos,
                           repeats=args.repeats, warmup=args.warmup,
                           results=results), fh, indent=2)
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
