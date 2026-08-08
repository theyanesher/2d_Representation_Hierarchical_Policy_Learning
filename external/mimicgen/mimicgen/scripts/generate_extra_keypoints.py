"""
Offline keypoint / subgoal generator — velocity-free, no simulator.

Reads the already-rendered per-timestep .npz dataset produced by
convert_dataset.py and writes ADDITIONAL goal_gripper_pcd variants computed by
the RDP-family methods (rdp, rdp_gripper, random, fixed_interval), the
B-spline knot method (bspline), and AWE / Automatic Waypoint Extraction
(awe). The original dataset is treated as READ-ONLY; new keys live in a
mirror tree, one per requested method-set so different --methods runs never
collide or overwrite each other:

    <DATA_ROOT>/<TASK>/<demo>/<t>.npz                                (original, untouched)
    <DATA_ROOT>/EXTRA_KEYPOINTS_<method1>_<method2>/<TASK>/<demo>/<t>.npz   (new keys, this script)

Each new .npz holds one key per method, saved IDENTICALLY to the original
`goal_gripper_pcd` ((1, 4, 3) float32) so they are drop-in interchangeable:

    goal_gripper_pcd_rdp, goal_gripper_pcd_rdp_gripper,
    goal_gripper_pcd_random, goal_gripper_pcd_fixed_interval,
    goal_gripper_pcd_bspline, goal_gripper_pcd_awe

Incremental & safe: adding a method later loads the existing mirror .npz, keeps
all prior keys verbatim, adds the new ones, and rewrites the whole file. Nothing
irreplaceable lives in the mirror tree, so a mistake costs only a recompute.

Example:
    python generate_extra_keypoints.py \
        --data_root /scratch/.../GROOT_STYLE_DATASET/D2 \
        --task COFFEE_PREPERATION_D1 \
        --methods rdp rdp_gripper \
        --episodes 5            # inspect a few first; drop for the full run
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

# third_party/robogen on the path (rdp_subgoal_decomp + subgoal_decomp +
# bspline_subgoal_decomp + awe_subgoal_decomp live there)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "robogen"))
from rdp_subgoal_decomp import compute_rdp_subgoal_gripper_pcd, VALID_METHODS as RDP_METHODS
from bspline_subgoal_decomp import compute_bspline_subgoal_gripper_pcd, VALID_METHODS as BSPLINE_METHODS
from awe_subgoal_decomp import compute_awe_subgoal_gripper_pcd

EXTRA_DIRNAME_PREFIX = "EXTRA_KEYPOINTS"
# awe_subgoal_decomp's own VALID_METHODS=("greedy","dp") names its internal
# solver, not an output key -- "awe" is the single goal_gripper_pcd_awe key
# this script produces; --awe_solver below picks which of greedy/dp computes it.
AWE_METHODS = ("awe",)
VALID_METHODS = RDP_METHODS + BSPLINE_METHODS + AWE_METHODS
METHOD_KEY = {m: "goal_gripper_pcd_{}".format(m) for m in VALID_METHODS}


def _extra_dirname(methods):
    """EXTRA_KEYPOINTS_<method1>_<method2>_... -- keeps different --methods
    runs (e.g. rdp-only vs. bspline-only vs. all) in separate mirror trees so
    they never collide or partially overwrite each other."""
    return "_".join([EXTRA_DIRNAME_PREFIX] + list(methods))


def _sorted_step_files(demo_dir):
    files = glob.glob(os.path.join(demo_dir, "*.npz"))
    return sorted(files, key=lambda p: int(os.path.basename(p)[:-4]))


def _load_demo_arrays(step_files):
    """Stack the per-step keys this generator needs into (T, ...) arrays."""
    eef_pos, eef_quat, gripper_qpos, action, gripper_pcd = [], [], [], [], []
    for f in step_files:
        d = np.load(f)
        eef_pos.append(d["eef_pos"][0])
        eef_quat.append(d["eef_quat"][0])
        gripper_qpos.append(d["gripper_qpos"][0])
        action.append(d["action"][0])
        gripper_pcd.append(d["gripper_pcd"][0])
    return (np.asarray(eef_pos), np.asarray(eef_quat), np.asarray(gripper_qpos),
            np.asarray(action), np.asarray(gripper_pcd, dtype=np.float32))


def _write_mirror(out_demo_dir, method_goals, n_steps):
    """Write per-step mirror .npz, preserving any keys already there.

    method_goals: {key_name -> (T, 4, 3) float32}
    """
    os.makedirs(out_demo_dir, exist_ok=True)
    for t in range(n_steps):
        out_path = os.path.join(out_demo_dir, "{}.npz".format(t))
        payload = {}
        if os.path.exists(out_path):                      # preserve-keys rewrite
            try:
                with np.load(out_path) as existing:
                    payload = {k: existing[k] for k in existing.files}
            except Exception:                             # truncated by a mid-write kill
                payload = {}
        for key, goal in method_goals.items():
            payload[key] = goal[t][None, :].astype(np.float32)   # (1, 4, 3)
        np.savez_compressed(out_path, **payload)


def _demo_is_complete(demo_dir, out_demo_dir, methods, dump_indices):
    """True iff out_demo_dir already holds a finished result for `methods`.

    Completion signals, cheapest-first:
      - every mirror step file 0..T-1 exists,
      - the last mirror file carries all requested method keys (so a run that
        only had *some* methods before is treated as incomplete and re-run),
      - if dump_indices, _keypoints.json exists with matching T (written LAST,
        so its presence proves the demo finished, not just wrote some steps).
    A demo killed mid-write fails one of these and is regenerated.
    """
    step_files = _sorted_step_files(demo_dir)
    T = len(step_files)
    if T == 0:
        return True                                       # nothing to generate
    for t in range(T):
        if not os.path.exists(os.path.join(out_demo_dir, "{}.npz".format(t))):
            return False
    last = os.path.join(out_demo_dir, "{}.npz".format(T - 1))
    try:
        with np.load(last) as d:
            have = set(d.files)
    except Exception:
        return False
    if not all(METHOD_KEY[m] in have for m in methods):
        return False
    if dump_indices:
        kp = os.path.join(out_demo_dir, "_keypoints.json")
        if not os.path.exists(kp):
            return False
        try:
            with open(kp) as fh:
                if json.load(fh).get("T") != T:
                    return False
        except Exception:
            return False
    return True


def process_demo(demo_dir, out_demo_dir, methods, opts, dump_indices=False):
    step_files = _sorted_step_files(demo_dir)
    T = len(step_files)
    if T == 0:
        return 0, {}
    eef_pos, eef_quat, gripper_qpos, action, gripper_pcd = _load_demo_arrays(step_files)

    method_goals = {}
    idx_record = {}
    for m in methods:
        if m in BSPLINE_METHODS:
            goal, switch_idxs = compute_bspline_subgoal_gripper_pcd(
                gripper_pcd=gripper_pcd,
                eef_pos=eef_pos,
                method=m,
                max_error=opts.max_error,
                degree=opts.degree,
                return_switch_idxs=True,
            )
        elif m in AWE_METHODS:
            goal, switch_idxs = compute_awe_subgoal_gripper_pcd(
                gripper_pcd=gripper_pcd,
                eef_pos=eef_pos,
                eef_quat=eef_quat,
                actions=action,
                err_threshold=opts.awe_err_threshold,
                method=opts.awe_solver,
                pos_only=False,
                return_switch_idxs=True,
            )
        else:
            goal, switch_idxs = compute_rdp_subgoal_gripper_pcd(
                gripper_pcd=gripper_pcd,
                eef_pos=eef_pos,
                method=m,
                eef_qpos=gripper_qpos,
                actions=action,
                epsilon=opts.epsilon,
                interval=opts.interval,
                n_random=opts.n_random,
                seed=opts.seed,
                snap_window=opts.snap_window,
                return_switch_idxs=True,
            )
        method_goals[METHOD_KEY[m]] = goal
        idx_record[m] = [int(x) for x in switch_idxs]

    _write_mirror(out_demo_dir, method_goals, T)

    if dump_indices:
        with open(os.path.join(out_demo_dir, "_keypoints.json"), "w") as fh:
            json.dump({"T": T, "keypoints": idx_record}, fh, indent=2)

    return T, idx_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="dataset root that holds <TASK>/ folders (e.g. .../GROOT_STYLE_DATASET/D2)")
    ap.add_argument("--task", required=True,
                    help="task folder name, e.g. COFFEE_PREPERATION_D1")
    ap.add_argument("--methods", nargs="+", default=["rdp"],
                    help="any of {} or 'all'".format(list(VALID_METHODS)))
    ap.add_argument("--episodes", "-n", type=int, default=None,
                    help="process only the first N demos (for inspection)")
    ap.add_argument("--epsilon", type=float, default=0.02, help="RDP tolerance (metres)")
    ap.add_argument("--interval", type=int, default=None,
                    help="fixed_interval step (default: T//20)")
    ap.add_argument("--n_random", type=int, default=20, help="keypoints for 'random'")
    ap.add_argument("--seed", type=int, default=0, help="seed for 'random'")
    ap.add_argument("--snap_window", type=int, default=5,
                    help="frames within which rdp_gripper snaps to a gripper transition")
    ap.add_argument("--max_error", type=float, default=0.08,
                    help="bspline: max-abs (Chebyshev) EEF reconstruction error budget, metres")
    ap.add_argument("--degree", type=int, default=3,
                    help="bspline: spline degree (3 = cubic, matches bspline-policy's default)")
    ap.add_argument("--awe_err_threshold", type=float, default=0.3,
                    help="awe: max reconstruction error (position in metres, "
                         "+ rotation in radians) before AWE adds another waypoint")
    ap.add_argument("--awe_solver", choices=["greedy", "dp"], default="dp",
                    help="awe: greedy (fast, near-optimal) or dp (optimal, "
                         "O(T^3) -- short demos only, roughly <= a few hundred frames)")
    ap.add_argument("--dump_indices", action="store_true",
                    help="also write _keypoints.json per demo (for viser inspection)")
    ap.add_argument("--force", action="store_true",
                    help="recompute every demo even if it looks complete (disable resume)")
    args = ap.parse_args()

    methods = list(VALID_METHODS) if args.methods == ["all"] else args.methods
    bad = [m for m in methods if m not in VALID_METHODS]
    if bad:
        raise SystemExit("Unknown method(s) {}. Valid: {}".format(bad, list(VALID_METHODS)))
    # canonical order so e.g. --methods bspline rdp and --methods rdp bspline
    # land in the same mirror tree instead of silently forking into two.
    methods = sorted(methods, key=list(VALID_METHODS).index)

    task_dir = os.path.join(args.data_root, args.task)
    out_task_dir = os.path.join(args.data_root, _extra_dirname(methods), args.task)
    if not os.path.isdir(task_dir):
        raise SystemExit("Task dir not found: {}".format(task_dir))

    demos = sorted(
        [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))],
        key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 1 << 30,
    )
    if args.episodes is not None:
        demos = demos[:args.episodes]

    print("[extra-keypoints] methods={} | {} demos | src={}".format(methods, len(demos), task_dir))
    print("[extra-keypoints] writing -> {}".format(out_task_dir))

    counts = {m: [] for m in methods}
    n_skipped = 0
    for i, demo in enumerate(demos):
        demo_dir = os.path.join(task_dir, demo)
        out_demo_dir = os.path.join(out_task_dir, demo)
        if not args.force and _demo_is_complete(demo_dir, out_demo_dir, methods, args.dump_indices):
            n_skipped += 1
            print("[skip] {} ({}/{}): already complete".format(demo, i + 1, len(demos)))
            continue
        T, idx_record = process_demo(
            demo_dir, out_demo_dir, methods, args, dump_indices=args.dump_indices,
        )
        for m in methods:
            counts[m].append(len(idx_record.get(m, [])))
        print("[done] {} ({}/{}): T={} | {}".format(
            demo, i + 1, len(demos), T,
            " ".join("{}={}kp".format(m, len(idx_record.get(m, []))) for m in methods)))

    if n_skipped:
        print("[resume] skipped {}/{} already-complete demos (use --force to recompute)".format(
            n_skipped, len(demos)))

    print("\n[summary] keypoints per demo:")
    for m in methods:
        c = np.array(counts[m]) if counts[m] else np.array([0])
        print("  {:16s} mean={:.1f}  min={}  max={}".format(m, c.mean(), int(c.min()), int(c.max())))


if __name__ == "__main__":
    main()
