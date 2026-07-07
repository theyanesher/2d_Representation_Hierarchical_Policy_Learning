"""
Offline keypoint / subgoal generator — velocity-free, no simulator.

Reads the already-rendered per-timestep .npz dataset produced by
convert_dataset.py and writes ADDITIONAL goal_gripper_pcd variants computed by
the RDP-family methods (rdp, rdp_gripper, random, fixed_interval). The original
dataset is treated as READ-ONLY; new keys live in a mirror tree:

    <DATA_ROOT>/<TASK>/<demo>/<t>.npz                      (original, untouched)
    <DATA_ROOT>/EXTRA_KEYPOINTS/<TASK>/<demo>/<t>.npz      (new keys, this script)

Each new .npz holds one key per method, saved IDENTICALLY to the original
`goal_gripper_pcd` ((1, 4, 3) float32) so they are drop-in interchangeable:

    goal_gripper_pcd_rdp, goal_gripper_pcd_rdp_gripper,
    goal_gripper_pcd_random, goal_gripper_pcd_fixed_interval

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

# third_party/robogen on the path (rdp_subgoal_decomp + subgoal_decomp live there)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "robogen"))
from rdp_subgoal_decomp import compute_rdp_subgoal_gripper_pcd, VALID_METHODS

EXTRA_DIRNAME = "EXTRA_KEYPOINTS"
METHOD_KEY = {m: "goal_gripper_pcd_{}".format(m) for m in VALID_METHODS}


def _sorted_step_files(demo_dir):
    files = glob.glob(os.path.join(demo_dir, "*.npz"))
    return sorted(files, key=lambda p: int(os.path.basename(p)[:-4]))


def _load_demo_arrays(step_files):
    """Stack the per-step keys this generator needs into (T, ...) arrays."""
    eef_pos, gripper_qpos, action, gripper_pcd = [], [], [], []
    for f in step_files:
        d = np.load(f)
        eef_pos.append(d["eef_pos"][0])
        gripper_qpos.append(d["gripper_qpos"][0])
        action.append(d["action"][0])
        gripper_pcd.append(d["gripper_pcd"][0])
    return (np.asarray(eef_pos), np.asarray(gripper_qpos),
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
            with np.load(out_path) as existing:
                payload = {k: existing[k] for k in existing.files}
        for key, goal in method_goals.items():
            payload[key] = goal[t][None, :].astype(np.float32)   # (1, 4, 3)
        np.savez_compressed(out_path, **payload)


def process_demo(demo_dir, out_demo_dir, methods, opts, dump_indices=False):
    step_files = _sorted_step_files(demo_dir)
    T = len(step_files)
    if T == 0:
        return 0, {}
    eef_pos, gripper_qpos, action, gripper_pcd = _load_demo_arrays(step_files)

    method_goals = {}
    idx_record = {}
    for m in methods:
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
        idx_record[m] = switch_idxs.tolist()

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
    ap.add_argument("--dump_indices", action="store_true",
                    help="also write _keypoints.json per demo (for viser inspection)")
    args = ap.parse_args()

    methods = list(VALID_METHODS) if args.methods == ["all"] else args.methods
    bad = [m for m in methods if m not in VALID_METHODS]
    if bad:
        raise SystemExit("Unknown method(s) {}. Valid: {}".format(bad, list(VALID_METHODS)))

    task_dir = os.path.join(args.data_root, args.task)
    out_task_dir = os.path.join(args.data_root, EXTRA_DIRNAME, args.task)
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
    for i, demo in enumerate(demos):
        T, idx_record = process_demo(
            os.path.join(task_dir, demo),
            os.path.join(out_task_dir, demo),
            methods, args, dump_indices=args.dump_indices,
        )
        for m in methods:
            counts[m].append(len(idx_record.get(m, [])))
        print("[done] {} ({}/{}): T={} | {}".format(
            demo, i + 1, len(demos), T,
            " ".join("{}={}kp".format(m, len(idx_record.get(m, []))) for m in methods)))

    print("\n[summary] keypoints per demo:")
    for m in methods:
        c = np.array(counts[m]) if counts[m] else np.array([0])
        print("  {:16s} mean={:.1f}  min={}  max={}".format(m, c.mean(), int(c.min()), int(c.max())))


if __name__ == "__main__":
    main()
