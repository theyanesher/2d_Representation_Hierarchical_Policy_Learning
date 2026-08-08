"""Generate ground-truth subgoals for the high-level policy using AWE
(Automatic Waypoint Extraction, https://pypi.org/project/waypoint-extraction/).

For every demo, AWE selects a sparse set of frame indices ("waypoints") along
the trajectory such that linearly interpolating the end-effector pose between
consecutive waypoints reconstructs the full trajectory within `err_threshold`.
Each frame is then assigned the *next upcoming* waypoint's gripper keypoints
as its subgoal — the same "goal repeats until the demo passes it" convention
NpyDataset/the RDP pipeline already use for `goal_gripper_pcd`.

NOTE on the AWE package API: despite what you may have seen elsewhere, the
installed `waypoint_extraction` package does NOT expose an `extract_waypoints`
function. It exposes two selection algorithms:
    waypoint_extraction.dp_waypoint_selection(...)      # optimal, O(T^3) - slow
    waypoint_extraction.greedy_waypoint_selection(...)  # near-optimal, faster
Both work purely geometrically (env=None) as long as `actions[:, -1]` carries
the gripper open/close command and `gt_states` carries eef pos/quat — no live
simulator is required. This script uses `greedy` by default; pass --method dp
only for short demos (a few hundred frames at most).

Output format (mirrors the RDP `extra_goals_dir` convention used by
NpyDataset's goal_source="rdp"/"rdp_gripper" -- see
src/lfd3d/datasets/npy/npy_dataset.py):

    <output_dir>/
        demo_0/
            0.npz   # goal_gripper_pcd_awe  (1, 4, 3) float32
            1.npz
            ...
        demo_1/
            ...
        _logs/
            demo_0.log   # captured AWE stdout for that demo (for debugging)
            ...
        _awe_generation_meta.json

To train the high-level model on these subgoals, point a dataset config at it:
    dataset.goal_source=awe
    dataset.extra_goals_dir=<output_dir>
(after adding "awe" to VALID_GOAL_SOURCES in npy_dataset.py.)

Parallelism: each demo is processed independently (own AWE run + own npz
writes), so --num_workers > 1 processes multiple demos concurrently via a
process pool. A watcher thread in the main process polls per-demo progress
(loading -> running_awe -> assigning_goals -> writing -> done) reported by the
workers through a shared dict, and prints a periodic snapshot -- readable both
interactively and in a SLURM log file (no terminal control codes).

Example:
    pixi run python scripts/generate_awe_subgoals.py \\
        --dataset_dir /data/theya/COFFEE_PREPERATION_D1_sample \\
        --output_dir  /data/theya/COFFEE_PREPERATION_D1_sample_AWE \\
        --err_threshold 0.03 --method greedy --num_workers 8
"""

import argparse
import contextlib
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import numpy as np
from waypoint_extraction import dp_waypoint_selection, greedy_waypoint_selection


def sorted_frame_files(demo_dir):
    return sorted(
        [f for f in os.listdir(demo_dir) if f.endswith(".npz")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )


def load_demo_trajectory(demo_dir, frames):
    """Load the per-frame fields AWE and the goal-assignment step need.

    Returns:
        eef_pos:      (T, 3) float64
        eef_quat:     (T, 4) float64  (x, y, z, w)
        gripper_cmd:  (T,)   float64  last column of `action` -- open/close command
        gripper_pcd:  (T, 4, 3) float32 -- the real 4-keypoint gripper pose per
                      frame, used verbatim as the subgoal for whichever frame
                      becomes a waypoint (no pose->template reconstruction
                      needed, unlike the RL Bench fallback in
                      run_gmm_on_dataset_batch_optimized.py -- this dataset
                      already ships an accurate per-frame gripper_pcd).
    """
    num_frames = len(frames)
    eef_pos = np.zeros((num_frames, 3), dtype=np.float64)
    eef_quat = np.zeros((num_frames, 4), dtype=np.float64)
    gripper_cmd = np.zeros((num_frames,), dtype=np.float64)
    gripper_pcd = np.zeros((num_frames, 4, 3), dtype=np.float32)

    for t, fname in enumerate(frames):
        d = np.load(os.path.join(demo_dir, fname), allow_pickle=True)
        eef_pos[t] = d["eef_pos"][0]
        eef_quat[t] = d["eef_quat"][0]
        gripper_cmd[t] = d["action"][0][-1]
        gripper_pcd[t] = d["gripper_pcd"][0]

    return eef_pos, eef_quat, gripper_cmd, gripper_pcd


def select_waypoints(  # noqa: PLR0913, PLR0917
    eef_pos, eef_quat, gripper_cmd, err_threshold, method, pos_only
):
    """Run AWE over one demo's trajectory and return a sorted list of
    frame indices (0-indexed, always including the last frame)."""
    num_frames = eef_pos.shape[0]

    # `actions[:, :3]` supplies the waypoint positions for geometric interpolation;
    # `actions[:, -1]` supplies the gripper open/close toggle signal. Passing the
    # eef position itself (rather than a delta action) is correct here since AWE's
    # geometric error is computed against absolute end-effector positions.
    actions = np.concatenate([eef_pos, gripper_cmd[:, None]], axis=1)  # (T, 4)
    gt_states = [
        {"robot0_eef_pos": eef_pos[t], "robot0_eef_quat": eef_quat[t]}
        for t in range(num_frames)
    ]

    if method == "dp":
        waypoints = dp_waypoint_selection(
            env=None,
            actions=actions,
            gt_states=gt_states,
            err_threshold=err_threshold,
            pos_only=pos_only,
        )
    elif method == "greedy":
        waypoints = greedy_waypoint_selection(
            env=None,
            actions=actions,
            gt_states=gt_states,
            err_threshold=err_threshold,
            geometry=True,  # stay on the geometric (no-simulator) path
            pos_only=pos_only,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return sorted({int(w) for w in waypoints})


def assign_subgoals(waypoints, gripper_pcd):
    """For every frame, assign the gripper keypoints of the next upcoming
    waypoint (inclusive) as its subgoal -- goals repeat until the demo passes
    each waypoint, matching the existing goal_gripper_pcd convention."""
    num_frames = gripper_pcd.shape[0]
    goals = np.zeros((num_frames, 4, 3), dtype=np.float32)
    wp_idx = 0
    for t in range(num_frames):
        while waypoints[wp_idx] < t:
            wp_idx += 1
        goals[t] = gripper_pcd[waypoints[wp_idx]]
    return goals


def _set_stage(status, demo_name, stage, **extra):
    """Update this demo's entry in the shared status dict (Manager proxies
    need whole-value reassignment, not in-place mutation, to sync)."""
    entry = dict(status.get(demo_name, {}))
    entry["stage"] = stage
    entry["t"] = time.time()
    entry.update(extra)
    status[demo_name] = entry


def _process_demo_worker(  # noqa: PLR0913, PLR0917
    demo_name,
    dataset_dir,
    output_dir,
    err_threshold,
    method,
    pos_only,
    key_suffix,
    status,
):
    """Runs in a worker process: full pipeline for one demo, reporting stage
    transitions into `status` (a Manager dict shared with the main process)
    for the watcher thread to display. Errors are caught and reported rather
    than raised, so one bad demo doesn't take down the whole pool."""
    t_start = time.time()
    log_buf = io.StringIO()
    try:
        demo_dir = os.path.join(dataset_dir, demo_name)
        out_demo_dir = os.path.join(output_dir, demo_name)

        _set_stage(status, demo_name, "loading", start=t_start)
        frames = sorted_frame_files(demo_dir)
        eef_pos, eef_quat, gripper_cmd, gripper_pcd = load_demo_trajectory(
            demo_dir, frames
        )

        _set_stage(status, demo_name, "running_awe", frames=len(frames))
        with contextlib.redirect_stdout(log_buf):
            waypoints = select_waypoints(
                eef_pos, eef_quat, gripper_cmd, err_threshold, method, pos_only
            )

        _set_stage(status, demo_name, "assigning_goals", waypoints=len(waypoints))
        goals = assign_subgoals(waypoints, gripper_pcd)

        _set_stage(status, demo_name, "writing")
        os.makedirs(out_demo_dir, exist_ok=True)
        key = f"goal_gripper_pcd_{key_suffix}"
        for t, fname in enumerate(frames):
            np.savez(
                os.path.join(out_demo_dir, fname),
                **{
                    key: goals[t][None]
                },  # (1, 4, 3), matches the per-frame npz convention
            )

        log_dir = os.path.join(output_dir, "_logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(
            os.path.join(log_dir, f"{demo_name}.log"), "w", encoding="utf-8"
        ) as f:
            f.write(log_buf.getvalue())

        _set_stage(
            status,
            demo_name,
            "done",
            waypoints=len(waypoints),
            frames=len(frames),
            elapsed=time.time() - t_start,
        )
        return demo_name, waypoints, len(frames), None
    except Exception as e:  # noqa: BLE001 - report, don't kill the pool
        _set_stage(
            status, demo_name, "error", error=str(e), elapsed=time.time() - t_start
        )
        return demo_name, None, None, str(e)


def _watch_progress(status, total, interval):
    """Background thread: print a periodic snapshot of every demo's stage.
    Plain-text, timestamp-free lines -- safe both on a live terminal and
    piped into a SLURM log file (no carriage-return / cursor tricks)."""
    stop_stages = {"done", "error"}
    while True:
        time.sleep(interval)
        snapshot = dict(status)
        n_done = sum(1 for v in snapshot.values() if v.get("stage") in stop_stages)
        lines = [f"[watch] {n_done}/{total} demos finished"]
        for name in sorted(snapshot, key=lambda n: int(n.split("_")[1])):
            entry = snapshot[name]
            stage = entry.get("stage", "?")
            if stage == "done":
                lines.append(
                    f"  {name}: done ({entry.get('frames')} frames, "
                    f"{entry.get('waypoints')} waypoints, "
                    f"{entry.get('elapsed', 0):.1f}s)"
                )
            elif stage == "error":
                lines.append(f"  {name}: ERROR - {entry.get('error')}")
            else:
                elapsed = time.time() - entry.get("start", time.time())
                lines.append(f"  {name}: {stage} ({elapsed:.1f}s so far)")
        print("\n".join(lines), flush=True)
        if n_done >= total:
            return


def main():  # noqa: PLR0914, PLR0915
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Root containing demo_*/ of per-frame npz (needs "
        "eef_pos, eef_quat, action, gripper_pcd per frame).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Parallel output root; one demo_N/ per input demo, "
        "each frame npz holding only goal_gripper_pcd_<key_suffix>.",
    )
    parser.add_argument(
        "--err_threshold",
        type=float,
        required=True,
        help="Max allowed reconstruction error (position in meters, "
        "+ rotation in radians if not --pos_only) before AWE adds "
        "another waypoint. Typical range 0.01-0.05; tune by "
        "visualizing output with visualize_npz_demo_matplotlib.py.",
    )
    parser.add_argument(
        "--method",
        choices=["greedy", "dp"],
        default="greedy",
        help="greedy (default): fast, near-optimal. dp: optimal "
        "(fewest waypoints for the threshold) but O(T^3) -- only "
        "use on short demos (roughly <= a few hundred frames).",
    )
    parser.add_argument(
        "--pos_only",
        action="store_true",
        default=False,
        help="Ignore end-effector rotation in the reconstruction error "
        "(position-only AWE). Default off -- orientation is scored too.",
    )
    parser.add_argument(
        "--key_suffix",
        default="awe",
        help="Output key becomes goal_gripper_pcd_<key_suffix>.",
    )
    parser.add_argument(
        "--start_demo",
        type=int,
        default=0,
        help="Skip demo_* dirs with index < start_demo.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Only process this many demo_* directories after filtering.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Demos processed concurrently via a process pool. Each demo "
        "is independent (own AWE run + own npz writes), so this scales "
        "close to linearly -- match it to --cpus-per-task on a SLURM job.",
    )
    parser.add_argument(
        "--progress_interval",
        type=float,
        default=10.0,
        help="Seconds between watcher progress snapshots.",
    )
    args = parser.parse_args()

    demo_dirs = sorted(
        [
            e
            for e in os.listdir(args.dataset_dir)
            if e.startswith("demo_")
            and os.path.isdir(os.path.join(args.dataset_dir, e))
        ],
        key=lambda x: int(x.split("_")[1]),
    )
    if args.start_demo > 0:
        demo_dirs = [d for d in demo_dirs if int(d.split("_")[1]) >= args.start_demo]
    if args.max_files is not None:
        demo_dirs = demo_dirs[: args.max_files]

    if not demo_dirs:
        print(f"No demo_* directories found under {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume: skip demos whose output already has one npz per source frame.
    pending = []
    for demo_name in demo_dirs:
        demo_dir = os.path.join(args.dataset_dir, demo_name)
        out_demo_dir = os.path.join(args.output_dir, demo_name)
        frames = sorted_frame_files(demo_dir)
        if os.path.isdir(out_demo_dir) and len(
            [f for f in os.listdir(out_demo_dir) if f.endswith(".npz")]
        ) == len(frames):
            print(f"  {demo_name}: skipping (already processed)")
            continue
        pending.append(demo_name)

    print(
        f"Processing {len(pending)}/{len(demo_dirs)} demo(s) from {args.dataset_dir} "
        f"(method={args.method}, err_threshold={args.err_threshold}, "
        f"pos_only={args.pos_only}, num_workers={args.num_workers})"
    )

    meta = {
        "dataset_dir": args.dataset_dir,
        "err_threshold": args.err_threshold,
        "method": args.method,
        "pos_only": args.pos_only,
        "key_suffix": args.key_suffix,
        "waypoints": {},
        "errors": {},
    }

    if pending:
        manager = Manager()
        status = manager.dict()
        watcher = threading.Thread(
            target=_watch_progress,
            args=(status, len(pending), args.progress_interval),
            daemon=True,
        )
        watcher.start()

        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [
                pool.submit(
                    _process_demo_worker,
                    demo_name,
                    args.dataset_dir,
                    args.output_dir,
                    args.err_threshold,
                    args.method,
                    args.pos_only,
                    args.key_suffix,
                    status,
                )
                for demo_name in pending
            ]
            for future in as_completed(futures):
                demo_name, waypoints, n_frames, error = future.result()
                if error is not None:
                    meta["errors"][demo_name] = error
                    print(f"  {demo_name}: ERROR - {error}")
                else:
                    meta["waypoints"][demo_name] = waypoints
                    print(
                        f"  {demo_name}: {n_frames} frames -> "
                        f"{len(waypoints)} waypoints"
                    )

        watcher.join(timeout=args.progress_interval + 5)

    meta_path = os.path.join(args.output_dir, "_awe_generation_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if meta["errors"]:
        print(
            f"[done] {len(meta['errors'])} demo(s) FAILED -- see meta json / _logs/. "
            f"Successful subgoals + metadata written to {args.output_dir}"
        )
    else:
        print(f"[done] wrote subgoals + metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
