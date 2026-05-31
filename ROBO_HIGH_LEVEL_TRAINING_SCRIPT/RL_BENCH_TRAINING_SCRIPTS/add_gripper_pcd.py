#!/usr/bin/env python3
"""One-time, in-place addition of the `gripper_pcd` key to an RL Bench dataset.

RL Bench .npz frames store the goal gripper keypoints (`goal_gripper_pcd`) but
NOT the *current* gripper keypoints (`gripper_pcd`) that MimicGen datasets have
and that the high-level (Articubot) training loop reads as `action_pcd`. This
script reconstructs `gripper_pcd` (1, 4, 3) from each frame's `state` vector
([x, y, z, qx, qy, qz, qw, gripper_open]) and writes it back into the same .npz,
permanently, so the dataloader reads it like MimicGen with zero per-sample cost.

Design (see RL_BENCH_TRAINING_SCRIPTS launch script):
  - Operates directly on the canonical /ocean source dir (Option 2).
  - Atomic per file: write a temp .npz in the same dir, then os.replace() — an
    interrupted run can never leave a half-written frame.
  - Idempotent / resumable: a frame that already has `gripper_pcd` is skipped.
  - Parallel across frames (multiprocessing).
  - On full success, drops a `.gripper_pcd_added` marker at --data_dir so the
    launch script skips this step on every subsequent run.

The 4-point template + open-width logic mirror exactly how the dataset's
`goal_gripper_pcd` was generated (verified against the stored goal points:
|wrist-center| = 0.08, center = midpoint(fingers), fingers perpendicular to the
wrist axis, finger gap = 0.102 at open = 1). Point order is
[left_finger, right_finger, wrist, grasp_center]; index 3 (grasp_center = finger
midpoint) is the canonical point the training loop reads.
"""
import argparse
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None

MARKER_NAME = ".gripper_pcd_added"

# Canonical gripper keypoints in the EE-local frame (metres).
_GRIPPER_TEMPLATE_4 = np.array(
    [
        [0.0, -0.0405, 0.0800],  # left finger
        [0.0, 0.0405, 0.0800],   # right finger
        [0.0, 0.0, 0.0000],      # wrist (EE origin)
        [0.0, 0.0, 0.0800],      # grasp center (finger midpoint)
    ],
    dtype=np.float32,
)


def _quat_to_rotmat(q):
    """Unit quaternion [qx, qy, qz, qw] (scalar-last) -> (3, 3) rotation."""
    q = np.asarray(q, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def gripper_pcd_from_state(state):
    """RL Bench `state` ([x,y,z,qx,qy,qz,qw,gripper_open]) -> (4, 3) world frame."""
    state = np.asarray(state, dtype=np.float32)
    t = state[:3]
    R = _quat_to_rotmat(state[3:7])
    open_val = float(np.clip(state[7], 0.0, 1.0))

    p = _GRIPPER_TEMPLATE_4.copy()
    delta = 0.021 * (open_val - 0.5)
    p[0, 1] -= delta
    p[1, 1] += delta
    p[3, :] = 0.5 * (p[0, :] + p[1, :])  # grasp center = finger midpoint

    return (p @ R.T + t).astype(np.float32)  # (4, 3)


def convert_one(path_str):
    """Add gripper_pcd to one .npz in place (atomic). Returns a status string."""
    path = Path(path_str)
    try:
        with np.load(path, allow_pickle=True) as d:
            if "gripper_pcd" in d.files:
                return "skipped"
            arrays = {k: d[k] for k in d.files}

        state = np.asarray(arrays["state"], dtype=np.float32)  # (1, 8)
        gp = gripper_pcd_from_state(state[0])                   # (4, 3)
        if gp.shape != (4, 3) or not np.isfinite(gp).all():
            return f"ERROR {path}: bad gripper_pcd {gp.shape}"
        arrays["gripper_pcd"] = gp[None].astype(np.float32)    # (1, 4, 3)

        # Atomic write: temp file in the same dir, then replace.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, path)  # atomic on the same filesystem
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return "converted"
    except Exception as e:  # noqa: BLE001
        return f"ERROR {path}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True, type=Path,
                    help="Dataset dir containing demo_*/N.npz.")
    ap.add_argument("--workers", type=int,
                    default=len(os.sched_getaffinity(0)),
                    help="Parallel worker processes (default: available CPUs).")
    args = ap.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        sys.exit(f"[add_gripper_pcd] data_dir does not exist: {data_dir}")

    marker = data_dir / MARKER_NAME
    if marker.exists():
        print(f"[add_gripper_pcd] marker {marker} exists — nothing to do.")
        return

    frames = sorted(
        (str(p) for d in sorted(data_dir.iterdir()) if d.is_dir()
         for p in d.glob("*.npz")),
    )
    if not frames:
        sys.exit(f"[add_gripper_pcd] no .npz frames found under {data_dir}")

    print(f"[add_gripper_pcd] {len(frames)} frames, {args.workers} workers, "
          f"dir={data_dir}")
    start = time.time()

    counts = {"converted": 0, "skipped": 0}
    errors = []
    with Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(convert_one, frames, chunksize=8)
        # tqdm progress bar when available; otherwise periodic prints. In a
        # SLURM log (non-TTY) tqdm still works — set mininterval so it refreshes
        # at most every few seconds instead of spamming the .out file.
        if tqdm is not None:
            results = tqdm(
                results, total=len(frames), desc="add_gripper_pcd",
                unit="frame", mininterval=2.0,
            )
        for i, status in enumerate(results, 1):
            if status in counts:
                counts[status] += 1
            else:
                errors.append(status)
            if tqdm is not None:
                results.set_postfix(
                    conv=counts["converted"], skip=counts["skipped"],
                    err=len(errors), refresh=False,
                )
            elif i % 1000 == 0 or i == len(frames):
                print(f"  [{i}/{len(frames)}] converted={counts['converted']} "
                      f"skipped={counts['skipped']} errors={len(errors)} "
                      f"({time.time() - start:.0f}s)")

    if errors:
        print(f"[add_gripper_pcd] FAILED with {len(errors)} error(s); "
              f"marker NOT written. First few:")
        for e in errors[:10]:
            print("   ", e)
        sys.exit(1)

    marker.write_text(
        f"gripper_pcd added to {counts['converted']} frames "
        f"({counts['skipped']} already had it) on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    print(f"[add_gripper_pcd] done in {time.time() - start:.0f}s. "
          f"converted={counts['converted']} skipped={counts['skipped']}. "
          f"Marker: {marker}")


if __name__ == "__main__":
    main()
