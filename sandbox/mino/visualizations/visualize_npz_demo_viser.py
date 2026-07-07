#!/usr/bin/env python3
"""Interactive viser viewer for one demo directory of per-timestep .npz files
produced by the MimicGen converter (external/mimicgen/mimicgen/scripts/convert_dataset.py).

This is the MimicGen/robosuite counterpart to visualize_npz_demo_RL_BENCH.py.
Unlike that RLBench viewer, it does NOT reconstruct the gripper from `state`
(MimicGen `state` is 10-D, not the RLBench 8-D pose+open layout). Instead it
reads the stored arrays directly — the same schema the open3d renderer
visualize_npz_demo.py uses:

  - scene point cloud  (gray)   from `point_cloud`       (N, 3)
  - current gripper    (red)    from `gripper_pcd`       (4, 3)
  - goal gripper       (green)  from --goal_key          (4, 3)

A timestep slider scrubs the demo. Browser-based, so it works headless.

The goal can come from a mirror tree (e.g. EXTRA_KEYPOINTS) via --goal_dir,
while the scene + gripper still come from --demo_dir. Files are paired by name.

Usage:
    # BOCPD goal (lives in the original npz):
    python visualize_npz_demo_viser.py --demo_dir .../D2/COFFEE_PREPERATION_D1/demo_0

    # An EXTRA_KEYPOINTS method (goal in the mirror tree):
    python visualize_npz_demo_viser.py \
        --demo_dir  .../D2/COFFEE_PREPERATION_D1/demo_0 \
        --goal_dir  .../D2/EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1/demo_0 \
        --goal_key  goal_gripper_pcd_rdp
Then open the printed http://localhost:<port> URL (via SSH tunnel if remote).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import viser


def _squeeze(x):
    x = np.asarray(x)
    return x[0] if x.ndim and x.shape[0] == 1 else x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", required=True, type=Path,
                   help="Directory of N.npz files for one demo (e.g. .../demo_0).")
    p.add_argument("--goal_key", type=str, default="goal_gripper_pcd",
                   help="npz key for the goal gripper PCD (e.g. goal_gripper_pcd_rdp).")
    p.add_argument("--goal_dir", type=Path, default=None,
                   help="Read --goal_key from this mirror demo dir (e.g. the EXTRA_KEYPOINTS "
                        "demo dir) instead of --demo_dir. Scene + gripper still come from --demo_dir.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--pcd_subsample", type=int, default=1,
                   help="Keep every Nth scene point if rendering is slow.")
    args = p.parse_args()

    npz_files = sorted(args.demo_dir.glob("*.npz"), key=lambda q: int(q.stem))
    if not npz_files:
        raise SystemExit(f"No .npz files in {args.demo_dir}")

    scenes, grippers, goals = [], [], []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        pc = _squeeze(d["point_cloud"]).astype(np.float32)            # (N, 3)
        if args.pcd_subsample > 1:
            pc = pc[::args.pcd_subsample]
        scenes.append(pc)
        grippers.append(_squeeze(d["gripper_pcd"]).astype(np.float32))  # (4, 3)
        gd = np.load(args.goal_dir / f.name, allow_pickle=True) if args.goal_dir is not None else d
        goals.append(_squeeze(gd[args.goal_key]).astype(np.float32))    # (4, 3)
    T = len(npz_files)
    goal_src = args.goal_dir if args.goal_dir is not None else args.demo_dir
    print(f"Loaded {T} timesteps from {args.demo_dir}")
    print(f"Goal '{args.goal_key}' from {goal_src}")

    server = viser.ViserServer(port=args.port)
    print(f"[viser] Open http://localhost:{args.port}. Scrub the Timestep slider. Ctrl+C to exit.")
    t_slider = server.gui.add_slider("Timestep", min=0, max=T - 1, step=1, initial_value=0)

    def render(t):
        server.scene.add_point_cloud(
            name="scene_pcd", points=scenes[t],
            colors=np.tile([180, 180, 180], (scenes[t].shape[0], 1)).astype(np.uint8),
            point_size=0.004)
        server.scene.add_point_cloud(
            name="gripper", points=grippers[t],
            colors=np.tile([255, 0, 0], (4, 1)).astype(np.uint8), point_size=0.020)
        server.scene.add_point_cloud(
            name="goal_gripper", points=goals[t],
            colors=np.tile([0, 200, 0], (4, 1)).astype(np.uint8), point_size=0.020)

    @t_slider.on_update
    def _(_):
        render(t_slider.value)

    render(0)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[viser] Exiting.")
        server.stop()


if __name__ == "__main__":
    main()
