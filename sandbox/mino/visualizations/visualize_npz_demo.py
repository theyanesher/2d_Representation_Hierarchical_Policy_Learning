#!/usr/bin/env python3
"""Render a video from one demo directory of per-timestep .npz files produced by
external/mimicgen/mimicgen/scripts/convert_dataset.py.

Each frame stitches together:
    [ open3d render of (point_cloud + gripper_pcd + goal_gripper_pcd) | rgb_agentview ]

The point cloud is white, the current gripper is red, the (sub)goal gripper is
green. Frame indices are taken from the .npz filename stems and sorted
numerically (0.npz, 1.npz, ..., 248.npz).

Usage:
    python visualize_npz_demo.py \
        --demo_dir .../LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2/Coffee_D2/demo_0 \
        --out      ./demo_0.mp4 \
        --fps      10
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import imageio.v2 as imageio
import cv2
from tqdm import tqdm


def _squeeze(x):
    return np.asarray(x)[0] if np.asarray(x).ndim and np.asarray(x).shape[0] == 1 else np.asarray(x)


def _make_pcd(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (points.shape[0], 1)))
    return pcd


def _render_pcd_frame(point_cloud, gripper_pcd, goal_gripper_pcd, width, height, view):
    geoms = [
        _make_pcd(point_cloud, [0.85, 0.85, 0.85]),
        _make_pcd(gripper_pcd, [1.0, 0.0, 0.0]),
        _make_pcd(goal_gripper_pcd, [0.0, 1.0, 0.0]),
    ]
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geoms:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.set_lookat(view["lookat"])
    ctr.set_front(view["front"])
    ctr.set_up(view["up"])
    ctr.set_zoom(view["zoom"])
    vis.poll_events()
    vis.update_renderer()
    img = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
    vis.destroy_window()
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", required=True, type=Path,
                   help="Directory containing N.npz files for one demo (e.g. .../demo_0).")
    p.add_argument("--out", required=True, type=Path, help="Output .mp4 path.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--height", type=int, default=480, help="Per-panel render height.")
    p.add_argument("--no_rgb", action="store_true",
                   help="Skip the agentview RGB panel and only render the point cloud view.")
    args = p.parse_args()

    npz_files = sorted(args.demo_dir.glob("*.npz"), key=lambda p: int(p.stem))
    if not npz_files:
        raise SystemExit(f"No .npz files in {args.demo_dir}")

    # Pre-compute a stable camera lookat from the first frame's PC so the view
    # doesn't drift across timesteps.
    first = np.load(npz_files[0], allow_pickle=True)
    pc0 = _squeeze(first["point_cloud"])
    view = dict(
        lookat=np.mean(pc0, axis=0).tolist(),
        front=[1.0, 1.0, 0.6],
        up=[0.0, 0.0, 1.0],
        zoom=0.55,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264", quality=8)

    H = args.height
    for f in tqdm(npz_files, desc="frames"):
        d = np.load(f, allow_pickle=True)
        pc   = _squeeze(d["point_cloud"])
        gp   = _squeeze(d["gripper_pcd"])
        gg   = _squeeze(d["goal_gripper_pcd"])
        pcd_img = _render_pcd_frame(pc, gp, gg, width=H, height=H, view=view)

        if args.no_rgb:
            frame = pcd_img
        else:
            rgb = _squeeze(d["rgb_agentview"])
            if rgb.shape[0] != H:
                rgb = cv2.resize(rgb, (H, H), interpolation=cv2.INTER_AREA)
            frame = np.concatenate([pcd_img, rgb], axis=1)

        # Annotate timestep + legend
        cv2.putText(frame, f"t={int(f.stem)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "red=gripper  green=goal_gripper", (8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        writer.append_data(frame)

    writer.close()
    print(f"Wrote {args.out} ({len(npz_files)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
