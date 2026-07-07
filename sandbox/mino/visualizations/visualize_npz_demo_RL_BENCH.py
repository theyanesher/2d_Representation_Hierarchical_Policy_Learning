#!/usr/bin/env python3
"""Viser visualization of one demo directory of per-timestep .npz files produced
by the RL Bench -> D2 converter (rlbench_to_d2_converter.py).

Uses viser (browser-based) so it works on headless nodes — unlike the open3d
offscreen renderer in visualize_npz_demo.py, which needs a local GL context.

A timestep slider scrubs the demo. Per step it shows:
  - scene point cloud  (gray)   from `point_cloud`            (N,3)
  - current gripper    (red)    derived from `state` (pose+open) via RVT template
  - goal gripper       (green)  from `goal_gripper_pcd`        (4,3)
  - current EE pose frame (gold axes)

The RL Bench npz does NOT store a current `gripper_pcd`; we build it from
`state` = [x,y,z, qx,qy,qz,qw, gripper_open].

Usage:
    python visualize_npz_demo_RL_BENCH.py \
        --demo_dir .../RL_BENCH_DATASET/sweep_to_dustpan_of_size/demo_0
Then open the printed http://0.0.0.0:<port> URL in a browser.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf


# --- RVT gripper template (copied from the RL Bench converter) -------------
def quat_to_rotmat_np(q):
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def make_gripper_template_4_np():
    left_finger = np.array([0.0, -0.0405, 0.0800], dtype=np.float32)
    right_finger = np.array([0.0, 0.0405, 0.0800], dtype=np.float32)
    wrist = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    center_contact = 0.5 * (left_finger + right_finger)
    return np.stack([left_finger, right_finger, wrist, center_contact], axis=0)


def build_gripper_pcd_from_pose_np(pose_7, open_1, template_4):
    """pose_7 (N,7) [x,y,z,qx,qy,qz,qw], open_1 (N,1) in [0,1] -> (N,4,3) world."""
    t = pose_7[:, :3]
    R = quat_to_rotmat_np(pose_7[:, 3:7])
    p = np.repeat(template_4[None, :, :], repeats=pose_7.shape[0], axis=0).copy()
    delta = 0.021 * (np.clip(open_1[:, 0], 0.0, 1.0) - 0.5)
    p[:, 0, 1] -= delta
    p[:, 1, 1] += delta
    p[:, 3, :] = 0.5 * (p[:, 0, :] + p[:, 1, :])
    p_world = np.einsum("bij,bpj->bpi", R, p) + t[:, None, :]
    return p_world.astype(np.float32)


def _squeeze(x):
    x = np.asarray(x)
    return x[0] if x.ndim and x.shape[0] == 1 else x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", required=True, type=Path,
                   help="Directory containing N.npz files for one demo (e.g. .../demo_0).")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--pcd_subsample", type=int, default=1,
                   help="Keep every Nth scene point if rendering is slow.")
    p.add_argument("--goal_key", type=str, default="goal_gripper_pcd",
                   help="npz key for the goal gripper PCD (e.g. goal_gripper_pcd_rdp).")
    p.add_argument("--goal_dir", type=Path, default=None,
                   help="Read --goal_key from this mirror demo dir (e.g. the EXTRA_KEYPOINTS "
                        "demo dir) instead of --demo_dir. Obs still come from --demo_dir.")
    args = p.parse_args()

    npz_files = sorted(args.demo_dir.glob("*.npz"), key=lambda q: int(q.stem))
    if not npz_files:
        raise SystemExit(f"No .npz files in {args.demo_dir}")

    template_4 = make_gripper_template_4_np()

    # Preload the light-weight arrays (skip images) so the slider is responsive.
    scenes, states, goals, cur_grips, actions = [], [], [], [], []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        scenes.append(_squeeze(d["point_cloud"]).astype(np.float32))      # (N,3)
        s = _squeeze(d["state"]).astype(np.float32)                       # (8,)
        states.append(s)
        gd = np.load(args.goal_dir / f.name, allow_pickle=True) if args.goal_dir is not None else d
        goals.append(_squeeze(gd[args.goal_key]).astype(np.float32))  # (4,3)
        actions.append(_squeeze(d["action"]).astype(np.float32))          # (8,) abs pose+open
        cur_grips.append(build_gripper_pcd_from_pose_np(
            s[:7][None], np.array([[s[7]]], np.float32), template_4)[0])  # (4,3)
    T = len(npz_files)
    print(f"Loaded {T} timesteps from {args.demo_dir}")

    server = viser.ViserServer(port=args.port)
    t_slider = server.gui.add_slider("Timestep t", 0, T - 1, 1, 0)
    show_scene = server.gui.add_checkbox("Scene point cloud (gray)", True)
    show_cur = server.gui.add_checkbox("Current gripper (red)", True)
    show_goal = server.gui.add_checkbox("Goal gripper (green)", True)
    show_frame = server.gui.add_checkbox("Current EE pose frame", True)
    show_actions = server.gui.add_checkbox("Next 8 actions (coordinate frames)", True)
    N_ACTIONS = 8

    def _redraw(_=None):
        t = int(t_slider.value)

        if show_scene.value:
            pc = scenes[t]
            if args.pcd_subsample > 1:
                pc = pc[::args.pcd_subsample]
            server.scene.add_point_cloud("/scene", points=pc, colors=(180, 180, 180),
                                         point_size=0.004)
        else:
            try: server.scene.remove_by_name("/scene")
            except Exception: pass

        if show_cur.value:
            server.scene.add_point_cloud("/gripper_current", points=cur_grips[t],
                                         colors=(255, 0, 0), point_size=0.012)
        else:
            try: server.scene.remove_by_name("/gripper_current")
            except Exception: pass

        if show_goal.value:
            server.scene.add_point_cloud("/gripper_goal", points=goals[t],
                                         colors=(0, 255, 0), point_size=0.012)
        else:
            try: server.scene.remove_by_name("/gripper_goal")
            except Exception: pass

        if show_frame.value:
            s = states[t]
            R = quat_to_rotmat_np(s[3:7])
            server.scene.add_frame("/ee", position=s[:3].astype(np.float32),
                                   wxyz=vtf.SO3.from_matrix(R).wxyz,
                                   axes_length=0.06, axes_radius=0.0035)
        else:
            try: server.scene.remove_by_name("/ee")
            except Exception: pass

        # Next 8 actions: each action[k] is the absolute target pose at frame k+1.
        # Draw them as small coordinate frames (red/green/blue xyz axes).
        for i in range(N_ACTIONS):
            try: server.scene.remove_by_name(f"/actions/{i}")
            except Exception: pass
        if show_actions.value:
            for i in range(N_ACTIONS):
                k = t + i
                if k >= T:
                    break
                a = actions[k]
                Ra = quat_to_rotmat_np(a[3:7])
                server.scene.add_frame(f"/actions/{i}", position=a[:3].astype(np.float32),
                                       wxyz=vtf.SO3.from_matrix(Ra).wxyz,
                                       axes_length=0.025, axes_radius=0.0012)

    for w in (t_slider, show_scene, show_cur, show_goal, show_frame, show_actions):
        w.on_update(_redraw)
    _redraw()
    print(f"Viser server running on port {args.port}. Open the URL printed above.")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
