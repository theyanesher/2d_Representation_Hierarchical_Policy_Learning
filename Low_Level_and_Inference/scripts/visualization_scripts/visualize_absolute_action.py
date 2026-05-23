"""
Viser visualization of the *absolute* action target derived from action/hybrid.

Convention: absolute[t] = state[t] ⊕ action/hybrid[t]
  (world-frame xyz + body-frame rotation composition + additive gripper)

Frame note
----------
`obs/state`, `obs/point_cloud`, `obs/present_gripper_pts`, `obs/goal_gripper_pts`
and the actions all live in the **robot-base frame**. The raw `obs/camN_*`
intrinsics/extrinsics describe cameras in the **world frame**, so unprojecting
depth into world frame puts the scene at a different origin than the actions.
We use `obs/point_cloud` directly to stay in the right frame.

Usage
-----
  pixi run python scripts/visualization_scripts/visualize_absolute_action.py \
      explore <path/to/demo_0.h5>

  pixi run python scripts/visualization_scripts/visualize_absolute_action.py \
      verify <path/to/demo_0.h5>
"""
import time

import h5py
import numpy as np
import typer
import viser
import viser.transforms as vtf

from manipulation.utils import rotation_transfer_6D_to_matrix


app = typer.Typer(help="Visualize state[t] + action/hybrid[t] absolute targets.")


def _state_to_pose(state_vec: np.ndarray):
    xyz = state_vec[:3].astype(np.float32)
    R = rotation_transfer_6D_to_matrix(state_vec[3:9].astype(np.float32))
    grip = float(state_vec[9])
    return xyz, R, grip


def _compose_absolute(state_vec, action_vec):
    """absolute = state composed with action/hybrid:
       p_abs = p_state + world Δp
       R_abs = R_state @ R_delta
       g_abs = g_state + Δg
    """
    p_s, R_s, g_s = _state_to_pose(state_vec)
    delta_R = rotation_transfer_6D_to_matrix(action_vec[3:9].astype(np.float32))
    p_abs = p_s + action_vec[:3]
    R_abs = R_s @ delta_R
    g_abs = g_s + float(action_vec[9])
    return p_abs, R_abs, g_abs


def _add_pose_frame(server, name, xyz, R, axes_length=0.04, axes_radius=0.002):
    server.scene.add_frame(
        name,
        position=xyz,
        wxyz=vtf.SO3.from_matrix(R).wxyz,
        axes_length=axes_length,
        axes_radius=axes_radius,
    )


def _clear(server, prefix, max_n):
    for i in range(max_n):
        try:
            server.scene.remove_by_name(f"{prefix}/{i}")
        except Exception:
            pass


@app.command()
def explore(h5_path: str, max_horizon: int = 16, pcd_subsample: int = 1):
    """Interactive scrub. `pcd_subsample` keeps every Nth point if rendering is slow."""
    server = viser.ViserServer()
    f = h5py.File(h5_path, "r")
    T = f["obs/state"].shape[0]
    print(f"Loaded {h5_path} — {T} timesteps")

    has_pcd = "obs/point_cloud" in f
    has_gripper_pts = "obs/present_gripper_pts" in f
    has_goal_pts = "obs/goal_gripper_pts" in f
    print(f"  point_cloud: {has_pcd}, present_gripper_pts: {has_gripper_pts}, "
          f"goal_gripper_pts: {has_goal_pts}")

    states = f["obs/state"][:]            # (T, 10)
    actions = f["action/hybrid"][:]       # (T, 10)

    t_slider = server.gui.add_slider("Timestep t", 0, T - 1, 1, 0)
    horizon_slider = server.gui.add_slider("Horizon", 1, max_horizon, 1, 8)
    show_absolute = server.gui.add_checkbox(
        "Commanded absolute targets = state[t] + action[t] (green)", True
    )
    show_observed_next = server.gui.add_checkbox(
        "Observed state[t+1..t+H] (blue)", True
    )
    show_pcd = server.gui.add_checkbox("Scene point cloud (obs/point_cloud)", True)
    show_gripper_pts = server.gui.add_checkbox(
        "Current gripper keypoints (obs/present_gripper_pts)", True
    )
    show_goal_pts = server.gui.add_checkbox(
        "Goal gripper keypoints (obs/goal_gripper_pts)", False
    )
    show_line = server.gui.add_checkbox("Connect target poses with lines", True)

    def _redraw(_=None):
        t = int(t_slider.value)
        H = max(1, min(int(horizon_slider.value), T - 1 - t))

        # Current EE pose (state[t]) — large gold-axis frame.
        cur_xyz, cur_R, _ = _state_to_pose(states[t])
        _add_pose_frame(server, "/current", cur_xyz, cur_R,
                        axes_length=0.06, axes_radius=0.0035)

        # Scene point cloud (in robot-base frame — same as state).
        if show_pcd.value and has_pcd:
            pcd = f["obs/point_cloud"][t]  # (N, 3)
            if pcd_subsample > 1:
                pcd = pcd[::pcd_subsample]
            server.scene.add_point_cloud(
                "/pcd/scene",
                points=pcd.astype(np.float32),
                colors=(180, 180, 180),
                point_size=0.003,
            )
        else:
            try: server.scene.remove_by_name("/pcd/scene")
            except Exception: pass

        # Current gripper keypoints — sanity check that state[t] sits with them.
        if show_gripper_pts.value and has_gripper_pts:
            pts = f["obs/present_gripper_pts"][t]  # (4, 3)
            server.scene.add_point_cloud(
                "/gripper_pts/current",
                points=pts.astype(np.float32),
                colors=(255, 200, 0),
                point_size=0.012,
            )
        else:
            try: server.scene.remove_by_name("/gripper_pts/current")
            except Exception: pass

        # Goal gripper keypoints — where the high-level model says to go.
        if show_goal_pts.value and has_goal_pts:
            pts = f["obs/goal_gripper_pts"][t]
            server.scene.add_point_cloud(
                "/gripper_pts/goal",
                points=pts.astype(np.float32),
                colors=(255, 80, 200),
                point_size=0.012,
            )
        else:
            try: server.scene.remove_by_name("/gripper_pts/goal")
            except Exception: pass

        # Commanded absolute targets (green): state[t+i] ⊕ action[t+i]
        _clear(server, "/absolute", max_horizon)
        abs_pts = [cur_xyz]
        if show_absolute.value:
            for i in range(H):
                p_abs, R_abs, _ = _compose_absolute(states[t + i], actions[t + i])
                _add_pose_frame(server, f"/absolute/{i}", p_abs, R_abs,
                                axes_length=0.025, axes_radius=0.0012)
                abs_pts.append(p_abs)
            if show_line.value and len(abs_pts) > 1:
                pts = np.asarray(abs_pts)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                server.scene.add_line_segments(
                    "/absolute_line", points=segs, colors=(60, 220, 60), line_width=3.0,
                )
            else:
                try: server.scene.remove_by_name("/absolute_line")
                except Exception: pass
        else:
            try: server.scene.remove_by_name("/absolute_line")
            except Exception: pass

        # Observed next states (blue): state[t+1..t+H]
        _clear(server, "/observed", max_horizon)
        if show_observed_next.value:
            obs_pts = [cur_xyz]
            for i in range(1, H + 1):
                p_obs, R_obs, _ = _state_to_pose(states[t + i])
                _add_pose_frame(server, f"/observed/{i - 1}", p_obs, R_obs,
                                axes_length=0.02, axes_radius=0.001)
                obs_pts.append(p_obs)
            if show_line.value and len(obs_pts) > 1:
                pts = np.asarray(obs_pts)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                server.scene.add_line_segments(
                    "/observed_line", points=segs, colors=(60, 130, 230), line_width=2.0,
                )
            else:
                try: server.scene.remove_by_name("/observed_line")
                except Exception: pass
        else:
            try: server.scene.remove_by_name("/observed_line")
            except Exception: pass

    for w in (t_slider, horizon_slider, show_absolute, show_observed_next,
              show_pcd, show_gripper_pts, show_goal_pts, show_line):
        w.on_update(_redraw)
    _redraw()
    print("Viser server running. Open the URL printed above.")
    while True:
        time.sleep(1)


@app.command()
def verify(h5_path: str):
    """Round-trip sanity: (state + action) decomposed back recovers action exactly,
    plus the controller under-tracking gap |absolute[t] - state[t+1]|."""
    f = h5py.File(h5_path, "r")
    states = f["obs/state"][:]
    actions = f["action/hybrid"][:]
    T = states.shape[0]

    max_xyz = 0.0
    max_R = 0.0
    max_g = 0.0
    for t in range(T - 1):
        p_abs, R_abs, g_abs = _compose_absolute(states[t], actions[t])
        p_s, R_s, g_s = _state_to_pose(states[t])
        dp = p_abs - p_s
        dR = R_s.T @ R_abs
        dg = g_abs - g_s
        gt_dR = rotation_transfer_6D_to_matrix(actions[t, 3:9].astype(np.float32))
        max_xyz = max(max_xyz, float(np.linalg.norm(dp - actions[t, :3])))
        max_R = max(max_R, float(np.linalg.norm(dR - gt_dR)))
        max_g = max(max_g, float(abs(dg - actions[t, 9])))

    print(f"Round-trip (state + action) decomposed back errors (should be ~1e-6):")
    print(f"  Δxyz error : {max_xyz:.3e}")
    print(f"  ΔR error   : {max_R:.3e}")
    print(f"  Δgrip error: {max_g:.3e}")

    gap_p = []
    for t in range(T - 1):
        p_abs, _, _ = _compose_absolute(states[t], actions[t])
        gap_p.append(np.linalg.norm(p_abs - states[t + 1, :3]))
    gap_p = np.asarray(gap_p)
    print(f"\nGap |absolute[t] - state[t+1]| (OSC controller under-tracking):")
    print(f"  max  = {gap_p.max():.4f} m")
    print(f"  mean = {gap_p.mean():.4f} m")


if __name__ == "__main__":
    app()
