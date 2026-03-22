import h5py
import numpy as np
import viser
import viser.transforms as vtf
import time
import typer
from pathlib import Path
from common.data_utils import process_pointmap, process_depth, process_plucker
from mino_utils.vis_utils import unproject_depth_rgb
from manipulation.utils import rotation_transfer_6D_to_matrix

app = typer.Typer(help="Fixed Viser visualization for flattened H5 structure.")

CAM_IDS = [0, 1, 2]
CAM_CONFIG = {
    0: {"label": "Right", "color": (239, 68, 68)},   # Soft Red (Tailwind Red 500)
    1: {"label": "Wrist", "color": (34, 197, 94)},   # Soft Green (Tailwind Green 500)
    2: {"label": "Left",  "color": (59, 130, 246)},  # Soft Blue (Tailwind Blue 500)

}
def get_cam_data(f, t, cam_idx):
    return {
        "rgb": f[f"obs/cam{cam_idx}_image"][t],
        "depth": process_depth(f[f"obs/cam{cam_idx}_depth"][t], compress=False),
        "intrinsic": f[f"obs/cam{cam_idx}_intrinsic"][t],
        "extrinsic": f[f"obs/cam{cam_idx}_extrinsic"][t],
        "plucker": process_plucker(f[f"obs/cam{cam_idx}_plucker"][t], compress=False) if f"obs/cam{cam_idx}_plucker" in f else None,
        "pointmap": process_pointmap(f[f"obs/cam{cam_idx}_pointmap"][t], compress=False) if f"obs/cam{cam_idx}_pointmap" in f else None,
    }

def _transform_points(pts, T):
    """Transform Nx3 points by a 4x4 matrix T."""
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.concatenate([pts, ones], axis=1)  # Nx4
    return (T @ pts_h.T).T[:, :3]


def render_scene_elements(server, f, t, use_pointmap=False, show_plucker=False, show_image=False, stride=16, world_to_gripper=None):
    """Renders frustums and pointclouds for all cameras at timestep t.

    If world_to_gripper is provided (4x4 array), all world-frame data is
    transformed into the gripper frame before rendering.
    """
    for c in CAM_IDS:
        data = get_cam_data(f, t, c)
        c2w = np.linalg.inv(data["extrinsic"]) # For frustum, we need cam2world

        if world_to_gripper is not None:
            c2w = world_to_gripper @ c2w

        h, w = data["rgb"].shape[:2]
        # Calculate vertical FOV from the intrinsic matrix (fy is at [1,1])
        fov = 2 * np.arctan2(h / 2, data["intrinsic"][1, 1])
        frustum_scale = 0.05

        # 1. Add Frustum (Smaller scale + optional image texture)
        server.scene.add_camera_frustum(
            name=f"/frames/cam_{c}",
            fov=float(fov * 180 / np.pi),
            aspect=w / h,
            position=c2w[:3, 3],
            wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            scale=frustum_scale,
            color=CAM_CONFIG[c]["color"],
        )

        plane_height = 2.0 * 0.2 * np.tan(fov / 2.0)
        plane_width = plane_height * (w / h)

        server.scene.add_image(
            name=f"/frames/cam_{c}/image",
            image=data["rgb"],
            render_width=plane_width,
            render_height=plane_height,
            position=(0, 0, frustum_scale),  # Position slightly in front of frustum
        )

        # 2. Add Pointcloud
        if use_pointmap and data["pointmap"] is not None:
            points = data["pointmap"].transpose(1,2,0).reshape(-1, 3)
            colors = data["rgb"].reshape(-1, 3) / 255.0
        else:
            pcd_result = unproject_depth_rgb(
                data["depth"],
                data["intrinsic"],
                data["rgb"],
                data["extrinsic"],
                sample_ratio=0.75
            )

            if isinstance(pcd_result, tuple):
                points, colors = pcd_result
            else:
                points, colors = pcd_result.points, pcd_result.colors

        if world_to_gripper is not None:
            points = _transform_points(points, world_to_gripper)

        server.scene.add_point_cloud(
            f"/pcd/cam_{c}", 
            points=points, 
            colors=colors, 
            point_size=0.002
        )

        if show_plucker and data["plucker"] is not None:
            plucker_map = data["plucker"].transpose(1, 2, 0)
            # Subsample spatially
            plucker_sub = plucker_map[::stride, ::stride, :]
            
            # Flatten to (N, 6)
            vecs = plucker_sub.reshape(-1, 6)
            
            # Split into Direction (d) and Moment (m)
            d = vecs[:, :3]
            m = vecs[:, 3:]
            N, _ = d.shape
            # 1. Recover Ray Origins from Plucker coords: o = (d x m) / |d|^2
            # (If d is normalized, this is just d x m)
            origins = c2w[:3, 3][None, :].repeat(N, axis=0)
            
            # 2. Visualize Direction Vectors (The Ray) - RED
            # Drawn from computed origin 'o' to 'o + d'
            server.scene.add_line_segments(
                f"/plucker/cam_{c}/direction", 
                points=np.stack([origins, origins + d * 0.2], axis=1),
                colors=(255, 50, 50), 
                line_width=1.0
            )

            # 3. Visualize Moment Vectors - BLUE
            # Drawn from computed origin 'o' to 'o + m'
            # (Moments are perpendicular to the ray)
            server.scene.add_line_segments(
                f"/plucker/cam_{c}/moment", 
                points=np.stack([origins, origins + m * 0.2], axis=1),
                colors=(50, 50, 255), 
                line_width=1.0
            )

@app.command()
def explore(h5_path: str):
    server = viser.ViserServer()
    f = h5py.File(h5_path, 'r')
    num_frames = f['obs/cam0_image'].shape[0]

    t_slider = server.gui.add_slider("Timestep", 0, num_frames - 1, 1, 0)
    mode_toggle = server.gui.add_checkbox("Use Pointmap", initial_value=True)
    plucker_toggle = server.gui.add_checkbox("Show Plucker", initial_value=False)
    gripper_frame_toggle = server.gui.add_checkbox("Gripper Frame", initial_value=False)
    goal_pcd_toggle = server.gui.add_checkbox("Show Goal Gripper PCD", initial_value=True)

    def _upd(_):
        t = t_slider.value
        world_to_gripper = None
        if gripper_frame_toggle.value and 'obs/gripper_to_world' in f:
            gripper_to_world = f['obs/gripper_to_world'][t]
            world_to_gripper = np.linalg.inv(gripper_to_world)

        render_scene_elements(
            server, f, t,
            use_pointmap=mode_toggle.value,
            show_plucker=plucker_toggle.value,
            show_image=True,  # ENABLED for explore
            world_to_gripper=world_to_gripper,
        )
        if goal_pcd_toggle.value and 'obs/goal_gripper_pcd' in f:
            g = f['obs/goal_gripper_pcd'][t]  # (10,): pos(3) + 6D rot(6) + gripper(1)
            goal_pos = g[:3]
            goal_rot = rotation_transfer_6D_to_matrix(g[3:9])
            if world_to_gripper is not None:
                goal_pos = _transform_points(goal_pos[None], world_to_gripper)[0]
                goal_rot = world_to_gripper[:3, :3] @ goal_rot
            server.scene.add_frame(
                "/goal_gripper_pcd",
                position=goal_pos,
                wxyz=vtf.SO3.from_matrix(goal_rot).wxyz,
                axes_length=0.05,
                axes_radius=0.003,
            )

        if gripper_frame_toggle.value and 'action/delta' in f:
            # action/delta: Δxyz and ΔR are both in the EE/gripper frame.
            # Integrate in the fixed gripper frame at t_viz:
            #   p_{i+1} = p_i + R_current @ Δp_ee
            #   R_{i+1} = R_current @ ΔR
            state_t = f['obs/state'][t][:]
            R_current = world_to_gripper[:3,:3] @ rotation_transfer_6D_to_matrix(state_t[3:9])
            assert np.allclose(R_current, np.eye(3), atol=1e-6), "After transforming to gripper frame, rotation should be identity"
            curr_pos = _transform_points(state_t[:3][None], world_to_gripper)[0]
            assert np.allclose(curr_pos, np.array([0.,0.,0.]), atol=1e-7), "After transforming to gripper frame, position should be at origin"
            for i in range(min(15, num_frames - t)):
                act = f['action/delta'][t + i]
                curr_pos = curr_pos + R_current @ act[:3]
                R_current = R_current @ rotation_transfer_6D_to_matrix(act[3:9])
                server.scene.add_frame(
                    f"/traj/a_{i}",
                    position=curr_pos,
                    wxyz=vtf.SO3.from_matrix(R_current).wxyz,
                    axes_length=0.02,
                    axes_radius=0.001,
                )
        elif not gripper_frame_toggle.value and 'action/hybrid' in f:
            # action/hybrid: Δxyz in world frame, ΔR in EE/gripper frame (right-multiply).
            #   p_{i+1} = p_i + Δp_world
            #   R_{i+1} = R_current @ ΔR
            state_t = f['obs/state'][t][:]
            R_current = rotation_transfer_6D_to_matrix(state_t[3:9])
            curr_pos = state_t[:3].copy()
            for i in range(min(15, num_frames - t)):
                act = f['action/hybrid'][t + i]
                curr_pos = curr_pos + act[:3]
                R_current = R_current @ rotation_transfer_6D_to_matrix(act[3:9])
                server.scene.add_frame(
                    f"/traj/a_{i}",
                    position=curr_pos,
                    wxyz=vtf.SO3.from_matrix(R_current).wxyz,
                    axes_length=0.02,
                    axes_radius=0.001,
                )

    t_slider.on_update(_upd)
    mode_toggle.on_update(_upd)
    plucker_toggle.on_update(_upd)
    gripper_frame_toggle.on_update(_upd)
    goal_pcd_toggle.on_update(_upd)
    
    _upd(None)
    while True:
        time.sleep(1)

@app.command()
def multi(data_dir: str, max_episodes: int = 50):
    server = viser.ViserServer()
    paths = sorted(list(Path(data_dir).glob("*.h5")))[:max_episodes]
    mode_toggle = server.gui.add_checkbox("Use Pointmap", initial_value=True)

    for i, p in enumerate(paths):
        with h5py.File(p, 'r') as f:
            for c in CAM_IDS:
                c2w = np.linalg.inv(f[f"obs/cam{c}_extrinsic"][0])
                server.scene.add_camera_frustum(
                    f"/static/d_{i}/c_{c}", 
                    fov=0.3, aspect=1.0, 
                    position=c2w[:3,3], 
                    wxyz=vtf.SO3.from_matrix(c2w[:3,:3]).wxyz,
                    scale=0.01, 
                    color=CAM_CONFIG[c]["color"]
                )

    demo_slider = server.gui.add_slider("Active Demo", 0, len(paths)-1, 1, 0)
    
    @demo_slider.on_update
    @mode_toggle.on_update
    def _upd(_):
        with h5py.File(paths[demo_slider.value], 'r') as f:
            render_scene_elements(
                server, f, 0, 
                use_pointmap=mode_toggle.value,
                show_image=False # DISABLED for multi to keep it fast
            )

    _upd(None)
    while True:
        time.sleep(1)

if __name__ == "__main__":
    app()