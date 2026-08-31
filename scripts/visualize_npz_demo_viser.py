"""Interactive viser viewer for demo trajectories + ground-truth subgoals.

Standalone counterpart to visualize_npz_demo_matplotlib.py: same data
(gripper path + any goal_gripper_pcd_<type> keys from a --goal_root tree),
but as a live, scrubbable 3D scene in the browser instead of static PNGs.
Unlike the viser viewer baked into run_gmm_on_dataset_batch_optimized.py, this
one needs NO trained checkpoint / model inference -- it only reads what's
already on disk, so it's the right tool for eyeballing raw AWE/RDP/etc.
subgoals right after generation.

Data layout (per-frame npz convention, see NpyDataset):
    demo_root/demo_N/t.npz : gripper_pcd (1, 4, 3), point_cloud (1, P, 3)
    goal_root/demo_N/t.npz : goal_gripper_pcd_<type> (1, 4, 3)
(goal_root may equal demo_root if goals live inline, e.g. the default source.)

GUI:
    - Demo dropdown: switch between demo_*/ without restarting the server.
    - Timestep slider: scrub frames within the selected demo.
    - Scene / gripper / all-4-points checkboxes: toggle the scene point
      cloud context (re-read + recolorized at the current timestep on every
      slider move, not frozen at frame 0), the current gripper marker (the
      sparse 4-keypoint gripper_pcd, drawn in green, always as all 4 points
      plus its centroid -- the "current waypoint" -- in a darker green, both
      NOT part of the dense scene point cloud, which is object/table-only,
      and not affected by the all-4-points toggle below), and whether GOAL
      markers are drawn as their centroid or all 4 keypoints.
    - Per goal type (auto-detected, one color each, consistent with the
      matplotlib script's palette):
        - ALL unique goals along the path, drawn faint/small (static).
        - The goal ACTIVE at the current timestep, drawn large/bright --
          watch it jump to the next dot exactly as you scrub past a
          transition frame.

Example:
    pixi run python scripts/visualize_npz_demo_viser.py \\
        --demo_root /data/theya/COFFEE_PREPERATION_D1_sample \\
        --goal_root /data/theya/COFFEE_PREPERATION_D1_sample_AWE --scene
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_npz_demo_matplotlib import (  # noqa: E402
    detect_goal_keys,
    key_to_folder,
    load_demo,
    sorted_frame_files,
    unique_consecutive_goals,
)

# RGB triplets, same order/intent as TYPE_COLORS in visualize_npz_demo_matplotlib.py.
VISER_TYPE_COLORS = [
    (220, 40, 40),  # red
    (60, 100, 220),  # royalblue
    (230, 150, 30),  # orange
    (200, 40, 200),  # magenta
    (60, 200, 90),  # limegreen
    (40, 200, 200),  # cyan
]
# Emoji swatches for the GUI legend, same order as VISER_TYPE_COLORS -- kept
# as plain markdown (no raw HTML/CSS) since viser's markdown widget doesn't
# render inline <span style="..."> color spans.
VISER_TYPE_SWATCHES = ["\U0001f534", "\U0001f537", "\U0001f7e0", "\U0001f7e3", "\U0001f7e2", "\U0001f7e6"]
VISER_TYPE_NAMES = ["red", "royalblue", "orange", "magenta", "limegreen", "cyan"]

GRAY = (170, 170, 170)
GRIPPER_RED = (0, 255, 0)
WAYPOINT_DARK_GREEN = (0, 100, 0)  # current-gripper centroid, distinct from the 4 keypoints' green
ACTIVE_WHITE_MIX = 0.55  # blend toward white for the "active goal" highlight
_MIN_VALID_DEPTH = 1e-6  # points at/behind the camera plane are invalid


def brighten(color, mix=ACTIVE_WHITE_MIX):
    c = np.array(color, dtype=np.float32)
    out = c * (1 - mix) + 255 * mix
    return tuple(out.astype(np.uint8).tolist())


def load_frame_scene(demo_dir, frame_fname, is_rl_bench):
    """Read one frame's point_cloud + primary-camera RGB/intrinsics/extrinsics,
    needed to colorize that frame's (already-fused, possibly multi-camera)
    `point_cloud` -- it has no per-point color of its own in the npz. Mirrors
    NpyDataset's schema switch (MimicGen agentview_* channel-last vs RL Bench
    front_* channel-first) in src/lfd3d/datasets/npy/npy_dataset.py."""
    d = np.load(os.path.join(demo_dir, frame_fname), allow_pickle=True)
    points = d["point_cloud"][0].astype(np.float32)
    if is_rl_bench:
        rgb = d["front_rgb"][0].transpose(1, 2, 0).astype(np.uint8)
        intrinsics = d["front_camera_intrinsics"][0].astype(np.float32)
        extrinsics = d["front_camera_extrinsics"][0].astype(np.float32)
    else:
        rgb = d["rgb_agentview"][0].astype(np.uint8)
        intrinsics = d["agentview_intrinsics"][0].astype(np.float32)
        extrinsics = d["agentview_extrinsics"][0].astype(np.float32)
    return points, rgb, intrinsics, extrinsics


def colorize_points(points_world, rgb, intrinsics, extrinsics):
    """Per-point RGB by projecting world-frame points into a camera image
    (extrinsics = world_from_cam, as stored in the npz). Points that land
    behind the camera or outside the image fall back to gray."""
    height, width = rgb.shape[:2]
    cam_from_world = np.linalg.inv(extrinsics)
    pts_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=points_world.dtype)],
        axis=1,
    )
    pts_cam = (cam_from_world @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = intrinsics @ pts_cam.T
        u = np.round(proj[0] / proj[2]).astype(int)
        v = np.round(proj[1] / proj[2]).astype(int)
    valid = (z > _MIN_VALID_DEPTH) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    colors = np.tile(np.array(GRAY, dtype=np.uint8), (points_world.shape[0], 1))
    colors[valid] = rgb[v[valid], u[valid]]
    return colors


def main():  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--demo_root",
        required=True,
        help="Dataset root containing demo_*/ of per-frame npz "
        "(needs gripper_pcd, point_cloud).",
    )
    parser.add_argument(
        "--goal_root",
        required=True,
        help="Root containing matching demo_*/ npz with the "
        "goal_gripper_pcd_<type> key(s). Can equal --demo_root.",
    )
    parser.add_argument(
        "--goal_keys",
        default=None,
        help="Comma-separated goal keys; default: auto-detect all "
        "'goal_gripper_pcd*' keys in the first demo's goal npz.",
    )
    parser.add_argument(
        "--demos",
        default=None,
        help="Comma-separated demo names (e.g. demo_0,demo_3); "
        "default: all demo_*/ present under both roots.",
    )
    parser.add_argument(
        "--scene",
        action="store_true",
        default=False,
        help="Start with the frame-0 scene point cloud shown.",
    )
    parser.add_argument(
        "--is_rl_bench",
        action="store_true",
        default=False,
        help="Use the RL Bench camera schema (front_rgb, channel-first) "
        "instead of the MimicGen schema (rgb_agentview, channel-last) "
        "when colorizing the scene point cloud.",
    )
    parser.add_argument(
        "--all_goal_points",
        action="store_true",
        default=False,
        help="Start showing all 4 gripper keypoints per goal "
        "instead of just the centroid.",
    )
    parser.add_argument("--path_point_size", type=float, default=0.001)
    parser.add_argument("--goal_point_size", type=float, default=0.003)
    parser.add_argument("--active_point_size", type=float, default=0.005)
    parser.add_argument("--scene_point_size", type=float, default=0.003)
    parser.add_argument("--gripper_point_size", type=float, default=0.002)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    import viser  # noqa: PLC0415 -- lazy: viser is an optional, heavy dependency

    if args.demos:
        demo_names = args.demos.split(",")
    else:
        demo_names = sorted(
            [
                e
                for e in os.listdir(args.demo_root)
                if e.startswith("demo_")
                and os.path.isdir(os.path.join(args.demo_root, e))
                and os.path.isdir(os.path.join(args.goal_root, e))
            ],
            key=lambda x: int(x.split("_")[1]),
        )
    if not demo_names:
        raise FileNotFoundError(
            f"No demo_*/ dirs found under both {args.demo_root} and {args.goal_root}"
        )

    first_goal_dir = os.path.join(args.goal_root, demo_names[0])
    frames0 = sorted_frame_files(first_goal_dir)
    goal_keys = (
        args.goal_keys.split(",")
        if args.goal_keys
        else detect_goal_keys(first_goal_dir, frames0)
    )
    if not goal_keys:
        raise KeyError(f"No 'goal_gripper_pcd*' keys found in {first_goal_dir}")

    folders = {k: key_to_folder(k) for k in goal_keys}
    colors = {
        k: VISER_TYPE_COLORS[i % len(VISER_TYPE_COLORS)]
        for i, k in enumerate(goal_keys)
    }
    print(f"Demos: {len(demo_names)}. Goal types: {', '.join(folders.values())}")
    print("Color legend (goal type -> RGB):")
    for k in goal_keys:
        print(f"  {folders[k]:<20s} {colors[k]}")

    server = viser.ViserServer(port=args.port)
    print(f"[viser] Open http://localhost:{args.port}")

    demo_dropdown = server.gui.add_dropdown(
        "Demo", demo_names, initial_value=demo_names[0]
    )
    timestep_slider = server.gui.add_slider(
        "Timestep", min=0, max=1, step=1, initial_value=0
    )
    scene_checkbox = server.gui.add_checkbox("Show scene", args.scene)
    gripper_checkbox = server.gui.add_checkbox("Show gripper", True)
    all_points_checkbox = server.gui.add_checkbox(
        "All 4 keypoints", args.all_goal_points
    )
    # Static legend mapping each goal type to its swatch color, so it's
    # obvious which dot color corresponds to which subgoal type in the
    # scene (colors are fixed per-run, assigned above from VISER_TYPE_COLORS).
    # Plain markdown only -- viser's markdown widget doesn't render raw HTML.
    legend_lines = ["**Legend**"]
    for i, k in enumerate(goal_keys):
        swatch = VISER_TYPE_SWATCHES[i % len(VISER_TYPE_SWATCHES)]
        name = VISER_TYPE_NAMES[i % len(VISER_TYPE_NAMES)]
        legend_lines.append(f"{swatch} {folders[k]} ({name})")
    server.gui.add_markdown("\n\n".join(legend_lines))
    info_md = server.gui.add_markdown("")

    state = {}  # populated by load_current_demo()
    # viser dispatches GUI callbacks on a worker thread pool, so they can fire
    # while `state` is still empty -- a browser that connects during startup
    # replays the GUI state, echoing back checkbox/slider updates before the
    # first refresh_all() has run -- or mid-swap during a demo change. The lock
    # serializes load/render; `state` being empty is the "not ready yet" flag.
    # RLock, not Lock: assigning to a GUI element inside a callback can re-enter
    # on the same thread, which would self-deadlock a plain Lock.
    render_lock = threading.RLock()

    def load_current_demo():
        demo = demo_dropdown.value
        demo_dir = os.path.join(args.demo_root, demo)
        goal_dir = os.path.join(args.goal_root, demo)
        # load_scene=False: point_cloud is now re-read per timestep (see
        # update_scene below) instead of once at frame 0, so it stays in sync
        # as the scene changes (objects moving, arm occlusion, etc.).
        gripper, goals, _ = load_demo(demo_dir, goal_dir, goal_keys, load_scene=False)
        path = gripper.mean(axis=1)  # (T, 3)
        uniq = {k: unique_consecutive_goals(goals[k]) for k in goal_keys}

        state.update(
            {
                "demo": demo,
                "demo_dir": demo_dir,
                "frames": sorted_frame_files(demo_dir),
                "gripper": gripper,
                "goals": goals,
                "path": path,
                "uniq": uniq,
            }
        )
        # Clamp the slider BEFORE resizing its range: assigning .max echoes an
        # update back from the client, and a stale value past the new demo's
        # last frame would index off the end of the freshly loaded arrays.
        if timestep_slider.value > len(path) - 1:
            timestep_slider.value = 0
        timestep_slider.max = len(path) - 1

    def update_scene(t):
        """Re-read this frame's own point_cloud + camera image and recolorize --
        called on every slider move (not just once at load), so the scene
        tracks the current timestep instead of staying frozen at frame 0."""
        if not scene_checkbox.value:
            if "scene_handle" in state:
                state["scene_handle"].visible = False
            return
        points, rgb, intrinsics, extrinsics = load_frame_scene(
            state["demo_dir"], state["frames"][t], args.is_rl_bench
        )
        scene_colors = colorize_points(points, rgb, intrinsics, extrinsics)
        state["scene_handle"] = server.scene.add_point_cloud(
            "scene",
            points=points,
            colors=scene_colors,
            point_size=args.scene_point_size,
            visible=True,
        )

    def render_static():
        cmap_t = np.linspace(0, 1, len(state["path"]))
        # Simple viridis-ish gradient without a matplotlib dependency at render time.
        path_colors = (
            np.stack(
                [
                    (68 + cmap_t * (253 - 68)),
                    (1 + cmap_t * (231 - 1)),
                    (84 + cmap_t * (37 - 84)),
                ],
                axis=1,
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
        server.scene.add_point_cloud(
            "path",
            points=state["path"],
            colors=path_colors,
            point_size=args.path_point_size,
        )

        for k in goal_keys:
            uniq_goals, _first_frames = state["uniq"][k]
            pts = (
                uniq_goals
                if all_points_checkbox.value
                else uniq_goals.mean(axis=1, keepdims=True)
            )
            pts = pts.reshape(-1, 3)
            n = pts.shape[0]
            server.scene.add_point_cloud(
                f"goals/{folders[k]}",
                points=pts,
                colors=np.tile(colors[k], (n, 1)).astype(np.uint8),
                point_size=args.goal_point_size,
            )

    def render_dynamic(t):
        # A slider echo queued before a demo swap can outrun the new load.
        t = min(t, len(state["path"]) - 1)
        update_scene(t)

        if not gripper_checkbox.value:
            if "gripper_handle" in state:
                state["gripper_handle"].visible = False
            if "waypoint_handle" in state:
                state["waypoint_handle"].visible = False
        else:
            gripper_t = state["gripper"][t]  # (4, 3) -- always all 4 points,
            # independent of the "All 4 keypoints" checkbox (that one only
            # governs the goal markers).
            state["gripper_handle"] = server.scene.add_point_cloud(
                "current_gripper",
                points=gripper_t.reshape(-1, 3),
                colors=np.tile(GRIPPER_RED, (gripper_t.shape[0], 1)).astype(np.uint8),
                point_size=args.gripper_point_size,
                visible=True,
            )
            # Current waypoint: the gripper's own centroid, shown alongside
            # its 4 keypoints in a darker green so the two are distinguishable.
            waypoint = gripper_t.mean(axis=0, keepdims=True)
            state["waypoint_handle"] = server.scene.add_point_cloud(
                "current_waypoint",
                points=waypoint,
                colors=np.tile(WAYPOINT_DARK_GREEN, (waypoint.shape[0], 1)).astype(np.uint8),
                point_size=args.active_point_size,
                visible=True,
            )

        lines = [f"**{state['demo']}** — frame {t}/{len(state['path']) - 1}"]
        for k in goal_keys:
            active = state["goals"][k][t]  # (4, 3), the goal in force at frame t
            pts = (
                active
                if all_points_checkbox.value
                else active.mean(axis=0, keepdims=True)
            )
            server.scene.add_point_cloud(
                f"active/{folders[k]}",
                points=pts.reshape(-1, 3),
                colors=np.tile(brighten(colors[k]), (pts.shape[0], 1)).astype(np.uint8),
                point_size=args.active_point_size,
            )
            n_uniq = state["uniq"][k][0].shape[0]
            lines.append(f"- {folders[k]}: {n_uniq} unique goals along this demo")
        info_md.content = "\n".join(lines)

    def refresh_all():
        load_current_demo()
        render_static()
        render_dynamic(timestep_slider.value)

    def register_callbacks():
        """Wire up the GUI handlers. Called only AFTER the first refresh_all(),
        so a client connecting during startup can't drive a render against an
        unpopulated `state`. The `if not state` guards keep that safe even if a
        future edit moves this call back before the initial load."""

        @demo_dropdown.on_update
        def _(_):
            with render_lock:
                refresh_all()

        @timestep_slider.on_update
        def _(_):
            with render_lock:
                if not state:
                    return
                render_dynamic(timestep_slider.value)

        @scene_checkbox.on_update
        def _(_):
            with render_lock:
                if not state:
                    return
                render_static()
                render_dynamic(timestep_slider.value)

        @gripper_checkbox.on_update
        def _(_):
            with render_lock:
                if not state:
                    return
                render_dynamic(timestep_slider.value)

        @all_points_checkbox.on_update
        def _(_):
            with render_lock:
                if not state:
                    return
                render_static()
                render_dynamic(timestep_slider.value)

    with render_lock:
        refresh_all()
    register_callbacks()
    print("[viser] Use the Demo dropdown / Timestep slider. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[viser] Exiting.")
        server.stop()


if __name__ == "__main__":
    main()
