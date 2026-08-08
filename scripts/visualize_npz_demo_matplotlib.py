"""Static 3D matplotlib views of demos: gripper path as a line, goal gripper
keypoints as sphere dots along it — one plot per keypoint type per demo, plus
a combined overlay of all types.

Sweeps every demo_*/ under --demo_root, auto-detects the goal keys present in
the matching --goal_root npz files (any key starting with 'goal_gripper_pcd'),
and writes:

    <out_dir>/
        rdp/                 demo_0.png, demo_1.png, ...   (goal_gripper_pcd_rdp)
        rdp_gripper/         demo_0.png, ...
        <other type>/        ...
        combined/            demo_0.png, ...   all keypoint types overlaid

Data layout (per-frame npz convention, see NpyDataset):
    demo_root/demo_N/t.npz : gripper_pcd   (1, 4, 3), point_cloud (1, P, 3)
    goal_root/demo_N/t.npz : goal_gripper_pcd_<type>  (1, 4, 3)

The path is the mean of the 4 gripper keypoints per frame. Per-frame goals
repeat until the demo passes each keypoint, so consecutive duplicates are
collapsed and only the unique goals are drawn.

Example:
  pixi run python scripts/visualize_npz_demo_matplotlib.py \\
      --demo_root .../D2/COFFEE_PREPERATION_D1 \\
      --goal_root .../D2/EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1 \\
      --out_dir   .../D2/KEYPOINT_PLOTS/COFFEE_PREPERATION_D1 --scene
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GOAL_KEY_PREFIX = "goal_gripper_pcd"


def key_to_folder(key):
    """goal_gripper_pcd_rdp -> rdp; bare goal_gripper_pcd -> default."""
    rest = key[len(GOAL_KEY_PREFIX):].lstrip("_")
    return rest or "default"


def sorted_frame_files(d):
    return sorted(
        [f for f in os.listdir(d) if f.endswith(".npz")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )


def detect_goal_keys(goal_demo_dir, frames):
    with np.load(os.path.join(goal_demo_dir, frames[0]), allow_pickle=True) as d:
        return sorted(k for k in d.keys() if k.startswith(GOAL_KEY_PREFIX))


def load_demo(demo_dir, goal_dir, goal_keys, load_scene):
    frames = sorted_frame_files(demo_dir)
    T = len(frames)
    gripper = np.zeros((T, 4, 3), dtype=np.float32)
    goals = {k: np.zeros((T, 4, 3), dtype=np.float32) for k in goal_keys}
    scene = None

    for t, fname in enumerate(frames):
        with np.load(os.path.join(demo_dir, fname), allow_pickle=True) as data:
            gripper[t] = data["gripper_pcd"][0]
            if t == 0 and load_scene:
                scene = data["point_cloud"][0]
        with np.load(os.path.join(goal_dir, fname), allow_pickle=True) as gdata:
            for k in goal_keys:
                goals[k][t] = gdata[k][0]

    return gripper, goals, scene


def unique_consecutive_goals(goals, tol=1e-5):
    """Collapse the per-frame goal stream (T, 4, 3) to its K unique goals,
    returning (K, 4, 3) plus the frame index where each goal first appears."""
    keep = [0]
    for t in range(1, len(goals)):
        if not np.allclose(goals[t], goals[keep[-1]], atol=tol):
            keep.append(t)
    return goals[keep], np.array(keep)


def set_equal_aspect(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 * 1.05
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


# One fixed color per keypoint type so the combined plot and the per-type
# plots stay visually consistent across demos.
TYPE_COLORS = ["red", "royalblue", "orange", "magenta", "limegreen", "cyan"]


def plot_demo(out_path, title, path, goal_sets, scene, args):
    """goal_sets: list of (label, color, uniq_goals (K,4,3), first_frames)."""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    if scene is not None:
        ax.scatter(*scene.T, s=0.3, c="lightgray", alpha=0.25,
                   depthshade=False, zorder=1)

    # Path line, shaded by time so direction of travel is readable.
    cmap = plt.get_cmap("viridis")
    for t in range(len(path) - 1):
        ax.plot(*path[t:t + 2].T, color=cmap(t / max(len(path) - 2, 1)),
                linewidth=2.0, zorder=3)
    ax.scatter(*path[0], s=60, c="green", marker="^", zorder=5, label="start")
    ax.scatter(*path[-1], s=60, c="black", marker="s", zorder=5, label="end")

    extent = [path]
    for label, color, uniq_goals, first_frames in goal_sets:
        pts = (uniq_goals if args.all_goal_points
               else uniq_goals.mean(axis=1))  # (K,4,3) or (K,3) centers
        pts = pts.reshape(-1, 3)
        ax.scatter(*pts.T, s=args.dot_size, color=color, edgecolors="black",
                   linewidths=0.5, depthshade=False, zorder=6,
                   label=f"{label} ({len(uniq_goals)} goals)")
        extent.append(pts)

    if scene is not None:
        extent.append(scene)
    set_equal_aspect(ax, np.concatenate(extent, axis=0))

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=args.elev, azim=args.azim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo_root", required=True,
                        help="dataset root containing demo_*/ of per-frame npz")
    parser.add_argument("--goal_root", required=True,
                        help="matching EXTRA_KEYPOINTS root containing demo_*/")
    parser.add_argument("--out_dir", required=True,
                        help="output root; one subfolder per keypoint type + combined/")
    parser.add_argument("--goal_keys", default=None,
                        help="comma-separated goal keys; default: auto-detect all "
                             f"'{GOAL_KEY_PREFIX}*' keys in the goal npz files")
    parser.add_argument("--demos", default=None,
                        help="comma-separated demo names (e.g. demo_0,demo_3); "
                             "default: all demo_*/ under demo_root")
    parser.add_argument("--scene", action="store_true",
                        help="draw the frame-0 scene point cloud as faint context")
    parser.add_argument("--dot_size", type=float, default=120.0)
    parser.add_argument("--all_goal_points", action="store_true",
                        help="plot all 4 gripper keypoints per goal instead of "
                             "just the goal center")
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    args = parser.parse_args()

    if args.demos:
        demo_names = args.demos.split(",")
    else:
        demo_names = sorted(
            [e for e in os.listdir(args.demo_root)
             if e.startswith("demo_")
             and os.path.isdir(os.path.join(args.demo_root, e))],
            key=lambda x: int(x.split("_")[1]),
        )
    if not demo_names:
        raise FileNotFoundError(f"No demo_*/ dirs under {args.demo_root}")

    # Detect keypoint types once, from the first demo's first goal frame.
    first_goal_dir = os.path.join(args.goal_root, demo_names[0])
    frames0 = sorted_frame_files(first_goal_dir)
    if args.goal_keys:
        goal_keys = args.goal_keys.split(",")
    else:
        goal_keys = detect_goal_keys(first_goal_dir, frames0)
    if not goal_keys:
        raise KeyError(f"No '{GOAL_KEY_PREFIX}*' keys found in {first_goal_dir}")

    folders = {k: key_to_folder(k) for k in goal_keys}
    colors = {k: TYPE_COLORS[i % len(TYPE_COLORS)] for i, k in enumerate(goal_keys)}
    for sub in list(folders.values()) + ["combined"]:
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    print(f"Keypoint types: {', '.join(f'{k} -> {folders[k]}/' for k in goal_keys)}")
    print(f"{len(demo_names)} demo(s) -> {args.out_dir}")

    for demo in demo_names:
        demo_dir = os.path.join(args.demo_root, demo)
        goal_dir = os.path.join(args.goal_root, demo)
        gripper, goals, scene = load_demo(demo_dir, goal_dir, goal_keys, args.scene)
        path = gripper.mean(axis=1)  # (T, 3) gripper center per frame

        uniq = {k: unique_consecutive_goals(goals[k]) for k in goal_keys}

        for k in goal_keys:
            ug, ff = uniq[k]
            plot_demo(
                os.path.join(args.out_dir, folders[k], f"{demo}.png"),
                f"{demo} — gripper path + {k}",
                path, [(folders[k], colors[k], ug, ff)], scene, args,
            )

        plot_demo(
            os.path.join(args.out_dir, "combined", f"{demo}.png"),
            f"{demo} — gripper path + all keypoint types",
            path,
            [(folders[k], colors[k], *uniq[k]) for k in goal_keys],
            scene, args,
        )
        counts = ", ".join(f"{folders[k]}={len(uniq[k][0])}" for k in goal_keys)
        print(f"  {demo}: {len(path)} frames, goals: {counts}")

    print(f"[done] plots under {args.out_dir}")


if __name__ == "__main__":
    main()
