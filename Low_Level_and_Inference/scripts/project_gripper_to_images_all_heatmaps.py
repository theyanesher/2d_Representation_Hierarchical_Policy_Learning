"""
Project goal_gripper_pcd onto camera images as BOTH ghost (sqrt/power) and
Gaussian (exp) heatmaps, saving both into a single output h5 file.

Both heatmap types always use all 4 gripper keypoints (one channel per keypoint).

Output h5 contains (in addition to all source data):
  obs/cam{i}_heatmap_ghost         : (T, H, W, 4) uint8  — goal,    sqrt/power distance
  obs/cam{i}_heatmap               : (T, H, W, 4) uint8  — goal,    Gaussian (exp), scaled [0,255]
  obs/cam{i}_present_heatmap_ghost : (T, H, W, 4) uint8  — present, sqrt/power distance
  obs/cam{i}_present_heatmap       : (T, H, W, 4) uint8  — present, Gaussian (exp), scaled [0,255]
  obs/goal_gripper_pts             : (T, 4, 3)   float32 — 4 goal keypoints per timestep
  obs/present_gripper_pts          : (T, 4, 3)   float32 — 4 present keypoints per timestep

Visualizations are saved as PNGs in separate subfolders:
  viz_dir/ghost/<stem>/t{t:04d}.png
  viz_dir/gaussian/<stem>/t{t:04d}.png

# Single file
pixi run python scripts/project_gripper_to_images_all_heatmaps.py data/rgb_mino_data/41510/2025-10-30-21-05-53.h5 --timestep 40

# All files in a directory
pixi run python scripts/project_gripper_to_images_all_heatmaps.py data/rgb_mino_data/41510/

# Custom output dirs / sigmas
pixi run python scripts/project_gripper_to_images_all_heatmaps.py ../../../ArticuBot/data/rgb_mino_data/41510/ --heatmap_dir outputs/All_Heatmap
_Dataset --viz_dir outputs/all_heatmap_viz --sigma_gaussian 20.0 --sigma_ghost 1.0
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python required: pip install opencv-python")

import torch
from scipy.spatial.transform import Rotation as _Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))
from manipulation.utils import rotation_transfer_6D_to_matrix

_ORIGINAL_GRIPPER_PCD = np.array([
    [ 0.10432111,  0.00228697,  0.8474241 ],
    [ 0.12816067, -0.04368229,  0.8114649 ],
    [ 0.08953098,  0.0484529 ,  0.80711854],
    [ 0.11198021,  0.00245327,  0.7828771 ],
], dtype=np.float64)
_ORIGINAL_GRIPPER_ORN = np.array([0.97841681, 0.19802945, 0.0581003, 0.01045192])

NUM_CAMS = 3


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quaternion_to_rotation_matrix(quat):
    return _Rotation.from_quat(quat).as_matrix()


def get_points_from_pos_rotation_matrix(pos, orient):
    """Convert 9D pose (pos[3] + rot6d[6]) to 4 gripper keypoints (4, 3)."""
    if isinstance(orient, torch.Tensor):
        orient = orient.numpy()
    absolute_rotation = rotation_transfer_6D_to_matrix(orient)
    original_R = _quaternion_to_rotation_matrix(_ORIGINAL_GRIPPER_ORN)
    rotation_transfer = absolute_rotation * original_R.T
    original_pcd = _ORIGINAL_GRIPPER_PCD - _ORIGINAL_GRIPPER_PCD[3]
    rotated_pcd = np.dot(original_pcd, rotation_transfer.T)
    return rotated_pcd + pos  # (4, 3)


def project_world_to_pixel(point_world, extrinsic, intrinsic):
    """Project a 3D world point to pixel (u, v, z). Returns None if behind camera."""
    p_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    p_cam = extrinsic @ p_h
    z = p_cam[2]
    if z <= 0:
        return None
    p_img = intrinsic @ p_cam[:3]
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    return float(u), float(v), float(z)


# ---------------------------------------------------------------------------
# Heatmap generators
# ---------------------------------------------------------------------------

def gaussian_heatmap_single(H, W, cx, cy, sigma):
    """(H, W) float32 Gaussian centered at (cx, cy), peak=1."""
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    return np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))


def compute_gaussian_heatmap_4ch(points_world, extrinsic, intrinsic, H, W, sigma):
    """
    Gaussian (exp) heatmap with one channel per keypoint.
    Returns (H, W, 4) uint8, values scaled to [0, 255].
    """
    out = np.zeros((H, W, 4), dtype=np.float32)
    for i, pt in enumerate(points_world[:4]):
        result = project_world_to_pixel(pt, extrinsic, intrinsic)
        if result is None:
            continue
        u, v, _ = result
        out[..., i] = gaussian_heatmap_single(H, W, u, v, sigma)
    return (out * 255).clip(0, 255).astype(np.uint8)


def compute_ghost_heatmap_4ch(points_world, extrinsic, intrinsic, H, W, n=0.5, sigma=1):
    """
    sqrt/power distance heatmap with one channel per keypoint.
    Returns (H, W, 4) uint8.
    """
    height, width = H, W
    max_distance = np.sqrt(width ** 2 + height ** 2)

    points_2d = []
    for pt in points_world[:4]:
        result = project_world_to_pixel(pt, extrinsic, intrinsic)
        if result is None:
            points_2d.append(np.array([0.0, 0.0]))
        else:
            u, v, _ = result
            points_2d.append(np.array([u, v]))
    points_2d = np.clip(np.array(points_2d), [0, 0], [width - 1, height - 1]).astype(int)

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    pixel_coords = np.stack([x_coords, y_coords], axis=-1)

    out = np.zeros((height, width, 4), dtype=np.float32)
    for i, pt2d in enumerate(points_2d):
        distances = np.linalg.norm(pixel_coords - pt2d, axis=-1)
        out[..., i] = distances

    out = (out / max_distance / sigma) ** n * 255
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _draw_keypoints(img, pts_world, extrinsic, intrinsic, H, W):
    """Draw projected keypoints as blue circles with index labels on img (in-place)."""
    for i, pt_world in enumerate(pts_world):
        result = project_world_to_pixel(pt_world, extrinsic, intrinsic)
        if result is None:
            continue
        u, v, _ = result
        cx, cy = int(round(u)), int(round(v))
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(img, (cx, cy), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.putText(img, str(i), (cx + 6, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)


def _make_ghost_viz_row(rgb, ghost_4ch, pts_world, extrinsic, intrinsic, H, W, cam_idx):
    """
    Build [overlay | ch0 | ch1 | ch2 | ch3] visualization for ghost heatmap.
    overlay = rgb blended with first 3 channels of ghost heatmap.
    """
    mask_rgb = np.ascontiguousarray(ghost_4ch[..., :3])  # use first 3 ch as RGB visualization
    overlay = np.clip(
        rgb.astype(np.float32) * 0.6 + mask_rgb.astype(np.float32) * 0.4, 0, 255
    ).astype(np.uint8)
    _draw_keypoints(overlay, pts_world, extrinsic, intrinsic, H, W)
    _draw_keypoints(mask_rgb, pts_world, extrinsic, intrinsic, H, W)
    cv2.putText(overlay, f"cam{cam_idx}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # Individual channel panels (grayscale → BGR)
    ch_panels = []
    for ch in range(4):
        ch_img = np.stack([ghost_4ch[..., ch]] * 3, axis=-1)
        cv2.putText(ch_img, f"cam{cam_idx} ch{ch}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        ch_panels.append(ch_img)

    return np.concatenate([overlay, mask_rgb] + ch_panels, axis=1)


def _make_gaussian_viz_row(rgb, gauss_4ch, pts_world, extrinsic, intrinsic, H, W, cam_idx):
    """
    Build [overlay | ch0 | ch1 | ch2 | ch3] visualization for Gaussian heatmap.
    overlay = green-channel blend of mean heatmap onto RGB.
    """
    mean_hm = gauss_4ch.mean(axis=-1).astype(np.uint8)  # (H, W) uint8
    overlay = rgb.astype(np.float32).copy()
    overlay[..., 1] = np.clip(overlay[..., 1] + mean_hm.astype(np.float32), 0, 255)
    overlay = overlay.astype(np.uint8)
    _draw_keypoints(overlay, pts_world, extrinsic, intrinsic, H, W)
    cv2.putText(overlay, f"cam{cam_idx}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    mean_panel = np.stack([mean_hm] * 3, axis=-1)
    _draw_keypoints(mean_panel, pts_world, extrinsic, intrinsic, H, W)

    ch_panels = []
    for ch in range(4):
        ch_img = np.stack([gauss_4ch[..., ch]] * 3, axis=-1)
        cv2.putText(ch_img, f"cam{cam_idx} ch{ch}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        ch_panels.append(ch_img)

    return np.concatenate([overlay, mean_panel] + ch_panels, axis=1)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _compute_norm_dist(goal_gripper_pts, present_gripper_pts, goal_pcd_all):
    """
    Compute normalised gripper-to-goal distance per timestep.

    f_gt(t) = clip(1 - d(t) / d(subgoal_start), 0, 1)
    where d(t) = mean L2 over 4 keypoints between present and goal.

    Returns (T,) float32 array and prints a monotonicity warning if needed.
    """
    num_frames = goal_gripper_pts.shape[0]

    # Subgoal boundaries: timesteps where goal_gripper_pcd changes
    subgoal_starts = [0] + [t for t in range(1, num_frames)
                            if not np.allclose(goal_pcd_all[t], goal_pcd_all[t - 1])]

    # Mean L2 distance per frame
    dists = np.linalg.norm(
        present_gripper_pts - goal_gripper_pts, axis=-1
    ).mean(axis=-1)  # (T,)

    # Map each frame to its subgoal's reference (start) distance
    subgoal_idx = np.zeros(num_frames, dtype=int)
    for i, s in enumerate(subgoal_starts[1:], 1):
        subgoal_idx[s:] = i
    d_start_per_frame = dists[np.array(subgoal_starts)][subgoal_idx]  # (T,)

    norm_dist = np.where(
        d_start_per_frame > 1e-6,
        np.clip(1.0 - dists / d_start_per_frame, 0.0, 1.0),
        1.0,
    ).astype(np.float32)

    # Monotonicity check per subgoal segment (warn if >10% of steps decrease)
    for i, s in enumerate(subgoal_starts):
        end = subgoal_starts[i + 1] if i + 1 < len(subgoal_starts) else num_frames
        seg = norm_dist[s:end]
        if len(seg) > 1:
            n_drops = int((np.diff(seg) < -0.01).sum())
            frac = n_drops / (len(seg) - 1)
            if frac > 0.10:
                print(f"  [WARN] subgoal {i} (t={s}..{end-1}): "
                      f"{n_drops}/{len(seg)-1} ({frac:.0%}) non-monotone steps")

    return norm_dist, subgoal_starts


def add_norm_dist_to_existing(h5_path: Path):
    """
    Open an already-processed h5 file from heatmap_dir in-place and add
    closed_loop_requirements/low_level/norm_dist_grippers_subgoals.
    Skips if the dataset already exists.
    """
    print(f"\nAdding norm_dist: {h5_path.name}")
    dest_key = "closed_loop_requirements/low_level/norm_dist_grippers_subgoals"
    try:
        with h5py.File(h5_path, "r+") as f:
            try:
                already = dest_key in f
            except Exception:
                already = False
                # Broken group from a previous interrupted write — delete and rewrite
                print(f"  [WARN] Broken closed_loop_requirements group detected — will delete and rewrite.")
                if "closed_loop_requirements" in f.keys():
                    del f["closed_loop_requirements"]
            if already:
                print(f"  Already exists — skipping.")
                return
    except Exception as e:
        print(f"  [ERROR] Could not open file: {e} — skipping.")
        return
    with h5py.File(h5_path, "r+") as f:

        goal_gripper_pts    = f["obs/goal_gripper_pts"][:]     # (T, 4, 3)
        present_gripper_pts = f["obs/present_gripper_pts"][:]  # (T, 4, 3)
        goal_pcd_all        = f["obs/goal_gripper_pcd"][:]     # (T, 10)

        norm_dist, subgoal_starts = _compute_norm_dist(
            goal_gripper_pts, present_gripper_pts, goal_pcd_all
        )
        print(f"  frames={len(norm_dist)}  subgoals={len(subgoal_starts)}  "
              f"starts={subgoal_starts}  "
              f"norm_dist range=[{norm_dist.min():.3f}, {norm_dist.max():.3f}]")

        grp = f.require_group("closed_loop_requirements").require_group("low_level")
        grp.create_dataset("norm_dist_grippers_subgoals", data=norm_dist,
                           compression="gzip", compression_opts=4)
    print(f"  Saved -> {h5_path}")


def process_h5_file(h5_path: Path, viz_dir: Path, heatmap_dir: Path,
                    sigma_gaussian: float, sigma_ghost: float,
                    timestep: int = None, viz: bool = False):
    fname_prefix = h5_path.stem

    if viz:
        ghost_viz_dir           = viz_dir / "goal_ghost"            / fname_prefix
        gauss_viz_dir           = viz_dir / "goal_gaussian"          / fname_prefix
        present_ghost_viz_dir   = viz_dir / "present_ghost"          / fname_prefix
        present_gauss_viz_dir   = viz_dir / "present_gaussian"       / fname_prefix
        for d in (ghost_viz_dir, gauss_viz_dir, present_ghost_viz_dir, present_gauss_viz_dir):
            d.mkdir(parents=True, exist_ok=True)

    out_h5_path = heatmap_dir / h5_path.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as src:
        num_frames = src['obs/cam0_image'].shape[0]
        H, W = src['obs/cam0_image'].shape[1:3]

        print(f"\nProcessing: {h5_path.name}  |  frames={num_frames}  |  sigma_gaussian={sigma_gaussian}  sigma_ghost={sigma_ghost}")

        # Pre-allocate heatmap arrays — goal and present, both types
        ghost_heatmaps         = {cam: np.zeros((num_frames, H, W, 4), dtype=np.uint8) for cam in range(NUM_CAMS)}
        gauss_heatmaps         = {cam: np.zeros((num_frames, H, W, 4), dtype=np.uint8) for cam in range(NUM_CAMS)}
        present_ghost_heatmaps = {cam: np.zeros((num_frames, H, W, 4), dtype=np.uint8) for cam in range(NUM_CAMS)}
        present_gauss_heatmaps = {cam: np.zeros((num_frames, H, W, 4), dtype=np.uint8) for cam in range(NUM_CAMS)}
        goal_gripper_pts    = np.zeros((num_frames, 4, 3), dtype=np.float32)
        present_gripper_pts = np.zeros((num_frames, 4, 3), dtype=np.float32)

        goal_pcd_all = src['obs/goal_gripper_pcd'][:]   # (T, 10)

        for t in range(num_frames):
            # Goal gripper keypoints
            goal_pcd = goal_pcd_all[t]  # (10,)
            pos    = np.array(goal_pcd[:3], dtype=np.float32)
            orient = torch.tensor(goal_pcd[3:9], dtype=torch.float32)
            goal_pts = get_points_from_pos_rotation_matrix(pos, orient)  # (4, 3)
            goal_gripper_pts[t] = goal_pts.astype(np.float32)

            # Present gripper keypoints (from obs/state, same 10D format)
            state = src['obs/state'][t]  # (10,)
            pres_pos    = np.array(state[:3], dtype=np.float32)
            pres_orient = torch.tensor(state[3:9], dtype=torch.float32)
            pres_pts = get_points_from_pos_rotation_matrix(pres_pos, pres_orient)  # (4, 3)
            present_gripper_pts[t] = pres_pts.astype(np.float32)

            for cam in range(NUM_CAMS):
                E = src[f'obs/cam{cam}_extrinsic'][t]
                K = src[f'obs/cam{cam}_intrinsic'][t]
                ghost_heatmaps[cam][t]         = compute_ghost_heatmap_4ch(goal_pts, E, K, H, W, sigma=sigma_ghost)
                gauss_heatmaps[cam][t]         = compute_gaussian_heatmap_4ch(goal_pts, E, K, H, W, sigma_gaussian)
                present_ghost_heatmaps[cam][t] = compute_ghost_heatmap_4ch(pres_pts, E, K, H, W, sigma=sigma_ghost)
                present_gauss_heatmaps[cam][t] = compute_gaussian_heatmap_4ch(pres_pts, E, K, H, W, sigma_gaussian)

        norm_dist, subgoal_starts = _compute_norm_dist(
            goal_gripper_pts, present_gripper_pts, goal_pcd_all
        )

        # ----- Write output h5 -----
        _existing_heatmap_keys = (
            {f"cam{c}_heatmap"               for c in range(NUM_CAMS)} |
            {f"cam{c}_heatmap_ghost"         for c in range(NUM_CAMS)} |
            {f"cam{c}_present_heatmap"       for c in range(NUM_CAMS)} |
            {f"cam{c}_present_heatmap_ghost" for c in range(NUM_CAMS)} |
            {"goal_gripper_pts", "present_gripper_pts"}
        )

        with h5py.File(out_h5_path, "w") as dst:
            dst.create_group("obs")
            for obs_key in src["obs"].keys():
                if obs_key not in _existing_heatmap_keys:
                    src.copy(f"obs/{obs_key}", dst["obs"])
            for key in src.keys():
                if key != "obs":
                    src.copy(key, dst)

            # Keypoints
            dst.create_dataset("obs/goal_gripper_pts",    data=goal_gripper_pts,    compression="gzip", compression_opts=4)
            dst.create_dataset("obs/present_gripper_pts", data=present_gripper_pts, compression="gzip", compression_opts=4)

            # Heatmaps — goal and present, both types
            for cam in range(NUM_CAMS):
                dst.create_dataset(f"obs/cam{cam}_heatmap_ghost",         data=ghost_heatmaps[cam],         compression="gzip", compression_opts=4)
                dst.create_dataset(f"obs/cam{cam}_heatmap",               data=gauss_heatmaps[cam],         compression="gzip", compression_opts=4)
                dst.create_dataset(f"obs/cam{cam}_present_heatmap_ghost", data=present_ghost_heatmaps[cam], compression="gzip", compression_opts=4)
                dst.create_dataset(f"obs/cam{cam}_present_heatmap",       data=present_gauss_heatmaps[cam], compression="gzip", compression_opts=4)

            # Normalised gripper-to-goal distance (f_gt)
            grp = dst.require_group("closed_loop_requirements").require_group("low_level")
            grp.create_dataset("norm_dist_grippers_subgoals", data=norm_dist, compression="gzip", compression_opts=4)

        print(f"  Saved -> {out_h5_path}")

        # ----- Visualizations (only when --viz is set) -----
        if viz:
            timesteps_to_viz = range(num_frames) if timestep is None else [min(timestep, num_frames - 1)]

            for t in timesteps_to_viz:
                goal_pts    = goal_gripper_pts[t]     # (4, 3)
                present_pts = present_gripper_pts[t]  # (4, 3)

                ghost_rows, gauss_rows, pres_ghost_rows, pres_gauss_rows = [], [], [], []
                for cam in range(NUM_CAMS):
                    rgb = src[f'obs/cam{cam}_image'][t]
                    E   = src[f'obs/cam{cam}_extrinsic'][t]
                    K   = src[f'obs/cam{cam}_intrinsic'][t]

                    ghost_rows.append(_make_ghost_viz_row(
                        rgb, ghost_heatmaps[cam][t], goal_pts, E, K, H, W, cam))
                    gauss_rows.append(_make_gaussian_viz_row(
                        rgb, gauss_heatmaps[cam][t], goal_pts, E, K, H, W, cam))
                    pres_ghost_rows.append(_make_ghost_viz_row(
                        rgb, present_ghost_heatmaps[cam][t], present_pts, E, K, H, W, cam))
                    pres_gauss_rows.append(_make_gaussian_viz_row(
                        rgb, present_gauss_heatmaps[cam][t], present_pts, E, K, H, W, cam))

                cv2.imwrite(str(ghost_viz_dir         / f"t{t:04d}.png"), cv2.cvtColor(np.concatenate(ghost_rows,      axis=0), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(gauss_viz_dir         / f"t{t:04d}.png"), cv2.cvtColor(np.concatenate(gauss_rows,      axis=0), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(present_ghost_viz_dir / f"t{t:04d}.png"), cv2.cvtColor(np.concatenate(pres_ghost_rows, axis=0), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(present_gauss_viz_dir / f"t{t:04d}.png"), cv2.cvtColor(np.concatenate(pres_gauss_rows, axis=0), cv2.COLOR_RGB2BGR))

            print(f"  Goal ghost viz     -> {ghost_viz_dir}/")
            print(f"  Goal gaussian viz  -> {gauss_viz_dir}/")
            print(f"  Pres ghost viz     -> {present_ghost_viz_dir}/")
            print(f"  Pres gaussian viz  -> {present_gauss_viz_dir}/  ({len(timesteps_to_viz)} frames each)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to a single .h5 file OR a directory of .h5 files "
                             "(not required when --norm_dist_only is set)")
    parser.add_argument("--timestep", type=int, default=None,
                        help="Only save visualization for this timestep (default: all). "
                             "The output h5 always contains all timesteps.")
    parser.add_argument("--viz_dir", default="outputs/all_heatmap_viz",
                        help="Root dir for visualization PNGs (default: outputs/all_heatmap_viz)")
    parser.add_argument("--heatmap_dir", default="outputs/All_Heatmap_Dataset",
                        help="Root dir for output h5 files (default: outputs/All_Heatmap_Dataset)")
    parser.add_argument("--sigma_gaussian", type=float, default=20.0,
                        help="Sigma for Gaussian (exp) heatmap in pixels (default: 20.0)")
    parser.add_argument("--sigma_ghost", type=float, default=1.0,
                        help="Sigma for ghost (sqrt/power) heatmap — scales the distance field (default: 1.0)")
    parser.add_argument("--viz", action="store_true", default=False,
                        help="Save heatmap visualization PNGs (default: off)")
    parser.add_argument("--norm_dist_only", action="store_true", default=False,
                        help="Only add norm_dist_grippers_subgoals to existing h5 files in "
                             "--heatmap_dir (in-place). 'input' arg is ignored in this mode.")
    args = parser.parse_args()

    heatmap_dir = Path(args.heatmap_dir)

    if args.norm_dist_only:
        h5_files = sorted(heatmap_dir.glob("*.h5"))
        if not h5_files:
            print(f"No .h5 files found in {heatmap_dir}")
            sys.exit(1)
        print(f"Found {len(h5_files)} .h5 files in {heatmap_dir} — adding norm_dist only")
        for h5_path in h5_files:
            add_norm_dist_to_existing(h5_path)
        print("\nAll done.")
        return

    if args.input is None:
        parser.error("'input' is required unless --norm_dist_only is set")

    input_path  = Path(args.input)
    viz_dir     = Path(args.viz_dir)

    if input_path.is_dir():
        h5_files = sorted(input_path.glob("*.h5"))
        if not h5_files:
            print(f"No .h5 files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(h5_files)} .h5 files in {input_path}")
        for h5_path in h5_files:
            process_h5_file(h5_path, viz_dir, heatmap_dir, args.sigma_gaussian, args.sigma_ghost,
                            args.timestep, viz=args.viz)
    elif input_path.is_file():
        process_h5_file(input_path, viz_dir, heatmap_dir, args.sigma_gaussian, args.sigma_ghost,
                        args.timestep, viz=args.viz)
    else:
        print(f"Path not found: {input_path}")
        sys.exit(1)

    print("\nAll done.")


if __name__ == "__main__":
    main()
