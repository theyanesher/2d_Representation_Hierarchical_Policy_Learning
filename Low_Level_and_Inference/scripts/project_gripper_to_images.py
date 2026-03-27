"""
Project goal_gripper_pcd onto camera images as Gaussian heatmaps,
save heatmaps into a new h5 dataset and save visualizations per file.

# Single file — visualize one timestep (heatmap h5 always has all timesteps)
pixi run python scripts/project_gripper_to_images.py data/rgb_mino_data/41510/2025-10-30-21-05-53.h5 --timestep 40

# Single file — visualize all timesteps
pixi run python scripts/project_gripper_to_images.py data/rgb_mino_data/41510/2025-10-30-21-05-53.h5 --sigma 20

# All files in directory
pixi run python scripts/project_gripper_to_images.py data/rgb_mino_data/41510/

# Custom dirs / sigma
pixi run python scripts/project_gripper_to_images.py data/rgb_mino_data/41510/ --heatmap_dir outputs/Heatmap_Articubot_Dataset --viz_dir outputs/gripper_projection --sigma 20.0

pixi run python scripts/project_gripper_to_images.py data/rgb_mino_data/41510/ --tmap_Articubot_Dataset --viz_dir outputs/gripper_projection --sigma 20.0


pixi run python scripts/project_gripper_to_images.py data/rgb_mino_data/41510/ --heatmap_dir outputs/Ghost_Heatmap_Dataset


pixi run python scripts/project_gripper_to_images.py data/rgb_mino_data/41510/ --no_ghost_heatmap --no_all_four_points --sigma 20.0



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

# Add manipulation to path for rotation_transfer_6D_to_matrix
sys.path.insert(0, str(Path(__file__).parent.parent))
from manipulation.utils import rotation_transfer_6D_to_matrix

_ORIGINAL_GRIPPER_PCD = np.array([
    [ 0.10432111,  0.00228697,  0.8474241 ],
    [ 0.12816067, -0.04368229,  0.8114649 ],
    [ 0.08953098,  0.0484529 ,  0.80711854],
    [ 0.11198021,  0.00245327,  0.7828771 ],
], dtype=np.float64)
_ORIGINAL_GRIPPER_ORN = np.array([0.97841681, 0.19802945, 0.0581003, 0.01045192])


def _quaternion_to_rotation_matrix(quat):
    return _Rotation.from_quat(quat).as_matrix()


def get_points_from_pos_rotation_matrix(pos, orient):
    """Convert 9D pose (pos[3] + rot6d[6]) to 4 gripper keypoints (4,3)."""
    if isinstance(orient, torch.Tensor):
        orient = orient.numpy()
    absolute_rotation = rotation_transfer_6D_to_matrix(orient)
    original_R = _quaternion_to_rotation_matrix(_ORIGINAL_GRIPPER_ORN)
    rotation_transfer = absolute_rotation * original_R.T
    original_pcd = _ORIGINAL_GRIPPER_PCD - _ORIGINAL_GRIPPER_PCD[3]
    rotated_pcd = np.dot(original_pcd, rotation_transfer.T)
    return rotated_pcd + pos


NUM_CAMS = 3


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def project_world_to_pixel(point_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    """
    Project a 3D world point to pixel coordinates.

    extrinsic : (4,4) world-to-camera transform (as stored in h5)
    intrinsic : (3,3) camera intrinsic matrix
    Returns (u, v, z) or None if point is behind camera.
    """
    p_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    p_cam = extrinsic @ p_h
    z = p_cam[2]
    if z <= 0:
        return None
    p_img = intrinsic @ p_cam[:3]
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    return float(u), float(v), float(z)


def gaussian_heatmap(H: int, W: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Return (H, W) float32 Gaussian centered at (cx, cy), peak=1, or zeros if not visible."""
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    return np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))


def ghost_heatmap(points_2d, img_shape, n=0.5, sigma=1):
    """
    Generate a 4-channel heatmap from projected 2D points.

    Creates a distance-based heatmap where each channel represents the distance
    from every pixel to one of the projected gripper points.

    Args:
        points_2d (np.ndarray): Nx2 array of 2D pixel coordinates
        img_shape (tuple): (height, width) of output image

    Returns:
        np.ndarray: HxWx4 heatmap image with uint8 values [0-255]
    """
    height, width = img_shape[:2]
    max_distance = np.sqrt(width**2 + height**2)

    # Clip points to image bounds
    clipped_points = np.clip(points_2d, [0, 0], [width - 1, height - 1]).astype(int)

    num_pts = clipped_points.shape[0]
    goal_image = np.zeros((height, width, num_pts))
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    pixel_coords = np.stack([x_coords, y_coords], axis=-1)

    for i in range(num_pts):
        target_point = clipped_points[i]  # (2,)
        distances = np.linalg.norm(pixel_coords - target_point, axis=-1)  # (height, width)
        goal_image[:, :, i] = distances

    # Apply square root transformation for steeper near-target gradients
    goal_image = (goal_image / max_distance / sigma) ** n  * 255
    goal_image = np.clip(goal_image, 0, 255).astype(np.uint8)
    return goal_image


def compute_heatmap_for_cam(point_world, extrinsic, intrinsic, H, W, sigma):
    """
    Returns (H, W) float32 Gaussian heatmap [0,1], or zeros if not visible.
    """
    result = project_world_to_pixel(point_world, extrinsic, intrinsic)
    if result is None:
        return np.zeros((H, W), dtype=np.float32)
    u, v, _ = result
    in_frame = (0 <= int(round(u)) < W) and (0 <= int(round(v)) < H)
    if not in_frame:
        return np.zeros((H, W), dtype=np.float32)
    return gaussian_heatmap(H, W, cx=u, cy=v, sigma=sigma)


def compute_ghost_heatmap_for_cam(points_world, extrinsic, intrinsic, H, W, n=0.5, sigma=1):
    """
    Projects multiple 3D world points to 2D and returns a ghost_heatmap (H, W, 4) uint8.

    points_world : (N, 3) array of world-space keypoints (at least 4 required).
    Falls back to repeating the first point if fewer than 4 are provided.
    """
    points_world = np.atleast_2d(points_world)  # (N, 3)
    # Ensure at least 4 points by repeating if necessary
    while points_world.shape[0] < 4:
        points_world = np.vstack([points_world, points_world[:1]])

    points_2d = []
    for pt in points_world[:4]:
        result = project_world_to_pixel(pt, extrinsic, intrinsic)
        if result is None:
            points_2d.append(np.array([0.0, 0.0]))
        else:
            u, v, _ = result
            points_2d.append(np.array([u, v]))
    points_2d = np.array(points_2d)  # (4, 2)
    return ghost_heatmap(points_2d, (H, W), n=n, sigma=sigma)


def make_overlay(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend heatmap (green channel) onto RGB image."""
    overlay = rgb.astype(np.float32)
    overlay[..., 1] = np.clip(overlay[..., 1] + heatmap * 255, 0, 255)
    return overlay.astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_h5_file(h5_path: Path, viz_dir: Path, heatmap_dir: Path,
                    sigma: float, timestep: int = None, use_ghost_heatmap: bool = False,
                    use_all_four_points: bool = False):
    """
    Process one h5 file:
      - Compute heatmaps (Gaussian or ghost) for cam0/1/2 at every timestep
      - Save heatmaps as obs/cam{i}_heatmap in a new h5 in heatmap_dir/
      - Save visualization PNGs in viz_dir/<stem>/
      - If timestep is not None, only save viz for that one timestep
    """
    fname_prefix = h5_path.stem

    # Visualization goes in its own subfolder named after the h5 file
    file_viz_dir = viz_dir / fname_prefix
    file_viz_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap h5 mirrors the source filename
    out_h5_path = heatmap_dir / h5_path.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as src:
        num_frames = src['obs/cam0_image'].shape[0]
        H, W = src['obs/cam0_image'].shape[1:3]

        print(f"\nProcessing: {h5_path.name}  |  frames={num_frames}  |  sigma={sigma}")

        # ----- Compute all heatmaps up front per camera -----
        # Gaussian: (T, H, W) float32   Ghost: (T, H, W, 4) uint8
        if use_ghost_heatmap:
            heatmaps = {cam: np.zeros((num_frames, H, W, 4), dtype=np.uint8)
                        for cam in range(NUM_CAMS)}
        else:
            heatmaps = {cam: np.zeros((num_frames, H, W), dtype=np.float32)
                        for cam in range(NUM_CAMS)}

        for t in range(num_frames):
            goal_pcd = src['obs/goal_gripper_pcd'][t]  # (10,): pos(3) + rot6d(6) + grip(1)
            for cam in range(NUM_CAMS):
                E = src[f'obs/cam{cam}_extrinsic'][t]
                K = src[f'obs/cam{cam}_intrinsic'][t]
                if use_ghost_heatmap:
                    if use_all_four_points:
                        # Convert 10D → 4 gripper keypoints (4, 3)
                        pos = np.array(goal_pcd[:3], dtype=np.float32)
                        orient = torch.tensor(goal_pcd[3:9], dtype=torch.float32)
                        pts = get_points_from_pos_rotation_matrix(pos, orient)  # (4, 3)
                    else:
                        pts = np.array(goal_pcd[:3], dtype=np.float32).reshape(1, 3)
                    heatmaps[cam][t] = compute_ghost_heatmap_for_cam(pts, E, K, H, W)
                else:
                    heatmaps[cam][t] = compute_heatmap_for_cam(goal_pcd[:3], E, K, H, W, sigma)

        # ----- Save everything to new h5: copy source + add heatmaps -----
        # When saving ghost heatmaps, skip any existing cam{i}_heatmap keys
        # so the output only has cam{i}_heatmap_ghost (not both).
        _heatmap_keys = {f"cam{c}_heatmap" for c in range(NUM_CAMS)}

        with h5py.File(out_h5_path, "w") as dst:
            for key in src.keys():
                if key == "obs" and use_ghost_heatmap:
                    dst.create_group("obs")
                    for obs_key in src["obs"].keys():
                        if obs_key not in _heatmap_keys:
                            src.copy(f"obs/{obs_key}", dst["obs"])
                else:
                    src.copy(key, dst)
            # Add heatmaps into obs/
            for cam in range(NUM_CAMS):
                if use_ghost_heatmap:
                    dst.create_dataset(
                        f"obs/cam{cam}_heatmap_ghost",
                        data=heatmaps[cam],  # (T, H, W, 4) uint8
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    dst.create_dataset(
                        f"obs/cam{cam}_heatmap",
                        data=heatmaps[cam].astype(np.float16),  # (T, H, W) float16
                        compression="gzip",
                        compression_opts=4,
                    )
        print(f"  Saved (source + heatmaps) -> {out_h5_path}")

        # ----- Save visualizations -----
        timesteps_to_viz = range(num_frames) if timestep is None else [min(timestep, num_frames - 1)]

        for t in timesteps_to_viz:
            rows_mask, rows_overlay = [], []

            # Recompute world keypoints for this timestep (used for dot overlay)
            goal_pcd_t = src['obs/goal_gripper_pcd'][t]
            if use_all_four_points:
                pos_t = np.array(goal_pcd_t[:3], dtype=np.float32)
                orient_t = torch.tensor(goal_pcd_t[3:9], dtype=torch.float32)
                pts_world = get_points_from_pos_rotation_matrix(pos_t, orient_t)  # (4, 3)
            else:
                pts_world = np.array(goal_pcd_t[:3], dtype=np.float32).reshape(1, 3)

            for cam in range(NUM_CAMS):
                rgb = src[f'obs/cam{cam}_image'][t]          # (H, W, 3) uint8
                E = src[f'obs/cam{cam}_extrinsic'][t]
                K = src[f'obs/cam{cam}_intrinsic'][t]

                if use_ghost_heatmap:
                    # ghost heatmap is (H, W, 4) uint8 — use first 3 channels for RGB viz
                    mask_img = np.ascontiguousarray(heatmaps[cam][t, :, :, :3])
                    overlay_img = np.clip(
                        rgb.astype(np.float32) * 0.6 + mask_img.astype(np.float32) * 0.4, 0, 255
                    ).astype(np.uint8)
                else:
                    hm = heatmaps[cam][t].astype(np.float16).astype(np.float32)  # simulate float16 storage
                    hm_u8 = (hm * 255).astype(np.uint8)
                    mask_img = np.stack([hm_u8] * 3, axis=-1)
                    overlay_img = make_overlay(rgb, hm)

                # Draw projected keypoints as blue filled circles with index label
                for i, pt_world in enumerate(pts_world):
                    result = project_world_to_pixel(pt_world, E, K)
                    if result is not None:
                        u, v, _ = result
                        cx, cy = int(round(u)), int(round(v))
                        if 0 <= cx < W and 0 <= cy < H:
                            cv2.circle(overlay_img, (cx, cy), radius=5,
                                       color=(255, 0, 0), thickness=-1)   # blue (BGR)
                            cv2.circle(mask_img,    (cx, cy), radius=5,
                                       color=(255, 0, 0), thickness=-1)
                            cv2.putText(overlay_img, str(i), (cx + 6, cy + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

                # Camera label
                cv2.putText(overlay_img, f"cam{cam}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(mask_img, f"cam{cam}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                rows_mask.append(mask_img)
                rows_overlay.append(overlay_img)

            # [cam0 | cam1 | cam2] stacked horizontally, overlay on top / mask below
            grid = np.concatenate(
                [np.concatenate(rows_overlay, axis=1),
                 np.concatenate(rows_mask, axis=1)],
                axis=0
            )

            out_path = file_viz_dir / f"t{t:04d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        print(f"  Visualizations saved -> {file_viz_dir}/  ({len(timesteps_to_viz)} frames)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input",
                        help="Path to a single .h5 file OR a directory containing .h5 files")
    parser.add_argument("--timestep", type=int, default=None,
                        help="Only save visualization for this timestep (default: all timesteps). "
                             "Heatmap h5 always contains all timesteps.")
    parser.add_argument("--viz_dir", default="outputs/gripper_projection",
                        help="Root dir for visualization PNGs (default: outputs/gripper_projection)")
    parser.add_argument("--heatmap_dir", default="outputs/Heatmap_Articubot_Dataset",
                        help="Root dir for heatmap h5 files (default: outputs/Heatmap_Articubot_Dataset)")
    parser.add_argument("--sigma", type=float, default=20.0,
                        help="Gaussian sigma in pixels (default: 20.0)")
    parser.add_argument("--no_ghost_heatmap", action="store_true",
                        help="Use Gaussian heatmap instead of the default ghost_heatmap (distance-field, 3-channel)")
    parser.add_argument("--no_all_four_points", action="store_true",
                        help="Use only the single position point instead of the default 4-keypoint "
                             "representation derived from the full 10D pose")
    args = parser.parse_args()

    input_path = Path(args.input)
    viz_dir = Path(args.viz_dir)
    heatmap_dir = Path(args.heatmap_dir)

    if input_path.is_dir():
        h5_files = sorted(input_path.glob("*.h5"))
        if not h5_files:
            print(f"No .h5 files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(h5_files)} .h5 files in {input_path}")
        for h5_path in h5_files:
            process_h5_file(h5_path, viz_dir, heatmap_dir, args.sigma, args.timestep,
                            use_ghost_heatmap=not args.no_ghost_heatmap,
                            use_all_four_points=not args.no_all_four_points)
    elif input_path.is_file():
        process_h5_file(input_path, viz_dir, heatmap_dir, args.sigma, args.timestep,
                        use_ghost_heatmap=not args.no_ghost_heatmap,
                        use_all_four_points=not args.no_all_four_points)
    else:
        print(f"Path not found: {input_path}")
        sys.exit(1)

    print("\nAll done.")


if __name__ == "__main__":
    main()
