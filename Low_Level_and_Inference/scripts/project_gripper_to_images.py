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


def make_overlay(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend heatmap (green channel) onto RGB image."""
    overlay = rgb.astype(np.float32)
    overlay[..., 1] = np.clip(overlay[..., 1] + heatmap * 255, 0, 255)
    return overlay.astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_h5_file(h5_path: Path, viz_dir: Path, heatmap_dir: Path,
                    sigma: float, timestep: int = None):
    """
    Process one h5 file:
      - Compute Gaussian heatmaps for cam0/1/2 at every timestep
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

        # ----- Compute all heatmaps up front (T, H, W) per camera -----
        heatmaps = {cam: np.zeros((num_frames, H, W), dtype=np.float32)
                    for cam in range(NUM_CAMS)}

        for t in range(num_frames):
            point_world = src['obs/goal_gripper_pcd'][t][:3]
            for cam in range(NUM_CAMS):
                E = src[f'obs/cam{cam}_extrinsic'][t]
                K = src[f'obs/cam{cam}_intrinsic'][t]
                heatmaps[cam][t] = compute_heatmap_for_cam(point_world, E, K, H, W, sigma)

        # ----- Save everything to new h5: copy source + add heatmaps -----
        with h5py.File(out_h5_path, "w") as dst:
            # Copy all existing groups/datasets from source
            for key in src.keys():
                src.copy(key, dst)
            # Add heatmaps into obs/
            for cam in range(NUM_CAMS):
                dst.create_dataset(
                    f"obs/cam{cam}_heatmap",
                    data=heatmaps[cam].astype(np.float16),  # (T, H, W) float16, values in [0,1]
                    compression="gzip",
                    compression_opts=4,
                )
        print(f"  Saved (source + heatmaps) -> {out_h5_path}")

        # ----- Save visualizations -----
        timesteps_to_viz = range(num_frames) if timestep is None else [min(timestep, num_frames - 1)]

        for t in timesteps_to_viz:
            point_world = src['obs/goal_gripper_pcd'][t][:3]
            rows_mask, rows_overlay = [], []

            for cam in range(NUM_CAMS):
                rgb = src[f'obs/cam{cam}_image'][t]          # (H, W, 3) uint8
                hm = heatmaps[cam][t].astype(np.float16).astype(np.float32)  # simulate float16 storage

                # Mask: white-on-black grayscale heatmap
                hm_u8 = (hm * 255).astype(np.uint8)
                mask_img = np.stack([hm_u8] * 3, axis=-1)

                overlay_img = make_overlay(rgb, hm)

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
    parser.add_argument("--sigma", type=float, default=8.0,
                        help="Gaussian sigma in pixels (default: 8.0)")
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
            process_h5_file(h5_path, viz_dir, heatmap_dir, args.sigma, args.timestep)
    elif input_path.is_file():
        process_h5_file(input_path, viz_dir, heatmap_dir, args.sigma, args.timestep)
    else:
        print(f"Path not found: {input_path}")
        sys.exit(1)

    print("\nAll done.")


if __name__ == "__main__":
    main()
