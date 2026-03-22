"""
Project fused multi-camera point clouds into canonical camera views.

Walks h5 files in a data directory, fuses point clouds from all 3 cameras
(cam0, cam1, cam2), and renders two canonical views (cam0_canonical, cam1_canonical)
using hardcoded canonical extrinsics/intrinsics. Results saved in-place as new keys.
"""

import h5py
import numpy as np
import typer
from tqdm import tqdm
from pathlib import Path

app = typer.Typer()

# ── Canonical camera parameters ──────────────────────────────────────────────
CAM0_CANONICAL_EXTRINSIC = np.array([  # W2C
    [ 0.70710677, -0.7071068 , -0.00000001, -0.49497473],
    [-0.1227878 , -0.12278777, -0.98480785,  0.47987458],
    [ 0.6963643 ,  0.6963642 , -0.17364815,  0.5820043 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ],
], dtype=np.float32)
CAM0_CANONICAL_INTRINSIC = np.array([
    [221.7025,   0.    , 128.    ],
    [  0.    , 221.7025, 128.    ],
    [  0.    ,   0.    ,   1.    ],
], dtype=np.float32)

CAM1_CANONICAL_EXTRINSIC = np.array([  # W2C
    [-0.7071068 , -0.70710677,  0.00000001,  0.49497476],
    [-0.12278779,  0.12278778, -0.98480785,  0.47987458],
    [ 0.6963642 , -0.6963643 , -0.17364815,  0.5820043 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ],
], dtype=np.float32)
CAM1_CANONICAL_INTRINSIC = np.array([
    [221.7025,   0.    , 128.    ],
    [  0.    , 221.7025, 128.    ],
    [  0.    ,   0.    ,   1.    ],
], dtype=np.float32)

CANONICAL_CAMERAS = {
    "cam0_canonical": (CAM0_CANONICAL_EXTRINSIC, CAM0_CANONICAL_INTRINSIC),
    "cam1_canonical": (CAM1_CANONICAL_EXTRINSIC, CAM1_CANONICAL_INTRINSIC),
}

SOURCE_CAMS = ["cam0", "cam1", "cam2"]


# ── Core functions ───────────────────────────────────────────────────────────

def unproject_to_world(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject depth+rgb from a single camera into world-frame points+colors.

    Args:
        depth: (H, W) float32 depth in meters.
        rgb: (H, W, 3) uint8 image.
        intrinsic: (3, 3) camera intrinsic matrix K.
        extrinsic_w2c: (4, 4) world-to-camera transform.

    Returns:
        points_world: (N, 3) valid 3D points in world frame.
        colors: (N, 3) uint8 colours for each point.
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)[None, :]  # (1, W)
    v = np.arange(h, dtype=np.float32)[:, None]  # (H, 1)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    points_cam = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # (H*W, 3)
    colors_flat = rgb.reshape(-1, 3)

    # Filter out invalid depth (zero or negative)
    valid = depth.reshape(-1) > 0
    points_cam = points_cam[valid]
    colors_flat = colors_flat[valid]

    # Camera-to-world: C2W = inv(W2C)
    c2w = np.linalg.inv(extrinsic_w2c)
    rot = c2w[:3, :3]
    trans = c2w[:3, 3]
    points_world = (rot @ points_cam.T).T + trans

    return points_world, colors_flat


def render_canonical_view(
    points_world: np.ndarray,
    colors: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Render a fused point cloud into a canonical camera view with z-buffering.

    Args:
        points_world: (N, 3) world-frame points.
        colors: (N, 3) uint8 colours.
        w2c: (4, 4) canonical world-to-camera extrinsic.
        intrinsic: (3, 3) canonical intrinsic matrix K.
        image_size: output image height and width (square).

    Returns:
        image: (H, W, 3) uint8 rendered image.
    """
    H = W = image_size

    # World → camera frame
    rot = w2c[:3, :3]
    trans = w2c[:3, 3]
    points_cam = (rot @ points_world.T).T + trans  # (N, 3)

    # Keep only points in front of the camera
    in_front = points_cam[:, 2] > 0
    points_cam = points_cam[in_front]
    colors_valid = colors[in_front]

    if len(points_cam) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Project to pixel coordinates
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    z = points_cam[:, 2]
    u = (fx * points_cam[:, 0] / z + cx).astype(np.int32)
    v = (fy * points_cam[:, 1] / z + cy).astype(np.int32)

    # Keep only points within image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    colors_valid = colors_valid[in_bounds]

    if len(u) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Z-buffer: keep closest point per pixel
    image = np.zeros((H, W, 3), dtype=np.uint8)
    z_buffer = np.full((H, W), np.inf, dtype=np.float32)

    # Sort by depth (farthest first so closest overwrites)
    order = np.argsort(-z)
    u = u[order]
    v = v[order]
    z = z[order]
    colors_valid = colors_valid[order]

    # Splat — painting farthest first means closest naturally wins
    image[v, u] = colors_valid
    z_buffer[v, u] = z

    return image


def process_timestep(
    f: h5py.File,
    t: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    """Process a single timestep: fuse point clouds, render canonical views.

    Returns:
        dict mapping canonical camera name to rendered uint8 image.
    """
    all_points = []
    all_colors = []

    for cam in SOURCE_CAMS:
        rgb = f[f"obs/{cam}_image"][t]  # (H, W, 3) uint8
        depth_raw = f[f"obs/{cam}_depth"][t]  # (H, W) uint16
        intrinsic = f[f"obs/{cam}_intrinsic"][t]  # (3, 3)
        extrinsic = f[f"obs/{cam}_extrinsic"][t]  # (4, 4) W2C

        # uint16 millimeters → float32 meters
        depth = depth_raw.astype(np.float32) / 1000.0

        pts, cols = unproject_to_world(depth, rgb, intrinsic, extrinsic)
        all_points.append(pts)
        all_colors.append(cols)

    points_world = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    results = {}
    for canon_name, (w2c, K) in CANONICAL_CAMERAS.items():
        rendered = render_canonical_view(points_world, colors, w2c, K, image_size)
        results[canon_name] = rendered

    return results


def process_h5_file(h5_path: Path, image_size: int) -> None:
    """Process a single h5 file: add canonical views for every timestep."""
    with h5py.File(h5_path, "a") as f:
        # Determine trajectory length from first camera
        T = f["obs/cam0_image"].shape[0]

        # Create or overwrite output datasets
        out_datasets: dict[str, h5py.Dataset] = {}
        for canon_name in CANONICAL_CAMERAS:
            key = f"obs/{canon_name}_rgb"
            if key in f:
                del f[key]
            out_datasets[canon_name] = f.create_dataset(
                key,
                shape=(T, image_size, image_size, 3),
                dtype=np.uint8,
            )

        for t in range(T):
            rendered = process_timestep(f, t, image_size)
            for canon_name, img in rendered.items():
                out_datasets[canon_name][t] = img


@app.command()
def main(
    data_dir: str = typer.Argument(..., help="Directory containing .h5 files"),
    image_size: int = typer.Option(256, help="Output image size (square)"),
) -> None:
    """Project fused point clouds into canonical camera views and save to h5."""
    data_path = Path(data_dir)
    h5_files = sorted(data_path.rglob("*.h5"))

    if not h5_files:
        typer.echo(f"No .h5 files found in {data_dir}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(h5_files)} h5 files in {data_dir}")
    typer.echo(f"Image size: {image_size}x{image_size}")
    typer.echo(f"Canonical cameras: {list(CANONICAL_CAMERAS.keys())}")

    for h5_path in tqdm(h5_files, desc="Processing h5 files"):
        process_h5_file(h5_path, image_size)

    typer.echo("Done!")


if __name__ == "__main__":
    app()
