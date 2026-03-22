#!/usr/bin/env python3
"""
Utility helpers for visualizing Genesis datasets.

Includes a small CLI to unproject RGB + depth images (plus extrinsics) into a 3D point cloud.
"""

from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING, Union
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    import open3d as o3d

def _apply_extrinsics(points: np.ndarray, world2cam: Optional[np.ndarray]) -> np.ndarray:
    if world2cam is None:
        return points
    if world2cam.shape != (4, 4):
        raise ValueError("Extrinsics must be a 4x4 matrix.")
    cam2world = np.linalg.inv(world2cam)
    rot = cam2world[:3, :3]
    trans = cam2world[:3, 3]
    return (rot @ points.T).T + trans


def unproject_depth_rgb(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    world2cam: Optional[np.ndarray] = None,
    sample_ratio: float = 1.0,
    return_open3d: bool = False,
    mask: Optional[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, Optional[np.ndarray], "o3d.geometry.PointCloud"],
]:
    h, w = depth.shape
    u = np.arange(w)[None, :]
    v = np.arange(h)[:, None]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    points = _apply_extrinsics(points, world2cam)
    colors = None if rgb is None else rgb.reshape(-1, rgb.shape[-1])

    if mask is not None:
        mask_flat = mask.reshape(-1)
        points = points[mask_flat]
        if colors is not None:
            colors = colors[mask_flat]

    if sample_ratio < 1.0 and points.shape[0] > 0:
        keep = np.random.rand(points.shape[0]) < sample_ratio
        points = points[keep]
        if colors is not None:
            colors = colors[keep]

    if return_open3d:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return pcd

    return points, colors
