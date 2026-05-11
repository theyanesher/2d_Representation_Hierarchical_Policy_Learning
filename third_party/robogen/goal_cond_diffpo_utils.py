import torch 
import numpy as np
from typing import Optional

def project_points(
    points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    return_2d: bool = True,
    normalization_factor = 1.0
) -> np.ndarray:
    B, N, _ = points.shape
    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None, ...]
    if extrinsic.ndim == 2:
        extrinsic = extrinsic[None, ...]
    ones = np.ones((B, N, 1), dtype=points.dtype)
    pts_h = np.concatenate([points, ones], axis=-1)
    cam_h = extrinsic @ pts_h.transpose(0, 2, 1)
    cam = cam_h[:, :3, :]
    pix_h = intrinsic @ cam
    pix_h[:, :2, :] /= pix_h[:, 2:3, :]
    if return_2d:
        pixels = pix_h[:, :2, :].transpose(0, 2, 1)
    else: 
        pixels = pix_h.transpose(0, 2, 1)
    pixels[...,:2] *= normalization_factor
    return pixels

def camera_frame_coords(
    points: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    B, N, _ = points.shape
    if extrinsic.ndim == 2:
        extrinsic = extrinsic[None, ...]
    ones = np.ones((B, N, 1), dtype=points.dtype)
    pts_h = np.concatenate([points, ones], axis=-1)
    cam_h = extrinsic @ pts_h.transpose(0, 2, 1)
    cam = cam_h[:, :3, :]
    return cam.transpose(0,2,1)

def coords_to_2d_image(
        coords: np.ndarray, # (N, M, 2)
        images: np.ndarray, # (N, H, W, C)
        clip_coords: bool = False, # clamp or drop keypoints
) -> np.ndarray:
    N, H, W, _ = images.shape
    M = coords.shape[1]

    # --- extract & clamp/drop keypoints ---
    xs = coords[..., 0].astype(np.float32)
    ys = coords[..., 1].astype(np.float32)
    if clip_coords:
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)
    xi = xs.astype(np.intp)
    yi = ys.astype(np.intp)

    if clip_coords:
        valid_pts = np.ones_like(xi, dtype=bool)
    else:
        valid_pts = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)

    mask = np.zeros((N, H, W), dtype=np.float32)
    batch_idx = np.repeat(np.arange(N, dtype=np.intp)[:, None], M, axis=1)
    bi = batch_idx[valid_pts]
    yv = yi[valid_pts]
    xv = xi[valid_pts]
    mask[bi, yv, xv] = 1.0
    return mask

def pytorch_camera_frame_coords(
    points: torch.Tensor, # (B, T, N, 3) or (B, N, 3)
    extrinsic: torch.Tensor,
) -> torch.Tensor:
    if len(points.shape) == 3:
        points = points.unsqueeze(1)
    B, T, N, _ = points.shape
    if extrinsic.ndim == 2:
        extrinsic = extrinsic[None, ...].expand(B, -1, -1)

    ones = torch.ones((B, T, N, 1), dtype=points.dtype, device=points.device)
    pts_h = torch.cat([points, ones], dim=-1)
    pts_h = pts_h.transpose(2, 3)

    cam_h = extrinsic.unsqueeze(1) @ pts_h
    cam = cam_h[..., :3, :].permute(0, 1, 3, 2)
    return cam

def pytorch_project_points(
    points: torch.Tensor, # (B, T, N, 3) or (B, N, 3)
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    return_2d: bool = True,
    normalization_factor: float = 1.0,
) -> torch.Tensor:
    if len(points.shape) == 3:
        points = points.unsqueeze(1)
    B, T, N, _ = points.shape
    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None, ...].expand(B, -1, -1)
    if extrinsic.ndim == 2:
        extrinsic = extrinsic[None, ...].expand(B, -1, -1)

    ones = torch.ones((B, T, N, 1), dtype=points.dtype, device=points.device)
    pts_h = torch.cat([points, ones], dim=-1)
    pts_h = pts_h.transpose(2, 3)

    cam_h = extrinsic.unsqueeze(1) @ pts_h
    cam = cam_h[..., :3, :]
    pix_h = intrinsic.unsqueeze(1) @ cam
    pix_h[..., :2, :] /= pix_h[..., 2:3, :] + 1e-8

    if return_2d:
        pixels = pix_h[..., :2, :].permute(0, 1, 3, 2)
    else:
        pixels = pix_h.permute(0, 1, 3, 2)
    pixels[...,:2] *= normalization_factor
    return pixels

def pytorch_coords_to_2d_image(
    coords: torch.Tensor, # (B, T, N, 2) or (B, N, 2), float
    images: torch.Tensor, # (B, T, C, H, W)
    clip_coords: bool = False, # clamp coords into [0..W-1]/[0..H-1]
) -> torch.Tensor:
    B, T, C, H, W = images.shape
    if coords.ndim == 3:
        coords = coords.unsqueeze(1)
        coords = coords.repeat(1, T, 1, 1)
    _, _, N, _      = coords.shape

    xs = coords[..., 0]  # (B, N)
    ys = coords[..., 1]  # (B, N)

    if clip_coords:
        xs = xs.clamp(0, W-1)
        ys = ys.clamp(0, H-1)

    xs_idx = xs.long()
    ys_idx = ys.long()

    if clip_coords:
        valid = torch.ones_like(xs_idx, dtype=torch.bool)
    else:
        valid = (
            (xs_idx >= 0) & (xs_idx < W) &
            (ys_idx >= 0) & (ys_idx < H)
        )

    mask = torch.zeros((B, T, H, W),
                       device=coords.device,
                       dtype=images.dtype)
    batch_idx = torch.arange(B, device=coords.device).reshape(B,1,1).expand(B,T,N)
    flat_valid = valid.reshape(-1)
    bi = batch_idx.reshape(-1)[flat_valid]
    xv = xs_idx.reshape(-1)[flat_valid]
    yv = ys_idx.reshape(-1)[flat_valid]
    M  = bi.shape[0]

    b_idx = bi[:, None].expand(M, T)
    t_idx = torch.arange(T, device=coords.device)[None, :].expand(M, T)
    y_idx = yv[:, None].expand(M, T)
    x_idx = xv[:, None].expand(M, T)
    mask[b_idx, t_idx, y_idx, x_idx] = 1.0

    mask = mask.unsqueeze(2)            # (B, T, 1, H, W)
    return mask

def coords_to_2d_image_displacements(
        goal_gripper_coords: np.ndarray, # (N, M, 2)
        images: np.ndarray, # (N, H, W, C)
        gripper_coords: np.ndarray = None, # (N, M, 2)
        pcd_displacements: Optional[np.ndarray] = None  # (N, M, D) if provided, will be used instead of displacements
) -> np.ndarray:
    N, H, W, C = images.shape
    N_, M, D_ = goal_gripper_coords.shape
    flow_mask = np.zeros((N, H, W, D_), dtype=np.float32)
    if pcd_displacements is not None:
        displacements = pcd_displacements
    else:
        displacements = goal_gripper_coords - gripper_coords
    displacements = displacements.reshape(-1,D_)
    xs = np.rint(gripper_coords[..., 0]).flatten().astype(np.intp)
    ys = np.rint(gripper_coords[..., 1]).flatten().astype(np.intp)

    valid_pts = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs = xs[valid_pts]
    ys = ys[valid_pts]

    batch_idx = np.repeat(np.arange(N, dtype=np.intp)[:, None], M, axis=1).flatten()
    batch_idx = batch_idx[valid_pts]
    for i in range(D_):
        flow_mask[batch_idx, ys, xs, i] = displacements[valid_pts,i].flatten()
    return flow_mask

def pytorch_coords_to_2d_image_displacements(
        goal_gripper_coords: torch.Tensor,  # (B, T, M, D)
        images: torch.Tensor,               # (B, T, C, H, W)
        gripper_coords: torch.Tensor,        # (B, T, M, D)
        return_images=False,
        pcd_displacements: Optional[np.ndarray] = None 
) -> torch.Tensor:
    B, T, C, H, W = images.shape
    _, _, M, D = goal_gripper_coords.shape
    device = images.device

    flow_mask = torch.zeros((B, T, D, H, W), device=device)

    if pcd_displacements is not None:
        disp = pcd_displacements
    else:
        disp = (goal_gripper_coords - gripper_coords)  # (B,T,M,D)

    xs = torch.round(gripper_coords[..., 0]).long()
    ys = torch.round(gripper_coords[..., 1]).long()

    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)  # (B,T,M)
    b_idx = torch.arange(B, device=device)[:, None, None].expand(B, T, M)  # (B,T,M)
    t_idx = torch.arange(T, device=device)[None, :, None].expand(B, T, M)  # (B,T,M)

    for c in range(D):
        bx = b_idx[valid]
        tx = t_idx[valid]
        yy = ys[valid]
        xx = xs[valid]
        dd = disp[..., c][valid]
        flow_mask[bx, tx, c, yy, xx] = dd
    if return_images:
        return torch.cat([images, flow_mask], dim=2)
    else:
        return flow_mask