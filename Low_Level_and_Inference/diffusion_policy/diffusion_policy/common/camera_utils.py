import numpy as np
import torch

# Adapted from https://ripl.github.io/know_your_camera/
def get_plucker_raymap(K, c2w, height, width, return_numpy=False, channel_first=True):
    is_batched = K.ndim == 3
    if not torch.is_tensor(K): K = torch.as_tensor(K)
    if not torch.is_tensor(c2w): c2w = torch.as_tensor(c2w)

    if not is_batched:
        K = K.unsqueeze(0)
        c2w = c2w.unsqueeze(0)
    B = K.shape[0]
    device, dtype = K.device, K.dtype
    vv, uu = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    rays_cam = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).view(1, -1, 3)
    rays_cam = rays_cam.expand(B, -1, -1)
    K_inv = torch.inverse(K)
    d_cam = torch.matmul(rays_cam, K_inv.transpose(-2, -1))
    R = c2w[:, :3, :3] # (B, 3, 3)
    t = c2w[:, :3, 3]  # (B, 3)
    
    d_world = torch.matmul(d_cam, R.transpose(-2, -1))
    d_world = torch.nn.functional.normalize(d_world, dim=-1, eps=1e-9)
    o = t.unsqueeze(1) 
    m = torch.cross(o, d_world, dim=-1)

    raymaps = torch.cat([d_world, m], dim=-1).view(B, height, width, 6)
    if channel_first:
        raymaps = raymaps.permute(0, 3, 1, 2).contiguous()
    if not is_batched:
        raymaps = raymaps.squeeze(0)
    if return_numpy:
        return raymaps.detach().cpu().numpy().astype(np.float32)
    return raymaps.to(torch.float32)


def get_pointmap(K, c2w, depth, return_numpy=False, channel_first=True):
    """Back-projects depth into a 3D pointmap (world coordinates).
    K: (3,3) or (B,3,3)
    c2w: (4,4) or (B,4,4)
    depth: (H,W) or (B,H,W) or (B,1,H,W)
    """
    is_batched = K.ndim == 3
    if not torch.is_tensor(K): K = torch.as_tensor(K)
    if not torch.is_tensor(c2w): c2w = torch.as_tensor(c2w)
    if not torch.is_tensor(depth): depth = torch.as_tensor(depth)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if not is_batched:
        K = K.unsqueeze(0)
        c2w = c2w.unsqueeze(0)
    B, _, H, W = depth.shape
    device, dtype = K.device, K.dtype
    vv, uu = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype) + 0.5,
        torch.arange(W, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([uu, vv, torch.ones_like(uu)], dim=0).unsqueeze(0)
    K_inv = torch.inverse(K)
    pixels_flat = pixels.view(1, 3, -1)
    rays_cam = torch.matmul(K_inv, pixels_flat)
    points_cam = rays_cam.view(B, 3, H, W) * depth
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3].view(B, 3, 1, 1)
    points_cam_flat = points_cam.view(B, 3, -1)
    points_world_flat = torch.matmul(R, points_cam_flat)
    pointmap = points_world_flat.view(B, 3, H, W) + t
    if not channel_first:
        pointmap = pointmap.permute(0, 2, 3, 1).contiguous()
    if not is_batched:
        pointmap = pointmap.squeeze(0)
    if return_numpy:
        return pointmap.detach().cpu().numpy().astype(np.float32)
    return pointmap.to(torch.float32)