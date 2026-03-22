import torch
import numpy as np

def to_torch(x, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    return x.float().to(device)

def generate_heatmap_from_points(points_3d, intrinsics, extrinsics, img_size, opengl2opencv=True):
    """
    Projects 3D points onto a 2D plane and generates a distance-based heatmap.

    Args:
        points_3d (np.ndarray or torch.Tensor): Shape (N, 3) or (B, N, 3).
        intrinsics (np.ndarray or torch.Tensor): Shape (3, 3) or (B, 3, 3).
        extrinsics (np.ndarray or torch.Tensor): Shape (4, 4) or (B, 4, 4).
        img_size (tuple): The target output size (height, width).
    Returns:
        heatmap (np.ndarray or torch.Tensor): The generated heatmap.
                                              Shape (B, N, H, W) or (N, H, W).
    """
    height, width = img_size
    is_numpy = isinstance(points_3d, np.ndarray)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not is_numpy else torch.device('cpu')

    points = to_torch(points_3d, device)
    intrinsics = to_torch(intrinsics, device)
    extrinsics = to_torch(extrinsics, device)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    B = points.shape[0]

    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).repeat(B, 1, 1)
    if extrinsics.ndim == 2:
        extrinsics = extrinsics.unsqueeze(0).repeat(B, 1, 1)

    if extrinsics.ndim == 4:
        extrinsics = extrinsics[:, -1]

    if opengl2opencv:
        opengl2opencv_mat = torch.diag(torch.tensor([1., -1., -1., 1.], device=device))
        extrinsics = torch.matmul(opengl2opencv_mat, extrinsics)

    ones = torch.ones((*points.shape[:-1], 1), device=device)
    points_hom = torch.cat([points, ones], dim=-1)
    cam_points = torch.matmul(points_hom, extrinsics.transpose(-1, -2))
    cam_points_3d = cam_points[..., :3]
    pix_points_hom = torch.matmul(cam_points_3d, intrinsics.transpose(-1, -2))
    depth = pix_points_hom[..., 2:3]
    uv = pix_points_hom[..., :2] / (depth)
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    pixel_coords = torch.stack([grid_x, grid_y], dim=-1).float()
    max_dist = (height**2 + width**2)**0.5

    pixels = pixel_coords.unsqueeze(0).unsqueeze(0)
    targets = uv.unsqueeze(-2).unsqueeze(-2)

    dist = torch.norm(pixels - targets, dim=-1)
    heatmap = torch.sqrt(dist / max_dist)
    heatmap = torch.clamp(heatmap, 0, 1)

    if is_numpy:
        heatmap_np = (heatmap * 255).clamp(0, 255).byte().cpu().numpy()
        if points_3d.ndim == 2:
            return heatmap_np[0]
        return heatmap_np
    return heatmap