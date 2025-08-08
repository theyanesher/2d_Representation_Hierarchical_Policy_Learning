import torch
from torch import nn
from torchvision.ops import MLP

from ptv3.model import Point, PointTransformerV3
import torch.nn.functional as F

def farthest_point_sample(xyz_, npoint, keep_gripper_in_fps=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if keep_gripper_in_fps: ### NOTE: assuming there are 4 gripper points
        xyz = xyz_[:, :-4, :]
        npoint = npoint - 4
    else:
        xyz = xyz_
    
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = farthest * 0 # set to 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    if keep_gripper_in_fps:
        gripper_indices = torch.Tensor([N, N+1, N+2, N+3]).long().to(device)
        gripper_indices = gripper_indices.unsqueeze(0).repeat(B, 1)
        centroids = torch.cat([centroids, gripper_indices], dim=1)
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class HighlevelPTv3(nn.Module):
    def __init__(self, ptv3_config, weight_head_mlp_config, offset_head_mlp_config, grid_size, fps_num=2048):
        super().__init__()
        self.grid_size = grid_size
        self.ptv3 = PointTransformerV3(**ptv3_config)
        self.mlp_head_weight = MLP(in_channels=self.ptv3.get_out_channels(), **weight_head_mlp_config)
        self.mlp_head_offset = MLP(in_channels=self.ptv3.get_out_channels(), **offset_head_mlp_config)
        self.fps_num = fps_num

    def forward(self, x, embedding=None, build_grasp=False):
        B, N, C = x.shape
        # form data_dict
        assert embedding is not None
        if embedding is not None: ### language as input
            # import pdb; pdb.set_trace()
            embedding = embedding.unsqueeze(1).repeat(1, N, 1)
            x = torch.cat([x, embedding], dim=2)
            B, N, C = x.shape
        offset = torch.arange(1, B + 1) * N
        data_dict = {
            "feat": x.reshape(-1, C),
            "coord": x[..., :3].reshape(-1, 3),
            "grid_size": self.grid_size,
            "offset": offset.to(x.device),
        }
        
        point = self.ptv3.forward(data_dict)
        # import pdb; pdb.set_trace()
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        # import pdb; pdb.set_trace()
        feat = feat.view(B, N, -1)
        positions = point.coord.view(B, N, 3)
        fps_indices = farthest_point_sample(positions, self.fps_num)
        pred_points = index_points(positions, fps_indices)
        feat = index_points(feat, fps_indices)
            
        weights = self.mlp_head_weight(feat)
        pred_scores = weights
        pred_offsets = self.mlp_head_offset(feat)
        pred_offsets = pred_offsets.view(B, self.fps_num, 4, 3)

        if build_grasp:
            pred_4_points = pred_points.unsqueeze(2).repeat(1, 1, 4, 1) + pred_offsets
            pred_grasps_cam, offset = self.build_6d_grasp_from_four_points(pred_4_points)  # B x N x 4 x 4
        else:
            pred_grasps_cam, offset = None, None
    
        pred = dict(
            pred_scores = pred_scores,
            pred_points =pred_points,
            pred_offsets=pred_offsets,  
            pred_grasps_cam= pred_grasps_cam,  # B x N x 4 x 4
            offset_pred=offset
        )
        
        return pred
    
    def build_6d_grasp_from_four_points(self, four_point_head, gripper_depth = 0.1034):
        B, N, _, _ = four_point_head.shape
        grasp_t = four_point_head[:, :, 0].unsqueeze(3)  # B x N x 3 x 1
        
        
        approach_direction = four_point_head[:, :, -1] - four_point_head[:, :, 0]  # B x N x 3
        baseline_direction = four_point_head[:, :, 2] - four_point_head[:, :, 1]  # B x N x 3
        
        # baseline_direction_normed = F.normalize(baseline_direction, p=2, dim=2)  # B x N x 3
        # dot_product = torch.sum(approach_direction * baseline_direction_normed, dim=2, keepdim=True)  # B x N x 1
        # projection = dot_product * baseline_direction_normed  # B x N x 3
        # approach_direction_orthog = F.normalize(approach_direction - projection, p=2, dim=2)  # B x N x 3
        # grasp_R = torch.stack([baseline_direction_normed, torch.cross(approach_direction_orthog, baseline_direction_normed),approach_direction_orthog], dim=3)  # B x N x 3 x 3
        
        approach_direction_normed = F.normalize(approach_direction, p=2, dim=2)  # B x N x 3
        dot_product = torch.sum(baseline_direction * approach_direction_normed, dim=2, keepdim=True)  # B x N x 1
        projection = dot_product * approach_direction_normed  # B x N x 3
        baseline_direction_orthog = F.normalize(baseline_direction - projection, p=2, dim=2)  # B x N x 3
        grasp_R = torch.stack([baseline_direction_orthog, torch.cross(approach_direction_normed, baseline_direction_orthog),approach_direction_normed], dim=3)  # B x N x 3 x 3
        
        ones = torch.ones((B, N, 1, 1), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 1
        zeros = torch.zeros((B, N, 1, 3), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 3
        homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
        grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4
        
        offset = torch.norm(four_point_head[:, :, 2] - four_point_head[:, :, 1], dim=-1, keepdim=True)  # B x N x 1
        
        return grasps, offset
