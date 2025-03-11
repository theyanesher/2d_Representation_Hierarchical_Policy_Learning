# NOTE:
# Trying to implement PointNet++
# Borrowed from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from diffusion_policy_3d.model.diffusion.transformers.original_conditional_transformer import FilmConditionalResidualBlock, FilmConditionalResidualBlockSmall
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, keep_gripper_in_fps=False):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.keep_gripper_in_fps = keep_gripper_in_fps
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S, self.keep_gripper_in_fps))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                # import pdb; pdb.set_trace()
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=3, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=0, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 256) (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 64) (B, 512, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 16) (B, 1024, 16)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 512, 64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes
    

class PointNet2_small2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_small2, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=0, mlp_list=[[16, 16, 16], [32, 32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=48, mlp_list=[[64, 64, 64], [64, 96, 64]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128, [[128, 196, 128], [128, 196, 128]])

        self.fp3 = PointNetFeaturePropagation(64+64+128+128, [128, 128])
        self.fp2 = PointNetFeaturePropagation(16+32+128, [64, 64])
        self.fp1 = PointNetFeaturePropagation(64, [64, 64, 64])
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 512) (B, 96, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 128) (B, 256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 32) (B, 512, 32)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 128)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 512)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes: outputing logtis

class PointNet2_super(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, cross_attn_bottleneck=False, 
                 attn_embedding_dim=60, attn_num_heads=3, attn_num_layers=2, demo_use_attn=True, demo_pn_type='large', demo_use_cur_obs=True, 
                 use_flow_in_demo=False, separate_demo_feature=False, use_hadamard_production=False, 
                  cross_attn_every_layer=False, bottleneck_film_cond = False,
                 always_train_with_conditioning=False, aligned_cross_attn=False,
                 condition_set_to_false=False, 
                 just_use_pn=False,
                 condition_prob=0.5,
                 small_film=False,
                 ):
        super(PointNet2_super, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel - 3, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512+512+256+256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.cross_attn_bottleneck = cross_attn_bottleneck
        self.cross_attn_every_layer = cross_attn_every_layer
        self.bottleneck_film_cond = bottleneck_film_cond
        self.just_use_pn = just_use_pn
        self.condition_set_to_false = condition_set_to_false
        self.always_train_with_conditioning = always_train_with_conditioning
        self.condition_prob = condition_prob
        
        if not just_use_pn:
        
            if self.cross_attn_bottleneck:
                self.cross_attention_layers = CrossAttentionModule(attn_embedding_dim, attn_num_heads, attn_num_layers)
                self.linear_down_l6 = nn.Linear(1024, attn_embedding_dim)
                self.linear_up_l6 = nn.Linear(attn_embedding_dim, 1024)
            elif self.bottleneck_film_cond:
                film_class = FilmConditionalResidualBlock if not small_film else FilmConditionalResidualBlockSmall
                if separate_demo_feature:
                    self.bottleneck_film_cond_layer = film_class(1024, 1024, attn_embedding_dim*2)
                else:
                    self.bottleneck_film_cond_layer = film_class(1024, 1024, attn_embedding_dim)
                self.linear_film_bottleneck_l6 = nn.Linear(1024, 1024)
                #self.linear_up_l6 = nn.Linear(attn_embedding_dim, 1024)
            if self.cross_attn_every_layer: # or True:
                self.linear_down_l5 = nn.Linear(512, attn_embedding_dim)
                self.linear_up_l5 = nn.Linear(attn_embedding_dim, 512)

                self.linear_down_l4 = nn.Linear(512, attn_embedding_dim)
                self.linear_up_l4 = nn.Linear(attn_embedding_dim, 512)

                self.linear_down_l3 = nn.Linear(256, attn_embedding_dim)
                self.linear_up_l3 = nn.Linear(attn_embedding_dim, 256)

                self.linear_down_l2 = nn.Linear(256, attn_embedding_dim)
                self.linear_up_l2 = nn.Linear(attn_embedding_dim, 256)

                self.linear_down_l1 = nn.Linear(128, attn_embedding_dim)
                self.linear_up_l1 = nn.Linear(attn_embedding_dim, 128)

                self.linear_down_l0 = nn.Linear(128, attn_embedding_dim)
                self.linear_up_l0 = nn.Linear(attn_embedding_dim, 128)

            self.use_hadamard_production = use_hadamard_production
            if self.use_hadamard_production:
                input_dim = attn_embedding_dim
                if separate_demo_feature:
                    input_dim = attn_embedding_dim * 2
                self.bottleneck_fc = nn.Linear(input_dim, 1024)
                self.fp6_fc = nn.Linear(input_dim, 512)
                self.fp5_fc = nn.Linear(input_dim, 512)
                self.fp4_fc = nn.Linear(input_dim, 256)
                self.fp3_fc = nn.Linear(input_dim, 256)
                self.fp2_fc = nn.Linear(input_dim, 128)
                self.fp1_fc = nn.Linear(input_dim, 128)
                
            pn_fc_layers = [128, 64] if attn_embedding_dim < 255 else [256, 256]
            if aligned_cross_attn:
                demo_pn_type = "large_return_sa"
            self.demo_transformer = Demo_processing_model(
                pn_input_channel=2 if not use_flow_in_demo else 12, 
                attn_embedding_dim=attn_embedding_dim,
                use_attn=demo_use_attn,
                use_cur_obs=demo_use_cur_obs,
                pn_type=demo_pn_type,
                use_flow_in_demo=use_flow_in_demo,
                separate_demo_feature=separate_demo_feature,
                pn_fc_layers=pn_fc_layers,
            )
        
            self.attn_embedding_dim = attn_embedding_dim
            self.separate_demo_feature = separate_demo_feature
            self.aligned_cross_attn = aligned_cross_attn
            
            if aligned_cross_attn:
                self.rotary_attn_layers = RelativeCrossAttentionModule(attn_embedding_dim, attn_num_heads, attn_num_layers)
                self.rotary_attn_layers = replace_bn_with_gn(self.rotary_attn_layers)
                self.rotary_attn_pos_enc = RotaryPositionEncoding3D(attn_embedding_dim)
                
                self.rotary_linear_down_sa5 = nn.Linear(1024, attn_embedding_dim)
                self.rotary_linear_down_sa6 = nn.Linear(1024, attn_embedding_dim)
                self.rotary_linear_up_sa5 = nn.Linear(attn_embedding_dim, 1024)
                self.rotary_linear_up_sa6 = nn.Linear(attn_embedding_dim, 1024)
                
                self.rotary_linear_down_fp5 = nn.Linear(512, attn_embedding_dim)
                self.rotary_linear_up_fp5 = nn.Linear(attn_embedding_dim, 512)
                
                self.l3_linear = nn.Linear(512, attn_embedding_dim)
                self.l4_linear = nn.Linear(512, attn_embedding_dim)
            
            
    def hadamard_production(self, fp_feature, condition_feature, linear_layer):
        condition_feature = linear_layer(condition_feature) # B, 1, attn_embedding_dim -> B, 1, fp_feature_dim
        ### needs to repeat condition feature for each point in fp_feature
        num_points = fp_feature.shape[1]
        condition_feature = condition_feature.repeat(1, num_points, 1) # B, num_points, fp_feature_dim
        update_feature = fp_feature * condition_feature
        return update_feature.permute(0, 2, 1) # B, fp_feature_dim, num_points
    
    def rotary_cross_attn(self, cond_xyz, cond_points, cur_xyz, cur_points, linear_down_cond, linear_down_cur, linear_up_cur):
        cond_xyz_embedding = self.rotary_attn_pos_enc(cond_xyz.permute(0, 2, 1)) # shape B 64 attn_embedding_dim
        cur_xyz_embedding = self.rotary_attn_pos_enc(cur_xyz.permute(0, 2, 1))
        cur_points_down = linear_down_cur(cur_points.permute(0, 2, 1)) # B, 64, attn_embedding_dim
        cond_points_down = linear_down_cond(cond_points.permute(0, 2, 1)) # B, 64, attn_embedding_dim
        # import pdb; pdb.set_trace()
        attn_output = self.rotary_attn_layers(
            query=cur_points_down.permute(1, 0, 2), value=cond_points_down.permute(1, 0, 2),
            query_pos=cur_xyz_embedding, value_pos=cond_xyz_embedding,
        )[-1] # L, B, C
        
        # import pdb; pdb.set_trace()
        cur_points = cur_points + linear_up_cur(attn_output.permute(1, 0, 2)).permute(0, 2, 1)
        return cur_points
        # return linear_up_cur(attn_output.permute(1, 0, 2)).permute(0, 2, 1)
        
    def normal_attention(self, cur_points, linear_down, linear_up, demo_conditioning_feature, demo_conditioning_feature_1, demo_conditioning_feature_2):
        if not self.separate_demo_feature:
            cur_points_attn = self.cross_attention_layers(linear_down(cur_points.permute(0, 2, 1)), 
                                                demo_conditioning_feature, demo_conditioning_feature)
        else:
            #import pdb; pdb.set_trace();
            cur_points_attn_features = linear_down(cur_points.permute(0, 2, 1)) # B, 16, attn_embedding_dim
            query = torch.cat([cur_points_attn_features, demo_conditioning_feature_1, demo_conditioning_feature_2], dim=1)
            cur_points_attn = self.cross_attention_layers(query, query, query) # self attention actually # B, 16, attn_embedding_dim
            cur_points_attn = cur_points_attn[:, :cur_points.shape[2], :]
            
        cur_points = F.relu(cur_points + linear_up(cur_points_attn).permute(0, 2, 1))
        return cur_points

    def forward_just_pn(self, xyz, demo_data):
        if (np.random.rand() > 0.5 and demo_data is not None) or self.always_train_with_conditioning: ### train with conditioning and without conditioning randomly
            B, _, N = xyz.shape
            n_points = N - 4
            assert n_points == 4500
            data_dict = demo_data
            demo_grasp_points, demo_grasp_goal = data_dict['demo_grasp_pcd'], data_dict['demo_grasp_goal_gripper_pcd'] # first frame
            demo_open_points, demo_open_goal = data_dict['demo_grasp_pcd'], data_dict['demo_open_gripper_pcd']
            
            # import pdb; pdb.set_trace()
            all_points = torch.cat([xyz.permute(0, 2, 1), demo_grasp_points, demo_grasp_goal, demo_open_goal], dim=1).permute(0, 2, 1)
            B, _, N = all_points.shape
            assert N == n_points * 2 + 3 * 4
            features = torch.zeros((B, N, 5), device=xyz.device)
            features[:, :n_points, 0] = 1 # cur object pcd
            features[:, n_points:n_points + 4, 1] = 1 # cur gripper pcd
            features[:, n_points+4:2*n_points + 4, 2] = 1 # demo object pcd
            features[:, 2*n_points + 4:2*n_points + 8, 3] = 1 # demo grasp gripper pcd
            features[:, 2*n_points + 8:2*n_points + 12, 4] = 1 # demo open gripper pcd
            features = features.permute(0, 2, 1)

        else:
            B, _, N = xyz.shape
            n_points = N - 4
            all_points = xyz
            features = torch.zeros((B, N, 5), device=xyz.device)
            features[:, :-4, 0] = 1
            features[:, -4:, 1] = 1
            features = features.permute(0, 2, 1)
            
        # import pdb; pdb.set_trace()
        l0_xyz = all_points
        l1_xyz, l1_points = self.sa1(all_points, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)
        
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x[:, :n_points+4, :] # x shape: B, N, num_classes

    def forward(self, xyz, demo_data):
        if self.just_use_pn:
            return self.forward_just_pn(xyz, demo_data)
        
        ### do demonstration conditioning processing here
        # print("always use demo: ", self.always_train_with_conditioning)
        use_condition = True
        if (np.random.rand() < self.condition_prob and demo_data is not None) or self.always_train_with_conditioning: ### train with conditioning and without conditioning randomly
            # import pdb; pdb.set_trace()
            demo_conditioning_feature = self.demo_transformer(demo_data)
            B, N, _ = xyz.shape
            if self.aligned_cross_attn:
                cond_l4_xyz, cond_l4_points, cond_l3_xyz, cond_l3_points = demo_conditioning_feature
            elif self.cross_attn_bottleneck or self.use_hadamard_production or self.bottleneck_film_cond:
                # import pdb; pdb.set_trace()
                if not self.separate_demo_feature:
                    demo_conditioning_feature = demo_conditioning_feature.unsqueeze(1)
                else:
                    demo_conditioning_feature_1, demo_conditioning_feature_2 = demo_conditioning_feature
                    demo_conditioning_feature_1 = demo_conditioning_feature_1.unsqueeze(1)
                    demo_conditioning_feature_2 = demo_conditioning_feature_2.unsqueeze(1)
                    # import pdb; pdb.set_trace()
            else:
                demo_conditioning_feature = demo_conditioning_feature.unsqueeze(1).expand(B, N, demo_conditioning_feature.shape[1])
        else:
            if self.condition_set_to_false:
                # print("setting use_condition to be false!")
                use_condition = False
            else:
                use_condition = True
                B, N, _ = xyz.shape
                if self.aligned_cross_attn:
                    cond_l3_xyz, cond_l3_points = torch.zeros((B, 3, 128), dtype=xyz.dtype, device=xyz.device), torch.zeros((B, 512, 128), dtype=xyz.dtype, device=xyz.device) # (B, 3, 64) (B, 512, 64)
                    cond_l4_xyz, cond_l4_points = torch.zeros((B, 3, 32), dtype=xyz.dtype, device=xyz.device), torch.zeros((B, 512, 32), dtype=xyz.dtype, device=xyz.device) # (B, 3, 16) (B, 512, 16)
                
                if self.cross_attn_bottleneck or self.use_hadamard_production or self.bottleneck_film_cond:
                    if not self.separate_demo_feature:
                        demo_conditioning_feature = torch.zeros((B, 1, self.attn_embedding_dim), dtype=xyz.dtype, device=xyz.device)
                    else:
                        demo_conditioning_feature_1, demo_conditioning_feature_2 = torch.zeros((B, 1, self.attn_embedding_dim), dtype=xyz.dtype, device=xyz.device), torch.zeros((B, 1, self.attn_embedding_dim), dtype=xyz.dtype, device=xyz.device)
                else:
                    demo_conditioning_feature = torch.zeros((B, N, self.attn_embedding_dim), dtype=xyz.dtype, device=xyz.device)
        
        l0_xyz = xyz[:, :3, :]
        
        if self.cross_attn_bottleneck or self.use_hadamard_production or self.aligned_cross_attn or self.bottleneck_film_cond:
            feature = xyz[:, 3:, :] if xyz.shape[1] > 3 else None
            l1_xyz, l1_points = self.sa1(l0_xyz, feature)
        else:
            if use_condition:
                feature = torch.cat([xyz[:, 3:, :], condition_feature], dim=1) if xyz.shape[1] > 3 else condition_feature
            else:
                feature = xyz[:, 3:, :] if xyz.shape[1] > 3 else None
                
            l1_xyz, l1_points = self.sa1(l0_xyz, feature)
    
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        if self.aligned_cross_attn and use_condition:
            l5_points = self.rotary_cross_attn(cond_l3_xyz, cond_l3_points, l5_xyz, l5_points, self.l3_linear, self.rotary_linear_down_sa5, self.rotary_linear_up_sa5)
        
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)
        if self.aligned_cross_attn and use_condition:
            l6_points = self.rotary_cross_attn(cond_l4_xyz, cond_l4_points, l6_xyz, l6_points, self.l4_linear, self.rotary_linear_down_sa6, self.rotary_linear_up_sa6)
        if self.cross_attn_bottleneck and use_condition:
            l6_points = self.normal_attention(l6_points, self.linear_down_l6, self.linar_up_l6, demo_conditioning_feature, demo_conditioning_feature_1, demo_conditioning_feature_2)
        if self.bottleneck_film_cond and use_condition:
            if self.separate_demo_feature:
                cond = torch.cat([demo_conditioning_feature_1.squeeze(1), demo_conditioning_feature_2.squeeze(1)], dim=1)
            else:
                cond = demo_conditioning_feature.squeeze(1)
            l6_points_film = self.bottleneck_film_cond_layer(l6_points,cond)
            l6_points = F.relu(l6_points + self.linear_film_bottleneck_l6(l6_points_film.permute(0, 2, 1)).permute(0, 2, 1)) 
        if self.use_hadamard_production and use_condition:
            if self.separate_demo_feature:
                demo_conditioning_feature = torch.cat([demo_conditioning_feature_1, demo_conditioning_feature_2], dim=-1)
            l6_points = self.hadamard_production(l6_points.permute(0, 2, 1), demo_conditioning_feature, self.bottleneck_fc)
            
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        if self.aligned_cross_attn and use_condition:
            l5_points = self.rotary_cross_attn(cond_l3_xyz, cond_l3_points, l5_xyz, l5_points, self.l3_linear, self.rotary_linear_down_fp5, self.rotary_linear_up_fp5)
        if self.cross_attn_every_layer and use_condition:
            l5_points = self.normal_attention(l5_points, self.linear_down_l5, self.linar_up_l5, demo_conditioning_feature, demo_conditioning_feature_1, demo_conditioning_feature_2)
        if self.use_hadamard_production and use_condition:
            l5_points = self.hadamard_production(l5_points.permute(0, 2, 1), demo_conditioning_feature, self.fp6_fc)

        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        if self.cross_attn_every_layer and use_condition:
            l4_points = self.normal_attention(l4_points, self.linear_down_l4, self.linar_up_l4, demo_conditioning_feature, demo_conditioning_feature_1, demo_conditioning_feature_2)
        if self.use_hadamard_production and use_condition:
            l4_points = self.hadamard_production(l4_points.permute(0, 2, 1), demo_conditioning_feature, self.fp5_fc)
        
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        if self.cross_attn_every_layer and use_condition:
            l3_points = self.normal_attention(l3_points, self.linear_down_l3, self.linar_up_l3, demo_conditioning_feature, demo_conditioning_feature_1, demo_conditioning_feature_2)
        if self.use_hadamard_production and use_condition:
            l3_points = self.hadamard_production(l3_points.permute(0, 2, 1), demo_conditioning_feature, self.fp4_fc)
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        if self.cross_attn_every_layer and use_condition:
            l2_points = self.normal_attention(l2_points, self.linear_down_l2, self.linar_up_l2, demo_conditioning_feature, demo_conditioning_feature_1, demo_conditioning_feature_2)
        if self.use_hadamard_production and use_condition:
            l2_points = self.hadamard_production(l2_points.permute(0, 2, 1), demo_conditioning_feature, self.fp3_fc)
        
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        if self.cross_attn_every_layer and use_condition:
            l1_points = self.normal_attention(l1_points, self.linear_down_l1, self.linar_up_l1, demo_conditioning_feature, demo_conditioning_feature_1, demo_conditioning_feature_2)
        if self.use_hadamard_production and use_condition:
            l1_points = self.hadamard_production(l1_points.permute(0, 2, 1), demo_conditioning_feature, self.fp2_fc)
                
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) # (B, 128, num_point)
        if self.use_hadamard_production and use_condition:
            l0_points = self.hadamard_production(l0_points.permute(0, 2, 1), demo_conditioning_feature, self.fp1_fc)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes

        
class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query):
        attn_output, attn_output_weights = self.multihead_attn(
            query=query,
            key=query,
            value=query,
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights.mean(dim=1)
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights.mean(dim=1)
        
class FeedforwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0, use_adaln = False):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output
        
class SelfAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, num_layers, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embedding_dim

        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(SelfAttentionLayer(embedding_dim, num_attn_heads))
            self.ffw_layers.append(FeedforwardLayer(embedding_dim, hidden_dim))

    def forward(self, query):
        output = []
        for i in range(len(self.attn_layers)):
            query, _ = self.attn_layers[i](query)
            query = self.ffw_layers[i](query)
            output.append(query)
        return output[-1]
    
class CrossAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, num_layers, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embedding_dim

        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(CrossAttentionLayer(embedding_dim, num_attn_heads))
            self.ffw_layers.append(FeedforwardLayer(embedding_dim, hidden_dim))

    def forward(self, query, key, value):
        output = []
        for i in range(len(self.attn_layers)):
            query, _ = self.attn_layers[i](query, key, value)
            query = self.ffw_layers[i](query)
            output.append(query)
        return output[-1]

class Demo_processing_model(nn.Module):
    def __init__(self, 
                 pn_input_channel=2,
                 pn_output_channel=60,
                 pn_keep_gripper_in_fps=False,
                 pn_type='large',
                 pn_fc_layers=[128, 64],
                 attn_embedding_dim=60,
                 attn_num_heads=3,
                 attn_num_layers=2,
                 use_attn=True,
                 use_cur_obs=True,
                 use_flow_in_demo=False,
                 separate_demo_feature=False,
                ):
        super(Demo_processing_model, self).__init__()
        
        if pn_type == 'super':
            self.pointnet_encoder = PointNet2_super_no_feature_prop(num_classes=attn_embedding_dim, input_channel=pn_input_channel, keep_gripper_in_fps=False)
        elif pn_type == 'large':
            self.pointnet_encoder = PointNet2_no_feature_prop(num_classes=attn_embedding_dim, input_channel=pn_input_channel, fc_layers=pn_fc_layers, keep_gripper_in_fps=False)
        elif pn_type == 'small':
            self.pointnet_encoder = PointNet2_small2_no_feature_prop(num_classes=attn_embedding_dim, input_channel=pn_input_channel, keep_gripper_in_fps=False)
        elif pn_type == 'large_return_sa':
            self.pointnet_encoder = PointNet2_no_feature_prop_2(num_classes=attn_embedding_dim, input_channel=pn_input_channel, fc_layers=pn_fc_layers, keep_gripper_in_fps=False)
            
        self.use_attn = use_attn
        self.use_cur_obs = use_cur_obs
        if self.use_attn:
            self.self_attn_layers = SelfAttentionModule(attn_embedding_dim, attn_num_heads, attn_num_layers)
        else:
            if not separate_demo_feature:
                self.linear = nn.Linear(2 * attn_embedding_dim, attn_embedding_dim)
            
        self.use_flow_in_demo = use_flow_in_demo
        self.separate_demo_feature = separate_demo_feature
        self.pn_type = pn_type
        
    def construct_pn_input(self, pcd, gripper_pcd):
        if not self.use_flow_in_demo:
            # first concat all points together
            B, N, _ = pcd.shape
            # import pdb; pdb.set_trace()
            all_points_xyz = torch.cat([pcd, gripper_pcd], dim=1) # B, (N+4), 3
            features = torch.zeros((B, N + 4, 2), device=all_points_xyz.device)
            features[:, :N, 0] = 1 # object points
            features[:, N:N+4, 1] = 1 # cur gripper points
            return all_points_xyz.permute(0, 2, 1), features.permute(0, 2, 1)
        else:
            B, N, _ = pcd.shape
            labels = gripper_pcd.unsqueeze(1) - pcd.unsqueeze(2) # B, N, 4, 3
            labels = labels.view(B, N, -1) # B, N, 12
            return pcd.permute(0, 2, 1), labels.permute(0, 2, 1)
            
    
    def forward(self, data_dict):
        if self.use_cur_obs:
            cur_obs_points, cur_obs_features = self.construct_pn_input(data_dict['pointcloud'], data_dict['gripper_pcd'])

        demo_grasp_points, demo_grasp_features = self.construct_pn_input(data_dict['demo_grasp_pcd'], data_dict['demo_grasp_goal_gripper_pcd']) # first frame
        # demo_open_points, demo_open_features = self.construct_pn_input(data_dict['demo_open_pcd'], data_dict['demo_open_gripper_pcd']) # -10 frame
        demo_open_points, demo_open_features = self.construct_pn_input(data_dict['demo_grasp_pcd'], data_dict['demo_open_gripper_pcd']) # -10 frame

        if self.use_cur_obs:
            cur_obs_embedding = self.pointnet_encoder(cur_obs_points, cur_obs_features) # B, pn_output_channel
        
        demo_grasp_embedding = self.pointnet_encoder(demo_grasp_points, demo_grasp_features)
        demo_open_embedding = self.pointnet_encoder(demo_open_points, demo_open_features)
        
        if self.use_attn:
            query = torch.stack([cur_obs_embedding, demo_grasp_embedding, demo_open_embedding], dim=1) # B, 3, pn_output_channel
            attn_output = self.self_attn_layers(query) # B, 3, attn_embedding
        
            return attn_output[:, 0, :] # use updated cur_obs_embedding
        
        else:
            if 'return_sa' in self.pn_type:
                grasp_l3_xyz, grasp_l3_points, grasp_l4_xyz, grasp_l4_points = demo_grasp_embedding
                open_l3_xyz, open_l3_points, open_l4_xyz, open_l4_points = demo_open_embedding
                l3_xyz = torch.cat([grasp_l3_xyz, open_l3_xyz], dim=-1)
                l4_xyz = torch.cat([grasp_l4_xyz, open_l4_xyz], dim=-1)
                l3_points = torch.cat([grasp_l3_points, open_l3_points], dim=-1)
                l4_points = torch.cat([grasp_l4_points, open_l4_points], dim=-1)
                return l3_xyz, l3_points, l4_xyz, l4_points
            else:
                if not self.separate_demo_feature:
                    concat_grasp_and_open = torch.cat([demo_grasp_embedding, demo_open_embedding], dim=-1)
                    return self.linear(concat_grasp_and_open)
                else:
                    return demo_grasp_embedding, demo_open_embedding
    
class PointNet2_super_no_feature_prop(nn.Module):
    def __init__(self, num_classes=128, input_channel=3, keep_gripper_in_fps=False):
        super(PointNet2_super_no_feature_prop, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps)

        self.sa_final = PointNetSetAbstraction(None, None, None, 1024 + 3, [512, 512, 512], True)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, xyz, feature):
        l0_points = feature
        l0_xyz = xyz[:, :3, :]
        B, _, _ = xyz.shape
        
        # import pdb; pdb.set_trace()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)
        final_xyz, final_points = self.sa_final(l6_xyz, l6_points)

        x = final_points.view(B, 512)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x # x shape: B, num_classes
    
class PointNet2_no_feature_prop_2(nn.Module):
    def __init__(self, num_classes=60, input_channel=3, keep_gripper_in_fps=False, fc_layers=[128, 64]):
        super(PointNet2_no_feature_prop_2, self).__init__()
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=3, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=input_channel, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 256], [256, 256, 256]])
        
    def forward(self, xyz, feature):
        l0_points = feature
        l0_xyz = xyz[:, :3, :]
        B, _, _ = xyz.shape
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 256) (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 64) (B, 512, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 16) (B, 512, 16)

        return l4_xyz, l4_points, l3_xyz, l3_points 
    
class PointNet2_no_feature_prop(nn.Module):
    def __init__(self, num_classes=60, input_channel=3, keep_gripper_in_fps=False, fc_layers=[128, 64]):
        super(PointNet2_no_feature_prop, self).__init__()
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=3, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=input_channel, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 256], [256, 256, 256]])
        self.sa_final = PointNetSetAbstraction(None, None, None, 512 + 3, [256, 256, 256], True)
        
        self.fc1 = nn.Linear(256, fc_layers[0])
        self.bn1 = nn.BatchNorm1d(fc_layers[0])
        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.bn2 = nn.BatchNorm1d(fc_layers[1])
        self.fc3 = nn.Linear(fc_layers[1], num_classes)

    def forward(self, xyz, feature):
        l0_points = feature
        l0_xyz = xyz[:, :3, :]
        B, _, _ = xyz.shape
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 256) (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 64) (B, 512, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 16) (B, 1024, 16)

        final_xyz, final_points = self.sa_final(l4_xyz, l4_points)

        x = final_points.view(B, 256)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x # x shape: B, N, num_classes
    
class PointNet2_small2_no_feature_prop(nn.Module):
    def __init__(self, num_classes=60, input_channel=3, keep_gripper_in_fps=False):
        super(PointNet2_small2_no_feature_prop, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=input_channel, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 64], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+64, [[128, 196, 128], [128, 196, 128]])
        self.sa_final = PointNetSetAbstraction(None, None, None, 128 + 128 + 3, [128, 128, 256], True)

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, xyz, feature):
        l0_points = feature
        l0_xyz = xyz[:, :3, :]
        B, _, _ = xyz.shape

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 512) (B, 96, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 128) (B, 256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 32) (B, 512, 32)

        final_xyz, final_points = self.sa_final(l3_xyz, l3_points)

        x = final_points.view(B, 256)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x # x shape: B, N, num_classes: outputing logtis
        
class PointNet2_superplus(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_superplus, self).__init__()
        self.sa0 = PointNetSetAbstractionMsg(npoint=2048, radius_list=[0.0125, 0.025], nsample_list=[16, 32], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128]])
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=64+128, mlp_list=[[64, 64, 128], [128, 196, 256]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=128+256, mlp_list=[[128, 196, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 256+256, [[256, 384, 512], [256, 384, 512]])
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 512+512, [[256, 384, 512], [256, 384, 512]])
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]])
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]])
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512, 512])
        self.fp5 = PointNetFeaturePropagation(512+512+512, [512, 512, 512])
        self.fp4 = PointNetFeaturePropagation(512+512+512, [512, 384, 256])
        self.fp3 = PointNetFeaturePropagation(256+256+256, [256, 256, 256])
        self.fp2 = PointNetFeaturePropagation(256+256+128, [256, 128, 128])
        self.fp1 = PointNetFeaturePropagation(128+128+64, [128, 128, 128])
        self.fp0 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l01_xyz, l01_points = self.sa0(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l01_xyz, l01_points) # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l01_points = self.fp1(l01_xyz, l1_xyz, l01_points, l1_points) # (B, 128, num_point)
        l0_points = self.fp0(l0_xyz, l01_xyz, None, l01_points) # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes

if __name__ == '__main__':

    from tqdm import tqdm
    model = PointNet2(num_classes=10).cuda()
    model.eval()
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True
    inpput = torch.rand(1, 3, 2000).cuda()
    out = model(inpput)
    max_diff = -1
    for _ in range(1):
        inpput_translated = inpput + 50
        out_translated = model(inpput_translated)
        diff = torch.norm(out-out_translated)
        max_diff = max(max_diff, diff)
        print("difference: ", diff)