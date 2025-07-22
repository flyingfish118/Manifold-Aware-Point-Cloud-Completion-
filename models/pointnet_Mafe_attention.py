#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Pingping Cai

import torch
from torch import nn, einsum
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation

from extensions.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from sklearn.manifold import Isomap
import numpy as np

class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


def MLP_Stacks(channels,bn=None):
    '''
    function    
        stack multiple layers of mlp based on input channel list
    input  
        channels: [list]
    output
        layer of multiple mlps
    '''
    layers = []
    last_channel = channels[0]
    for out_channel in channels[1:-1]:
        layers.append(MLP_Res(last_channel, None, out_channel))
        if bn:
            layers.append(nn.BatchNorm1d(out_channel))
        layers.append(nn.LeakyReLU())
        last_channel = out_channel
    layers.append(MLP_Res(last_channel, None, channels[-1]))
    return nn.Sequential(*layers)

def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)

    idx = ball_query(radius, nsample, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous()) # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    # scale to center
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self, npoint, nsample, radius, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(xyz, points, self.npoint, self.nsample, self.radius, self.use_xyz)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        return new_xyz, new_points


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:MLP_CONV
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous())
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0/dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx, weight) # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
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
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    sorted_dist, indices = torch.sort(sqrdists, dim=-1, descending=False)
    idx = indices[:, :, pad: nsample+pad]
    #sdist = sorted_dist[:,:,pad: nsample+pad]
    return idx.int()


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


import torch
import torch.nn as nn
import torch.nn.functional as F

class DGCNN_Grouper_Res(nn.Module):
    def __init__(self, npoint, nsample, input_trans):
        super().__init__()
        '''
        input_trans: [in_dim, mid_dim, out_dim], e.g. [3, 128, 128]
        '''
        print('using group version 2')
        self.k = nsample
        self.npoint = npoint
        self.nsample = nsample
        self.input_trans_dim = input_trans

        self.input_trans = nn.Conv1d(input_trans[0], input_trans[1] // 2, 1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_trans[1], input_trans[2], kernel_size=1, bias=False),
            nn.GroupNorm(4, input_trans[2]),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 替代 MLP_Res 的模块
        self.mlp1 = nn.Sequential(
            nn.Conv2d(input_trans[2], 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(input_trans[2] + 512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.ReLU()
        )

        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.ReLU()
        )

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        '''
        coor_q, coor_k: (B, 3, N)
        x_q, x_k: (B, C, N)
        '''
        k = self.k
        B, _, N = x_q.shape

        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(2, 1), coor_q.transpose(2, 1))  # (B, N, k)
            idx = idx.transpose(1, 2).contiguous()  # (B, k, N)
            idx_base = torch.arange(0, B, device=x_q.device).view(-1, 1, 1) * N
            idx = idx + idx_base
            idx = idx.view(-1)

        C = x_k.shape[1]
        x_k = x_k.transpose(2, 1).contiguous()  # (B, N, C)
        neighbors = x_k.view(B * N, -1)[idx, :]  # (B*k*N, C)
        neighbors = neighbors.view(B, k, N, C).permute(0, 3, 2, 1)  # (B, C, N, k)
        x_q = x_q.view(B, C, N, 1).repeat(1, 1, 1, k)

        feature = torch.cat([neighbors - x_q, x_q], dim=1)  # (B, 2C, N, k)
        return feature

    def forward(self, x, f):
        '''
        Input:
            x: (B, 3, N)   — coordinates
            f: (B, in_dim, N) — features (same as x at first call)
        Output:
            global_feat: (B, 512, 1)
            local_feat:  (B, 128, N)
        '''
        B, _, N = x.shape
        coor = x

        f = self.input_trans(f)  # (B, mid_dim//2, N)
        feat = self.get_graph_feature(coor, f, coor, f)  # (B, mid_dim, N, k)
        feat = self.layer1(feat)                        # (B, out_dim, N, k)

        local_base = feat.max(dim=-1)[0]  # (B, out_dim, N)

        global_feat = self.mlp1(feat).max(dim=-1)[0]    # (B, 512, N)
        global_max = global_feat.max(dim=-1, keepdim=True)[0]  # (B, 512, 1)

        # expand global feature to concatenate
        global_expand = global_max.expand(-1, -1, N)  # (B, 512, N)

        local_feat = self.mlp2(torch.cat([local_base, global_expand], dim=1))  # (B, 128, N)

        global_feat = self.mlp3(local_feat).max(dim=-1, keepdim=True)[0]  # (B, 512, 1)

        return global_feat.contiguous(), local_feat.contiguous()


class DGCNN_Grouper(nn.Module):
    def __init__(self, npoint, nsample, input_trans, ex_g=False):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.ex_g = ex_g
        self.k = nsample
        self.npoint = npoint
        self.nsample = nsample
        self.input_trans_dim = input_trans
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(self.input_trans_dim[0], int(self.input_trans_dim[1] / 2), 1)

        self.layer1 = nn.Sequential(nn.Conv2d(self.input_trans_dim[1]  , self.input_trans_dim[2], kernel_size=1, bias=False),
                                   nn.GroupNorm(4, self.input_trans_dim[2]),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x , fps_idx

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np

            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, f):
        '''
            INPUT:
                x : bs 3, N
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs 3 N
                f    bs C(128)  N
        '''
        x = x.contiguous()
        coor = x

        f = self.input_trans(f) #

        f = self.get_graph_feature(coor, f, coor, f) # b c n k
        f = self.layer1(f)#b c n k
        f = f.max(dim=-1, keepdim=False)[0] # b c n

        coor_q, f_q, fps_idx = self.fps_downsample(coor, f, self.npoint)
        if self.ex_g == True:
            f_q = f.max(dim=-1, keepdim=False)[0].unsqueeze(-1)

        return coor_q.contiguous(), f_q.contiguous(), fps_idx

class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        '''
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        '''
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd



class LocalGeometricRelationshipPerception(nn.Module):
    def __init__(self, out_channels, k=16, m=5):
        super(LocalGeometricRelationshipPerception, self).__init__()
        self.k = k
        self.m = m
        self.mlp = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x_q, x, fps_idx,
                anchor_points,  # (B, M, 3)
                map_idx,        # (B, N)
                anchor_dist     # (B, M, M)
               ):
        """
        Args:
            x_q:         (B, n, 3)  查询点集 (已通过与 x 相同的 fps_idx 下采样)
            x:           (B, n, 3)  点云 (同一个 fps_idx 下采样得到)
            fps_idx:     (B, n)     下采样索引, 指示原始点 x[b, fps_idx[b, i]] => 现有的 x[b, i]
            anchor_points: (B, M, 3) Anchor点坐标
            map_idx:     (B, N)     原始 N 个点各自的 Anchor 索引(范围 [0..M-1])
            anchor_dist: (B, M, M)  两个 Anchor 间的测地距离 (Step4里的D_iso)

        Returns:
            local_features: (B, n, out_channels)
            weights:        (B, out_channels, n, k)
        """
        B, n, _ = x.shape

        # ========== 1) 为下采样后的点 x / x_q 找到各自对应的 Anchor 索引 ==========
        # map_idx[b, idx] => 该 idx 点属于哪个 Anchor
        # fps_idx[b, i]   => 原始点云中的点索引
        # A_idx_for_x: (B, n), A_idx_for_xq: (B, n) 若 x_q == x，可以直接复

        # ========== 2) 搜 k 个邻居 (在 x 中找 x_q 的邻居) ==========
        # idx: (B, n, k)
        idx = knn_point(self.k, x, x_q)

        # Gather 邻居的坐标 => neighbors: (B, n, k, 3)
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        neighbors  = torch.gather(
            x.unsqueeze(2).expand(-1, -1, self.k, -1), 1, idx_expand
        )

        # ========== 3) R1: 欧几里得差 ==========
        # (B, n, k, 3)
        R1 = torch.abs(x_q.unsqueeze(2) - neighbors)

        # ========== 4) 用 Step 4 的公式来构造 R2 (基于 Anchor) ==========
        # 公式: dis(a, b) = ||a - Anchor[A]|| + anchor_dist[A, B] + ||b - Anchor[B]||
        
        # (a) 找到 x_q 和 neighbors 对应的 Anchor 索引 => A_idx_q, B_idx
        # A_idx_q: (B, n)
        # B_idx:   (B, n, k)  (neighbors 的 Anchor)
        A_idx_q = map_idx
        B_idx   = torch.gather(map_idx.unsqueeze(2).expand(-1, -1, idx.shape[2]), 1, idx)  # 在 (B, n) 里按 idx 搜

        # (b) 根据 A_idx_q / B_idx 搜到 Anchor 的真实坐标
        # anchor_points: (B, M, 3)
        anchorA = _gather_anchors_by_index(anchor_points, A_idx_q)      # (B, n, 3)
        anchorB = _gather_anchors_by_index(anchor_points, B_idx)        # (B, n, k, 3)

        # (c) 计算 ||x_q - anchorA|| 和 ||neighbors - anchorB||
        dist_xq_anchorA    = torch.norm(x_q - anchorA, dim=-1)          # (B, n)
        dist_neighbors_B   = torch.norm(neighbors - anchorB, dim=-1)    # (B, n, k)

        # (d) gather anchor_dist[A_idx, B_idx]
        # anchor_dist: (B, M, M)
        anchorDistAB = _gather_anchor_dist(anchor_dist, A_idx_q, B_idx) # (B, n, k)

        # (e) 组合： R2_scalar[b, i, j] = dist_xq_anchorA[b, i] + anchorDistAB[b, i, j] + dist_neighbors_B[b, i, j]
        dist_xq_anchorA_expand = dist_xq_anchorA.unsqueeze(-1)          # (B, n, 1)
        R2_scalar = dist_xq_anchorA_expand + anchorDistAB + dist_neighbors_B  # (B, n, k)

        # (f) 拓展到 3 通道，以拼接到 R1
        # => R2: (B, n, k, 3)
        R2 = R2_scalar.unsqueeze(-1).expand(-1, -1, -1, 3)

        # ========== 5) 拼接 [R1, R2] 并送入 MLP ==========
        relationship_features = torch.cat([R1, R2], dim=-1).permute(0, 3, 1, 2)
        # => (B, 6, n, k)

        weights = self.mlp(relationship_features)  # (B, out_channels, n, k)

        # ========== 6) 聚合邻居特征 ==========
        local_features, _ = torch.max(weights, dim=-1)  # (B, out_channels, n)

        # 输出: (B, n, out_channels), (B, out_channels, n, k)
        return local_features.permute(0, 2, 1), weights

# -------------------------------------------------------------------------
# 辅助函数：批量 gather anchor_points / anchor_dist
# -------------------------------------------------------------------------

def _gather_anchors_by_index(anchor_points, idx):
    """
    anchor_points: (B, M, 3)
    idx: (B, n) or (B, n, k)
    return: (B, n, 3) or (B, n, k, 3)
    """
    B, M, _ = anchor_points.shape
    shape_out = list(idx.shape) + [3]  # => (B, n, 3) or (B, n, k, 3)

    idx_flat = idx.reshape(B, -1)     # (B, n) or (B, n*k)
    batch_idx = torch.arange(B, device=anchor_points.device).view(B, 1)
    batch_idx = batch_idx.expand(-1, idx_flat.shape[1])  # (B, n) or (B, n*k)

    gathered = anchor_points[batch_idx, idx_flat, :]      # (B, n or n*k, 3)
    gathered = gathered.view(*shape_out)                  # reshape back
    return gathered

def _gather_anchor_dist(anchor_dist, A_idx, B_idx):
    """
    anchor_dist: (B, M, M)
    A_idx: (B, n)
    B_idx: (B, n, k)
    return: (B, n, k)
    """
    B, M, _ = anchor_dist.shape
    _, n = A_idx.shape
    _, n2, k = B_idx.shape
    assert n == n2, "A_idx and B_idx dimension mismatch in n"

    # flatten
    B_idx_flat = B_idx.reshape(B, -1)  # (B, n*k)

    # broadcast A_idx => (B, n, k)
    A_idx_expand = A_idx.unsqueeze(-1).expand(-1, -1, k)  # (B, n, k)
    A_idx_flat = A_idx_expand.reshape(B, -1)              # (B, n*k)

    batch_idx = torch.arange(B, device=anchor_dist.device).view(B, 1)
    batch_idx = batch_idx.expand(-1, A_idx_flat.shape[1]) # (B, n*k)

    dist_gathered = anchor_dist[batch_idx, A_idx_flat, B_idx_flat]  # (B, n*k)
    dist_gathered = dist_gathered.view(B, n, k)
    return dist_gathered



class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )
        self.lgrp = LocalGeometricRelationshipPerception(out_channels=dim, k=16)

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos, fps_idx, P_anchor, map_idx, anchor_dist):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        _, f_l = self.lgrp(pos.transpose(-1, -2), pos.transpose(-1, -2), fps_idx, P_anchor, map_idx, anchor_dist)
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn


        attention = self.attn_mlp(qk_rel + pos_embedding + f_l)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding + f_l

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y+identity

def get_nearest_index(target, source, k=1, return_dis=False):
    """
    Args:
        target: (bs, 3, v1)
        source: (bs, 3, v2)
    Return:
        nearest_index: (bs, v1, 1)
    """
    inner = torch.bmm(target.transpose(1, 2), source)  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=1)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=1)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(
        2) - 2 * inner  # (bs, v1, v2)
    nearest_dis, nearest_index = torch.topk(d_norm_2,
                                            k=k,
                                            dim=-1,
                                            largest=False)
    if not return_dis:
        return nearest_index
    else:
        return nearest_index, nearest_dis


def indexing_neighbor(x, index):
    """
    Args:
        x: (bs, dim, num_points0)
        index: (bs, num_points, k)
    Return:
        feature: (bs, dim, num_points, k)
    """
    batch_size, num_points, k = index.size()

    id_0 = torch.arange(batch_size).view(-1, 1, 1)

    x = x.transpose(2, 1).contiguous()  # (bs, num_points, num_dims)
    feature = x[id_0, index]  # (bs, num_points, k, num_dims)
    feature = feature.permute(0, 3, 1,
                              2).contiguous()  # (bs, num_dims, num_points, k)

    return feature



class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True, 
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )
        # self.lgrp = LocalGeometricRelationshipPerception(out_channels=64, k=20)

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor,1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = self.mlp_v(torch.cat([key, query], 1)) # (B, dim, N)
        identity = value
        key = self.conv_key(key) # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        # _, f_l = self.lgrp(pos_flipped, pos_flipped)# (B, dim, N, k)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key 
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)
        pos_embedding = pos_embedding 

        # upfeat embedding
        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat) # (B, dim, N)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn)  # (B, dim, N, k)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel ) # (B, dim, N*up_factor, k)

        # softmax function
        attention = self.scale(attention)

        # knn value is correct
        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel # (B, dim, N, k)
        value = self.upsample1(value) # (B, dim, N*up_factor, k)

        agg = torch.einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg) # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity) # (B, out_dim, N)
        identity = self.upsample2(identity) # (B, out_dim, N*up_factor)

        return y+identity

class UpLayer(nn.Module):
    """
    Upsample Layer with upsample transformers
    """
    def __init__(self, dim, seed_dim, up_factor=2, i=0, radius=1, n_knn=20, interpolate='three', attn_channel=True):
        super(UpLayer, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.n_knn = n_knn
        self.interpolate = interpolate

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + seed_dim, layer_dims=[256, dim])

        self.uptrans1 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, up_factor=None)
        self.uptrans2 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, attn_channel=attn_channel, up_factor=self.up_factor)

        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim*2, hidden_dim=dim, out_dim=dim)

        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, seed, seed_feat, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, feat_dim, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_new: Tensor, upsampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape

        # Collect seedfeature
        if self.interpolate == 'nearest':
            idx = get_nearest_index(pcd_prev, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3) # (B, seed_dim, N_prev)
        elif self.interpolate == 'three':
            # three interpolate
            idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True) # (B, N_prev, 3), (B, N_prev, 3)
            dist_recip = 1.0 / (dis + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True) # (B, N_prev, 1)
            weight = dist_recip / norm # (B, N_prev, 3)
            feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1), dim=-1) # (B, seed_dim, N_prev)
        else:
            raise ValueError('Unknown Interpolation: {}'.format(self.interpolate))

        # Query mlps
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_upsample], 1)
        Q = self.mlp_2(feat_1)

        # Upsample Transformers
        H = self.uptrans1(pcd_prev, K_prev if K_prev is not None else Q, Q, upfeat=feat_upsample) # (B, 128, N_prev)
        feat_child = self.uptrans2(pcd_prev, K_prev if K_prev is not None else H, H, upfeat=feat_upsample) # (B, 128, N_prev*up_factor)

        # Get current features K
        H_up = self.upsample(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        # New point cloud
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_new = self.upsample(pcd_prev)
        pcd_new = pcd_new + delta

        return pcd_new, K_curr
    
class DictFormer(nn.Module):
    def __init__(self, dim_feat=512, hidd_dim=64):
        super(DictFormer, self).__init__()
        #self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.query = nn.Conv1d(dim_feat, hidd_dim, 1)
        self.key = nn.Conv1d(dim_feat, hidd_dim, 1)
        self.weight_mlp = nn.Sequential(
            nn.Conv1d(hidd_dim, hidd_dim, 1),
            nn.BatchNorm1d(hidd_dim),
            nn.ReLU(),
            nn.Conv1d(hidd_dim, 1, 1)
        )

    def forward(self, feat, dict_feats):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        b,c,n = feat.size()
        
        q = self.query(feat)  # (b, 64, 256)
        #query_embed.unsqueeze(0) # (1, 512, num_dicts)
        k = self.key(dict_feats.unsqueeze(0)) # (b, 64, 128)

        qk_rel = q * k
        w = self.weight_mlp(qk_rel)
        w = torch.softmax(w, -1)
        #print(att.size())
        agg = torch.sum(dict_feats*w,dim=2,keepdim=True)  # b, dim, n
        return agg+feat,w


















