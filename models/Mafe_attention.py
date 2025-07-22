#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Pingping Cai

import torch
import torch.nn as nn
import math
from extensions.chamfer_dist import ChamferDistanceL2,ChamferDistanceL1
from sklearn.manifold import Isomap
import numpy as np

from models.pointnet_Mafe_attention import PointNet_SA_Module_KNN, MLP_Res, UpLayer, fps_subsample, Transformer, DGCNN_Grouper
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation

from .build import MODELS
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter("ignore", SparseEfficiencyWarning)
warnings.filterwarnings("ignore", message="The number of connected components")
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function")

class UNet(nn.Module):
    def __init__(self, dim_feat=384, num_seeds = 1024, seed_fea=128):
        '''
        Extract information from partial point cloud
        '''
        super(UNet, self).__init__()
        self.num_seed = num_seeds
        self.sa_module_1 = DGCNN_Grouper(num_seeds//2, 16,[3, 8, seed_fea])
        self.pos_embed_1 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, seed_fea)
        )  
        self.transformer_1 = Transformer(seed_fea, dim=64)
        self.sa_module_2 = DGCNN_Grouper(num_seeds//8, 16,[seed_fea, seed_fea, 256])
        self.pos_embed_2 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 256)
        )
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_4 = DGCNN_Grouper(1, num_seeds//8,[256, 256, dim_feat], ex_g=True)
        
        self.ps_0 = nn.ConvTranspose1d(dim_feat, 256, num_seeds//16, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 256, hidden_dim=256, out_dim=256)

        self.ps_1 = nn.ConvTranspose1d(256, 256, 2,2, bias=True)
        self.ps_2 = nn.ConvTranspose1d(256, 128, 2,2, bias=True)

        self.transformer_4 = Transformer(128, dim=64)

        self.mlp_3 = MLP_Res(in_dim=seed_fea, hidden_dim=32, out_dim=3)
        self.mlp_proj_1 = nn.Sequential(
            nn.Linear(128, 128 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(128 * 2, 128)
        )
        self.mlp_proj_2 = nn.Sequential(
            nn.Linear(128, 128 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(128 * 2, 256)
        )

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        
        l0_xyz = point_cloud
        l0_fea = point_cloud
        P_anchor = fps_subsample(point_cloud.permute(0, 2, 1).contiguous(), 128) #b n 3
        map_idx = map_to_nearest_anchor(point_cloud, P_anchor.permute(0, 2, 1)) #b n 
        anchor_dist = isomap(P_anchor.permute(0, 2, 1)) #b n n

        ## Encoder        
        l1_xyz, l1_fea, fps_idx = self.sa_module_1(l0_xyz, l0_fea)  # (B, 3, 512), (B, 128, 512)
        map_idx  = torch.gather(map_idx, 1, fps_idx.long())  # (B, n)

        anchor_emb = build_anchor_geodesic_encoding(map_idx, anchor_dist)  # (N, 128)

        anchor_emb = self.mlp_proj_1(anchor_emb)
        l1_fea = l1_fea + anchor_emb.permute(0, 2, 1)


        l1_fea = self.transformer_1(l1_fea, l1_xyz, fps_idx ,P_anchor, map_idx, anchor_dist)
        l2_xyz, l2_fea, fps_idx = self.sa_module_2(l1_xyz, l1_fea)  # (B, 3, 128), (B, 256, 128)
        map_idx  = torch.gather(map_idx, 1, fps_idx.long())  # (B, n)

        anchor_emb = build_anchor_geodesic_encoding(map_idx, anchor_dist)  # (N, 128)
        anchor_emb = self.mlp_proj_2(anchor_emb)
        l2_fea = l2_fea + anchor_emb.permute(0, 2, 1)


        l2_fea = self.transformer_2(l2_fea, l2_xyz, fps_idx, P_anchor, map_idx, anchor_dist)
        

        l4_xyz, l4_fea, fps_idx = self.sa_module_4(l2_xyz, l2_fea)  # (B, 3, 1), (B, out_dim, 1)
        
        ## Decoder
        #seed generate 0
        u0_fea = self.ps_0(l4_fea)  # (b, 256, 128)
        u0_fea = self.mlp_1(torch.cat([u0_fea, l4_fea.repeat((1, 1, u0_fea.size(2)))], 1))

        #upconv1
        u1_fea = self.ps_1(u0_fea)


        # skip concat
        u1_fea = torch.cat([l2_fea,u1_fea],dim=2)  # (b, 256, 256)

        #upconv 2
        u2_fea = self.ps_2(u1_fea)  # (b, 64, 512)

        # skip concat
        u2_xyz = self.mlp_3(u2_fea)
        
        u2_xyz = torch.cat([l1_xyz,u2_xyz],dim=2) # (b, 3, 1024)

        map_idx = map_to_nearest_anchor(u2_xyz, P_anchor.permute(0, 2, 1))

        u2_fea = torch.cat([l1_fea,u2_fea],dim=2) # (b, 64, 1024)

        u2_fea = self.transformer_4(u2_fea, u2_xyz, fps_idx, P_anchor, map_idx, anchor_dist)



        return l4_fea, u2_xyz, u2_fea

def build_anchor_geodesic_encoding(map_idx, anchor_geodesic):
    """
    map_idx: (B, 512)
    anchor_geodesic: (B, 128, 128)
    return: (B, 512, 128)
    """
    B, N = map_idx.shape
    device = map_idx.device

    # 构造 batch 维索引 (B, 1)
    batch_idx = torch.arange(B, device=device).unsqueeze(1)    # (B, 1)
    batch_idx = batch_idx.expand(-1, N)                        # (B, 512)

    # 用 batch_idx 和 map_idx 共同索引 anchor_geodesic
    # anchor_geodesic[batch_idx, map_idx] : (B, 512, 128)
    return anchor_geodesic[batch_idx, map_idx]                # (B, 512, 128)


def isomap(point_cloud):
        """
        使用 Isomap 计算三维流形上各点间的测地距离矩阵
        
        Args:
            point_cloud: (B, 3, N)
                一批点云数据，批大小 B，每个点云有 N 个点，每个点 3 维坐标。
                
        Returns:
            dist_matrices: (B, N, N)
                每个批次点云对应的 NxN 测地距离矩阵。
        """
        B, C, N = point_cloud.shape

        dist_list = []
        for i in range(B):
            # 提取当前 batch 的点云数据，并转置为 (N, 3)
            pc_np = point_cloud[i].permute(1, 0).cpu().detach().numpy()  # shape: (N, 3)
            
            # 创建 Isomap 对象（此时不关心 n_components，因为我们只要测地距离）
            isomap = Isomap(n_components=2, n_neighbors=3)
            
            # 拟合以构建测地图
            isomap.fit(pc_np)
            
            # 直接读取测地距离矩阵 (N, N)
            dist_matrix_np = isomap.dist_matrix_
            
            # 转为 torch.Tensor
            dist_matrix_torch = torch.tensor(dist_matrix_np, 
                                             dtype=point_cloud.dtype, 
                                             device=point_cloud.device)
            
            dist_list.append(dist_matrix_torch)
        
        # 拼接成 (B, N, N)
        dist_matrices = torch.stack(dist_list, dim=0)
        return dist_matrices



def map_to_nearest_anchor(P_all, P_anchor):
    """
    给定原始点云和代表点云，计算每个原始点最近的代表点索引。

    Args:
        P_all:    (B, 3, N)  原始点云
        P_anchor: (B, 3, M)  代表点云
    
    Returns:
        map_idx:  (B, N)     每个原始点对应的最近代表点索引
    """
    B, _, N = P_all.shape
    _, _, M = P_anchor.shape

    # 转置为 (B, N, 3) / (B, M, 3) 便于批量计算 pairwise 距离
    P_all_flipped = P_all.permute(0, 2, 1).contiguous()    # (B, N, 3)
    P_anchor_flipped = P_anchor.permute(0, 2, 1).contiguous()  # (B, M, 3)

    # 计算 pairwise 距离：
    # diff[b, n, m, :] = P_all_flipped[b, n, :] - P_anchor_flipped[b, m, :]
    # 得到距离张量 dists[b, n, m]
    diff = P_all_flipped.unsqueeze(2) - P_anchor_flipped.unsqueeze(1)  # (B, N, M, 3)
    dists = torch.norm(diff, dim=-1)  # (B, N, M)

    # 取沿 M 维度的 argmin，得到最近代表点索引
    map_idx = torch.argmin(dists, dim=2)  # (B, N)

    return map_idx




    
class PostProcess(nn.Module):
    def __init__(self, embed_dim=128, upscale=[1,4,4]):
        super(PostProcess, self).__init__()
        #self.mlp_1 = nn.Conv1d(64, 128, 1) #
        #self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[32, 128])
        ##Parametric Surface Constrained Upsampler(self,upscale=4, dim_feat=512)
        up_layers = []
        for i, factor in enumerate(upscale):
            up_layers.append(UpLayer(dim=embed_dim, seed_dim=embed_dim, up_factor=factor, i=i, n_knn=20, radius=1, 
                             interpolate='three', attn_channel=True))
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, seed, fea):
        """
        Args:
            global_shape_fea: Tensor, (b, dim_feat, 1)
            pcd: Tensor, (b, 3, n)
        """
        pred_pcds = [seed.permute(0, 2, 1).contiguous()]
        # Upsample layers
        K_prev = None
        pcd = seed # (B, 3, 256)
        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, seed, fea, K_prev)
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())

        return pred_pcds


# 3D completion
@MODELS.register_module()
class Mafe_attention(nn.Module):
    def __init__(self, config, **kwargs):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            up_factors: list of int
        """
        super(Mafe_attention, self).__init__()
        self.num_pred = config.num_pred
        self.use_L2 = config.l2_loss
        self.feat_extractor = UNet(dim_feat=config.dim_feat,num_seeds=config.num_seeds,seed_fea=config.seed_fea)
        self.decoder = PostProcess(embed_dim=config.seed_fea,upscale = config.upscales)
        self.build_loss_func()

    def build_loss_func(self):
        if self.use_L2==0:
            self.loss_func = ChamferDistanceL1()
        else:
            self.loss_func = ChamferDistanceL2()
        #self.penalty_func = expansionPenaltyModule()

    def get_loss(self, ret, gt):
        Pc, P3, P1, P2 = ret

        cd0 = self.loss_func(Pc, gt)
        cd1 = self.loss_func(P1, gt)
        cd2 = self.loss_func(P2, gt)
        cd3 = self.loss_func(P3, gt)

        loss_all = (1*cd0 + 1*cd1 + 1*cd2 + 1*cd3) * 1e3 
        return loss_all, cd0, cd3
    

    def forward(self, partial_point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """

        #pcd_bnc = point_cloud
        in_pcd = partial_point_cloud.permute(0, 2, 1).contiguous()     

        partial_shape_code,xyz,fea= self.feat_extractor(in_pcd)

        pcd,p1,p2,p3 = self.decoder(xyz, fea)

        return pcd,p3,p1,p2