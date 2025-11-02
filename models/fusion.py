# # ============================================
# # 3. models/fusion.py - 多模态融合
# # ============================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple

# class MultiModalFusion(nn.Module):
#     """三模态特征融合模块"""
#     def __init__(self, visual_dim=1024, text_dim=512):
#         super().__init__()
        
#         # RGB-D融合
#         self.rgbd_fusion = nn.Sequential(
#             nn.Conv2d(visual_dim * 2, visual_dim, 1),
#             nn.BatchNorm2d(visual_dim),
#             nn.ReLU()
#         )
        
#         # 文本特征投影
#         self.text_proj = nn.Linear(text_dim, visual_dim)
        
#         # 跨模态注意力
#         self.cross_attn = nn.MultiheadAttention(
#             visual_dim, num_heads=8, batch_first=True
#         )
        
#     def forward(self, rgb_feat, depth_feat, text_feat):
#         """
#         Args:
#             rgb_feat: [B, C, H, W]
#             depth_feat: [B, C, H, W]
#             text_feat: [B, D]
#         Returns:
#             fused_feat: [B, C, H, W]
#         """
#         B, C, H, W = rgb_feat.shape
        
#         # 1. RGB-D特征拼接融合
#         rgbd_cat = torch.cat([rgb_feat, depth_feat], dim=1)
#         rgbd_fused = self.rgbd_fusion(rgbd_cat)  # [B, C, H, W]
        
#         # 2. 文本特征处理
#         text_feat_proj = self.text_proj(text_feat)  # [B, C]
#         text_feat_exp = text_feat_proj.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
#         # 3. 文本引导的特征增强（简单方式：加权）
#         text_weight = torch.sigmoid(text_feat_exp)
#         enhanced_feat = rgbd_fused * (1 + text_weight)
        
#         return enhanced_feat


#777
# models/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusion(nn.Module):
    """三模态特征融合模块"""
    def __init__(self, visual_dim=256, text_dim=512):  # ← 改为256
        super().__init__()
        
        # RGB-D融合：256*2 → 256
        self.rgbd_fusion = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, 1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU()
        )
        
        # 文本特征投影
        self.text_proj = nn.Linear(text_dim, visual_dim)
        
    def forward(self, rgb_feat, depth_feat, text_feat):
        """
        Args:
            rgb_feat: [B, 256, H, W]
            depth_feat: [B, 256, H, W]
            text_feat: [B, 512]
        Returns:
            fused_feat: [B, 256, H, W]
        """
        B, C, H, W = rgb_feat.shape
        
        # 1. RGB-D拼接融合
        rgbd_cat = torch.cat([rgb_feat, depth_feat], dim=1)  # [B,512,H,W]
        rgbd_fused = self.rgbd_fusion(rgbd_cat)              # [B,256,H,W]
        
        # 2. 文本增强
        text_feat_proj = self.text_proj(text_feat)           # [B,256]
        text_feat_exp = text_feat_proj.unsqueeze(-1).unsqueeze(-1)
        text_weight = torch.sigmoid(text_feat_exp)
        enhanced_feat = rgbd_fused * (1 + text_weight)
        
        return enhanced_feat  # [B,256,H,W]