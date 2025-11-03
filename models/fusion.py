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
# # models/fusion.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiModalFusion(nn.Module):
#     """三模态特征融合模块"""
#     def __init__(self, visual_dim=256, text_dim=512):  # ← 改为256
#         super().__init__()
        
#         # RGB-D融合：256*2 → 256
#         self.rgbd_fusion = nn.Sequential(
#             nn.Conv2d(visual_dim * 2, visual_dim, 1),
#             nn.BatchNorm2d(visual_dim),
#             nn.ReLU()
#         )
        
#         # 文本特征投影
#         self.text_proj = nn.Linear(text_dim, visual_dim)
        
#     def forward(self, rgb_feat, depth_feat, text_feat):
#         """
#         Args:
#             rgb_feat: [B, 256, H, W]
#             depth_feat: [B, 256, H, W]
#             text_feat: [B, 512]
#         Returns:
#             fused_feat: [B, 256, H, W]
#         """
#         B, C, H, W = rgb_feat.shape
        
#         # 1. RGB-D拼接融合
#         rgbd_cat = torch.cat([rgb_feat, depth_feat], dim=1)  # [B,512,H,W]
#         rgbd_fused = self.rgbd_fusion(rgbd_cat)              # [B,256,H,W]
        
#         # 2. 文本增强
#         text_feat_proj = self.text_proj(text_feat)           # [B,256]
#         text_feat_exp = text_feat_proj.unsqueeze(-1).unsqueeze(-1)
#         text_weight = torch.sigmoid(text_feat_exp)
#         enhanced_feat = rgbd_fused * (1 + text_weight)
        
#         return enhanced_feat  # [B,256,H,W]

# models/fusion.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiModalFusion(nn.Module):
#     def __init__(self, visual_dim=256, text_dim=512):
#         super().__init__()
        
#         # RGB-D融合
#         self.rgbd_fusion = nn.Sequential(
#             nn.Conv2d(visual_dim * 2, visual_dim, 1),
#             nn.BatchNorm2d(visual_dim),
#             nn.ReLU()
#         )
        
#         # ===== 关键改进：文本引导的空间注意力 =====
#         self.text_proj = nn.Linear(text_dim, visual_dim)
#         self.text_spatial_attn = nn.Sequential(
#             nn.Conv2d(visual_dim, visual_dim // 4, 1),
#             nn.ReLU(),
#             nn.Conv2d(visual_dim // 4, 1, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, rgb_feat, depth_feat, text_feat):
#         """
#         rgb_feat: [B, C, H, W]
#         depth_feat: [B, C, H, W]  (or compatible spatial size)
#         text_feat: [B, text_dim]
#         """
#         B, C, H, W = rgb_feat.shape

#         # 1. RGB-D 融合（保持原逻辑）
#         rgbd_cat = torch.cat([rgb_feat, depth_feat], dim=1)  # [B, 2C, H, W] if depth has same C
#         rgbd_fused = self.rgbd_fusion(rgbd_cat)              # [B, visual_dim, H, W]
#         # 记住 visual_dim == rgbd_fused.shape[1] (通常 256)

#         # 2. 文本语义投影到 visual_dim
#         # self.text_proj 在 __init__ 中就是 nn.Linear(text_dim, visual_dim)
#         text_proj = self.text_proj(text_feat)  # [B, visual_dim]

#         # 3. 保证 text_proj 与通道数一致（兼容性保护）
#         vis_C = rgbd_fused.shape[1]
#         if text_proj.shape[1] != vis_C:
#             # 如果维度不匹配，用一个线性变换把 text_proj 映射到 vis_C
#             text_proj = torch.nn.functional.linear(text_proj, 
#                                                 torch.eye(text_proj.shape[1], vis_C, device=text_proj.device)[:, :].t().contiguous())
#             # 上面是一个纯张量映射（无需新增参数），保证运行时不会崩

#         # 4. 通道级别调制（文本->通道权重）
#         text_channel_weight = torch.sigmoid(text_proj).view(B, vis_C, 1, 1)  # [B, C, 1, 1]
#         modulated_feat = rgbd_fused * text_channel_weight

#         # 5. 文本引导的空间注意力
#         text_guided_feat = modulated_feat + text_proj.view(B, vis_C, 1, 1)
#         spatial_attn = self.text_spatial_attn(text_guided_feat)  # [B,1,H,W]

#         # 6. 最终融合（保留原组合）
#         enhanced_feat = rgbd_fused * spatial_attn + modulated_feat

#         return enhanced_feat

# # models/fusion.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiModalFusion(nn.Module):
#     """RGB-D-Text 三模态融合模块（推理 & 竞赛版稳定实现）"""
#     def __init__(self, visual_dim=256, text_dim=512):
#         super().__init__()
        
#         # 1. RGB-D融合: [B,512,H,W] -> [B,256,H,W]
#         self.rgbd_fusion = nn.Sequential(
#             nn.Conv2d(visual_dim * 2, visual_dim, 1),
#             nn.BatchNorm2d(visual_dim),
#             nn.ReLU(inplace=True)
#         )
        
#         # 2. 文本特征投影: [B,512] -> [B,256]
#         self.text_proj = nn.Linear(text_dim, visual_dim)

#         # 3. 文本引导的空间注意力
#         self.text_spatial_attn = nn.Sequential(
#             nn.Conv2d(visual_dim, visual_dim // 4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(visual_dim // 4, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, rgb_feat, depth_feat, text_feat):
#         """
#         Args:
#             rgb_feat: [B,256,H,W]
#             depth_feat: [B,256,H,W]
#             text_feat: [B,512]
#         """
#         B, C, H, W = rgb_feat.shape
        
#         # (1) RGB-D融合
#         rgbd_cat = torch.cat([rgb_feat, depth_feat], dim=1)  # [B,512,H,W]
#         rgbd_fused = self.rgbd_fusion(rgbd_cat)              # [B,256,H,W]

#         # (2) 文本语义调制
#         text_proj = self.text_proj(text_feat)                # [B,256]
#         text_channel_weight = torch.sigmoid(text_proj).view(B, C, 1, 1)
#         modulated_feat = rgbd_fused * (1 + text_channel_weight)

#         # (3) 文本引导的空间注意力
#         spatial_attn = self.text_spatial_attn(modulated_feat)  # [B,1,H,W]

#         # (4) 最终融合
#         enhanced_feat = modulated_feat * (1 + spatial_attn)

#         return enhanced_feat
# models/fusion.py - 修复版
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusion(nn.Module):
    """RGB-D-Text 三模态融合模块"""
    def __init__(self, visual_dim=256, text_dim=512):
        super().__init__()
        
        # RGB-D融合: [B,512,H,W] -> [B,256,H,W]
        self.rgbd_fusion = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, 1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True)
        )
        
        # ✅ 修复：文本投影到 visual_dim (256)
        self.text_proj = nn.Linear(text_dim, visual_dim)  # 512->256

        # 文本引导的空间注意力
        self.text_spatial_attn = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(visual_dim // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_feat, text_feat):
        """
        Args:
            rgb_feat: [B,256,H,W]
            depth_feat: [B,256,H,W]
            text_feat: [B,512]
        Returns:
            enhanced_feat: [B,256,H,W]
        """
        B, C, H, W = rgb_feat.shape
        
        # 1. RGB-D融合
        rgbd_cat = torch.cat([rgb_feat, depth_feat], dim=1)  # [B,512,H,W]
        rgbd_fused = self.rgbd_fusion(rgbd_cat)              # [B,256,H,W]

        # 2. 文本语义投影
        text_proj = self.text_proj(text_feat)  # [B,512] -> [B,256]
        
        # ✅ 修复：确保 text_proj 是 [B, 256]
        assert text_proj.shape == (B, C), f"text_proj shape {text_proj.shape} != ({B}, {C})"
        
        # 3. 通道调制
        text_channel_weight = torch.sigmoid(text_proj).view(B, C, 1, 1)  # ✅ 现在正确了
        modulated_feat = rgbd_fused * (1 + text_channel_weight)

        # 4. 空间注意力
        spatial_attn = self.text_spatial_attn(modulated_feat)  # [B,1,H,W]

        # 5. 最终融合
        enhanced_feat = modulated_feat * (1 + spatial_attn)

        return enhanced_feat