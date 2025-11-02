# # ============================================
# # 4. models/tracker.py - 主跟踪器
# # ============================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.backbone import RGBDBackbone
# from models.fusion import MultiModalFusion
# from models.text_encoder import TextEncoder

# class RGBDTextTracker(nn.Module):
#     """RGB-D-Text多模态跟踪器"""
#     def __init__(self):
#         super().__init__()
        
#         self.backbone = RGBDBackbone()
#         self.text_encoder = TextEncoder()
#         self.fusion = MultiModalFusion()
        
#         # 跟踪头（简单的相关层）
#         self.head = nn.Sequential(
#             nn.Conv2d(1024, 256, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 1, 1)  # 输出热力图
#         )
        
#         # 边界框回归
#         self.bbox_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, 4)  # [x, y, w, h]
#         )
        
#     def forward(self, template_rgb, template_depth, text,
#                 search_rgb, search_depth):
#         """
#         Args:
#             template_rgb/depth: [B, C, 127, 127] 模板图像
#             text: list of strings
#             search_rgb/depth: [B, C, 255, 255] 搜索区域
#         Returns:
#             bbox: [B, 4]
#         """
#         # 提取模板特征
#         temp_rgb_feat, temp_depth_feat = self.backbone(
#             template_rgb, template_depth
#         )
#         text_feat = self.text_encoder(text)
#         temp_feat = self.fusion(temp_rgb_feat, temp_depth_feat, text_feat)
        
#         # 提取搜索区域特征
#         search_rgb_feat, search_depth_feat = self.backbone(
#             search_rgb, search_depth
#         )
#         search_feat = self.fusion(search_rgb_feat, search_depth_feat, text_feat)
        
#         # 相关性计算（简化版）
#         response_map = self.head(search_feat)
        
#         # 边界框预测
#         bbox = self.bbox_head(search_feat)
        
#         return bbox, response_map

# # models/tracker.py 777
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.backbone import RGBDBackbone
# from models.fusion import MultiModalFusion
# from models.text_encoder import TextEncoder

# class RGBDTextTracker(nn.Module):
#     """RGB-D-Text多模态跟踪器"""
#     def __init__(self):
#         super().__init__()
        
#         self.backbone = RGBDBackbone()
#         self.text_encoder = TextEncoder()
#         self.fusion = MultiModalFusion(visual_dim=256, text_dim=512)  # ← 匹配256
        
#         # 跟踪头
#         self.head = nn.Sequential(
#             nn.Conv2d(256, 256, 3, padding=1),  # ← 改为256输入
#             nn.ReLU(),
#             nn.Conv2d(256, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 1, 1)
#         )
        
#         # 边界框回归
#         self.bbox_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 256),  # ← 改为256输入
#             nn.ReLU(),
#             nn.Linear(256, 4)
#         )
        
#     def forward(self, template_rgb, template_depth, text,
#                 search_rgb, search_depth):
#         # 提取模板特征
#         temp_rgb_feat, temp_depth_feat = self.backbone(template_rgb, template_depth)
#         text_feat = self.text_encoder(text)
#         temp_feat = self.fusion(temp_rgb_feat, temp_depth_feat, text_feat)
        
#         # 提取搜索区域特征
#         search_rgb_feat, search_depth_feat = self.backbone(search_rgb, search_depth)
#         search_feat = self.fusion(search_rgb_feat, search_depth_feat, text_feat)
        
#         # 预测
#         response_map = self.head(search_feat)
#         bbox = self.bbox_head(search_feat)
        
#         return bbox, response_map

# models/tracker.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import RGBDBackbone
from models.fusion import MultiModalFusion
from models.text_encoder import TextEncoder

class RGBDTextTracker(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = RGBDBackbone()
        self.text_encoder = TextEncoder()
        self.fusion = MultiModalFusion(visual_dim=256, text_dim=512)
        
        # ===== 关键改进1：模板-搜索相关性模块 =====
        self.correlation = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # ===== 关键改进2：位置敏感的响应图 =====
        self.response_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # ===== 关键改进3：空间保持的bbox回归 =====
        self.bbox_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 1)  # 输出4通道热力图
        )
        
    def compute_correlation(self, template_feat, search_feat):
        """计算模板与搜索区域的相关性"""
        B, C, Ht, Wt = template_feat.shape
        _, _, Hs, Ws = search_feat.shape
        
        # 全局池化模板特征作为kernel
        template_kernel = F.adaptive_avg_pool2d(template_feat, 1)  # [B,C,1,1]
        
        # 深度可分离卷积模拟相关操作
        corr_feat = search_feat * template_kernel  # 广播乘法
        corr_feat = self.correlation(corr_feat)
        
        return corr_feat
        
    def forward(self, template_rgb, template_depth, text, search_rgb, search_depth):
        # 提取特征
        temp_rgb_feat, temp_depth_feat = self.backbone(template_rgb, template_depth)
        text_feat = self.text_encoder(text)
        temp_feat = self.fusion(temp_rgb_feat, temp_depth_feat, text_feat)
        
        search_rgb_feat, search_depth_feat = self.backbone(search_rgb, search_depth)
        search_feat = self.fusion(search_rgb_feat, search_depth_feat, text_feat)
        
        # ===== 核心：相关性匹配 =====
        corr_feat = self.compute_correlation(temp_feat, search_feat)  # [B,128,H,W]
        
        # 响应图
        response_map = self.response_head(corr_feat)  # [B,1,H,W]
        
        # bbox回归（空间保持）
        bbox_map = self.bbox_head(corr_feat)  # [B,4,H,W]
        
        # ===== 从响应图中提取bbox =====
        bbox = self.extract_bbox_from_map(bbox_map, response_map)
        
        return bbox, response_map
    
    def extract_bbox_from_map(self, bbox_map, response_map):
        """从空间热力图中提取最终bbox"""
        B, _, H, W = bbox_map.shape
        
        # 找到响应最大的位置
        response_flat = response_map.view(B, -1)
        max_idx = response_flat.argmax(dim=1)
        
        # 提取对应位置的bbox
        bbox_flat = bbox_map.view(B, 4, -1)
        bbox = torch.stack([bbox_flat[i, :, max_idx[i]] for i in range(B)])
        
        # 缩放到图像尺寸（假设输入256x256）
        bbox = bbox * 256.0
        bbox = torch.clamp(bbox, 0, 256)
        
        return bbox