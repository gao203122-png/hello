# ============================================
# 4. models/tracker.py - 主跟踪器
# ============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import RGBDBackbone
from models.fusion import MultiModalFusion
from models.text_encoder import TextEncoder

class RGBDTextTracker(nn.Module):
    """RGB-D-Text多模态跟踪器"""
    def __init__(self):
        super().__init__()
        
        self.backbone = RGBDBackbone()
        self.text_encoder = TextEncoder()
        self.fusion = MultiModalFusion()
        
        # 跟踪头（简单的相关层）
        self.head = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1)  # 输出热力图
        )
        
        # 边界框回归
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # [x, y, w, h]
        )
        
    def forward(self, template_rgb, template_depth, text,
                search_rgb, search_depth):
        """
        Args:
            template_rgb/depth: [B, C, 127, 127] 模板图像
            text: list of strings
            search_rgb/depth: [B, C, 255, 255] 搜索区域
        Returns:
            bbox: [B, 4]
        """
        # 提取模板特征
        temp_rgb_feat, temp_depth_feat = self.backbone(
            template_rgb, template_depth
        )
        text_feat = self.text_encoder(text)
        temp_feat = self.fusion(temp_rgb_feat, temp_depth_feat, text_feat)
        
        # 提取搜索区域特征
        search_rgb_feat, search_depth_feat = self.backbone(
            search_rgb, search_depth
        )
        search_feat = self.fusion(search_rgb_feat, search_depth_feat, text_feat)
        
        # 相关性计算（简化版）
        response_map = self.head(search_feat)
        
        # 边界框预测
        bbox = self.bbox_head(search_feat)
        
        return bbox, response_map
