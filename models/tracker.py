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
# models/tracker.py - 基于JVG架构重构
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import RGBDBackbone
from models.text_encoder import TextEncoder

class MSRM(nn.Module):
    """多源关系建模模块(简化版JVG)"""
    def __init__(self, dim=256, nhead=8, num_layers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim*4, 
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, dim) * 0.02)
        
    def forward(self, lang_tokens, template_tokens, search_tokens):
        """
        Args:
            lang_tokens: [B, L_l, D] 文本特征
            template_tokens: [B, L_z, D] 模板tokens
            search_tokens: [B, L_t, D] 搜索tokens
        """
        B = lang_tokens.shape[0]
        # 拼接所有tokens
        all_tokens = torch.cat([lang_tokens, template_tokens, search_tokens], dim=1)
        L = all_tokens.shape[1]
        
        # 位置编码
        pos = self.pos_embed[:, :L, :].expand(B, -1, -1)
        all_tokens = all_tokens + pos
        
        # Transformer编码
        enhanced = self.transformer(all_tokens)
        
        # 分割输出
        L_l, L_z = lang_tokens.shape[1], template_tokens.shape[1]
        hl = enhanced[:, :L_l, :]
        hz = enhanced[:, L_l:L_l+L_z, :]
        ht = enhanced[:, L_l+L_z:, :]
        
        return hl, hz, ht

class TargetDecoder(nn.Module):
    """目标解码器"""
    def __init__(self, dim=256, nhead=8, num_layers=2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim*4,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.target_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
    def forward(self, ht, text_feat):
        """
        Args:
            ht: [B, L_t, D] 增强后的搜索tokens
            text_feat: [B, D] 文本特征(用于初始化query)
        """
        B = ht.shape[0]
        tq = self.target_query.expand(B, -1, -1)  # [B,1,D]
        
        # 用文本特征增强query
        tq = tq + text_feat.unsqueeze(1)
        
        # 解码
        dec_out = self.decoder(tq, ht)  # [B,1,D]
        return dec_out

class LocalizationHead(nn.Module):
    """定位头(相似度加权+MLP回归)"""
    def __init__(self, dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 4)
        )
        
    def forward(self, dec_out, ht):
        """
        Args:
            dec_out: [B, 1, D] 解码器输出
            ht: [B, L_t, D] 搜索tokens
        Returns:
            bbox: [B, 4] 归一化的[cx,cy,w,h]
        """
        query = dec_out.squeeze(1)  # [B,D]
        
        # 计算相似度注意力
        scores = F.cosine_similarity(query.unsqueeze(1), ht, dim=-1)  # [B,L_t]
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B,L_t,1]
        
        # 加权池化
        target_feat = (attn * ht).sum(dim=1)  # [B,D]
        
        # 回归bbox
        bbox = self.mlp(target_feat)  # [B,4]
        bbox = torch.sigmoid(bbox)  # 归一化到[0,1]
        
        return bbox

class RGBDTextTracker(nn.Module):
    """完整的RGB-D-Text跟踪器(基于JVG架构)"""
    def __init__(self):
        super().__init__()
        
        self.backbone = RGBDBackbone()
        self.text_encoder = TextEncoder()
        
        # RGB-D融合(Early Fusion)
        self.rgbd_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1),  # 256+256->256
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # === 核心模块 ===
        self.msrm = MSRM(dim=256, nhead=8, num_layers=3)
        self.decoder = TargetDecoder(dim=256, nhead=8, num_layers=2)
        self.loc_head = LocalizationHead(dim=256)
        
        # 文本投影
        self.text_proj = nn.Linear(512, 256)
        
    def extract_tokens(self, rgbd_feat):
        """将特征图转换为tokens"""
        B, C, H, W = rgbd_feat.shape
        tokens = rgbd_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return tokens
    
    def forward(self, template_rgb, template_depth, text, search_rgb, search_depth):
        """
        Args:
            template_rgb/depth: [B, C, H, W]
            text: list of strings
            search_rgb/depth: [B, C, H, W]
        """
        # === 1. 特征提取 ===
        temp_rgb, temp_depth = self.backbone(template_rgb, template_depth)
        temp_fused = self.rgbd_fusion(torch.cat([temp_rgb, temp_depth], dim=1))
        
        search_rgb, search_depth = self.backbone(search_rgb, search_depth)
        search_fused = self.rgbd_fusion(torch.cat([search_rgb, search_depth], dim=1))
        
        # === 2. 转换为tokens ===
        temp_tokens = self.extract_tokens(temp_fused)    # [B, L_z, 256]
        search_tokens = self.extract_tokens(search_fused)  # [B, L_t, 256]
        
        # === 3. 文本特征 ===
        text_feat = self.text_encoder(text)  # [B, 512]
        text_proj = self.text_proj(text_feat)  # [B, 256]
        lang_tokens = text_proj.unsqueeze(1)  # [B, 1, 256]
        
        # === 4. MSRM多源关系建模 ===
        hl, hz, ht = self.msrm(lang_tokens, temp_tokens, search_tokens)
        
        # === 5. 目标解码 ===
        dec_out = self.decoder(ht, text_proj)  # [B, 1, 256]
        
        # === 6. 定位 ===
        bbox_norm = self.loc_head(dec_out, ht)  # [B, 4] 归一化
        
        # 转换为图像坐标(假设256x256)
        bbox = bbox_norm * 256.0
        bbox = torch.clamp(bbox, 0, 256)
        
        # 生成伪响应图(用于loss计算)
        response = None
        
        return bbox, response