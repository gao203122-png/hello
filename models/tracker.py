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
# # models/tracker.py - 终极修复版
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.backbone import RGBDBackbone
# from models.text_encoder import TextEncoder
# import math

# class MSRM(nn.Module):
#     """多源关系建模（动态位置编码）"""
#     def __init__(self, dim=256, nhead=8, num_layers=3):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim, nhead=nhead, dim_feedforward=dim*4, 
#             dropout=0.1, activation="gelu", batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
#         self.dim = dim
        
#     def get_sinusoidal_pos_encoding(self, seq_len, device):
#         """动态生成正弦位置编码（无长度限制）"""
#         position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) * 
#                              (-math.log(10000.0) / self.dim))
#         pos_enc = torch.zeros(seq_len, self.dim, device=device)
#         pos_enc[:, 0::2] = torch.sin(position * div_term)
#         pos_enc[:, 1::2] = torch.cos(position * div_term)
#         return pos_enc.unsqueeze(0)  # [1, L, D]
        
#     def forward(self, lang_tokens, template_tokens, search_tokens):
#         B = lang_tokens.shape[0]
#         all_tokens = torch.cat([lang_tokens, template_tokens, search_tokens], dim=1)
#         L = all_tokens.shape[1]
        
#         # ✅ 动态位置编码，无长度限制
#         pos = self.get_sinusoidal_pos_encoding(L, all_tokens.device).expand(B, -1, -1)
#         all_tokens = all_tokens + pos
        
#         enhanced = self.transformer(all_tokens)

#         L_l, L_z = lang_tokens.shape[1], template_tokens.shape[1]
#         return enhanced[:, :L_l, :], enhanced[:, L_l:L_l+L_z, :], enhanced[:, L_l+L_z:, :]

# class TargetDecoder(nn.Module):
#     def __init__(self, dim=256, nhead=8, num_layers=2):
#         super().__init__()
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=dim, nhead=nhead, dim_feedforward=dim*4,
#             dropout=0.1, activation="gelu", batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
#         self.target_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
#     def forward(self, ht, text_feat):
#         B = ht.shape[0]
#         tq = self.target_query.expand(B, -1, -1) + text_feat.unsqueeze(1)
        
#         dec_out = self.decoder(tq, ht)
        
#         return dec_out

# class LocalizationHead(nn.Module):
#     def __init__(self, dim=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim), nn.ReLU(),
#             nn.Linear(dim, dim//2), nn.ReLU(),
#             nn.Linear(dim//2, 4)
#         )
        
#     def forward(self, dec_out, ht):
#         query = dec_out.squeeze(1)
#         scores = F.cosine_similarity(query.unsqueeze(1), ht, dim=-1)
#         attn = F.softmax(scores, dim=-1).unsqueeze(-1)
#         target_feat = (attn * ht).sum(dim=1)
#         bbox = self.mlp(target_feat)
#         bbox = torch.sigmoid(bbox)
#         return bbox

# class RGBDTextTracker(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.backbone = RGBDBackbone()
#         self.text_encoder = TextEncoder()
        
#         self.rgbd_fusion = nn.Sequential(
#             nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU()
#         )
        
#         # ✅ 核心模块
#         self.msrm = MSRM(dim=256, nhead=8, num_layers=3)
#         self.decoder = TargetDecoder(dim=256, nhead=8, num_layers=2)
#         self.loc_head = LocalizationHead(dim=256)
        
#         self.text_proj = nn.Linear(512, 256)
        
#     def forward(self, template_rgb, template_depth, text, search_rgb, search_depth):
#         # 特征提取
#         temp_rgb, temp_depth = self.backbone(template_rgb, template_depth)
#         temp_fused = self.rgbd_fusion(torch.cat([temp_rgb, temp_depth], dim=1))
        
#         search_rgb, search_depth = self.backbone(search_rgb, search_depth)
#         search_fused = self.rgbd_fusion(torch.cat([search_rgb, search_depth], dim=1))
        
#         # 转tokens
#         temp_tokens = temp_fused.flatten(2).transpose(1, 2)
#         search_tokens = search_fused.flatten(2).transpose(1, 2)

#         # 文本
#         text_feat = self.text_encoder(text)
#         text_proj = self.text_proj(text_feat)
#         lang_tokens = text_proj.unsqueeze(1)
        
#         # ✅ MSRM（必须运行）
#         hl, hz, ht = self.msrm(lang_tokens, temp_tokens, search_tokens)
        
#         # ✅ Decoder
#         dec_out = self.decoder(ht, text_proj)
        
#         # 定位
#         bbox_norm = self.loc_head(dec_out, ht)
#         bbox = bbox_norm * 256.0
#         bbox = torch.clamp(bbox, 0, 256)
        
#         return bbox, None


# models/tracker.py - 终极修复版（融合VP+UniMod+JVG）
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.backbone import RGBDBackbone
# from models.text_encoder import TextEncoder
# import math

# class VisualPrompt(nn.Module):
#     """可学习视觉提示（增强特征表达）"""
#     def __init__(self, num_prompts=10, dim=256):
#         super().__init__()
#         self.prompt = nn.Parameter(torch.randn(num_prompts, dim) * 0.02)
    
#     def forward(self, tokens):
#         B = tokens.shape[0]
#         p = self.prompt.unsqueeze(0).expand(B, -1, -1)
#         return torch.cat([p, tokens], dim=1)

# class MSRM(nn.Module):
#     """多源关系建模（修复版）"""
#     def __init__(self, dim=256, nhead=8, num_layers=2):  # ← 减少层数加速
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim, nhead=nhead, dim_feedforward=dim*2,  # ← 减小FFN
#             dropout=0.1, activation="gelu", batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
#         self.dim = dim
        
#     def get_sinusoidal_pos_encoding(self, seq_len, device):
#         position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) * 
#                              (-math.log(10000.0) / self.dim))
#         pos_enc = torch.zeros(seq_len, self.dim, device=device)
#         pos_enc[:, 0::2] = torch.sin(position * div_term)
#         pos_enc[:, 1::2] = torch.cos(position * div_term)
#         return pos_enc.unsqueeze(0)
        
#     def forward(self, lang_tokens, template_tokens, search_tokens):
#         B = lang_tokens.shape[0]
#         all_tokens = torch.cat([lang_tokens, template_tokens, search_tokens], dim=1)
#         L = all_tokens.shape[1]
        
#         pos = self.get_sinusoidal_pos_encoding(L, all_tokens.device).expand(B, -1, -1)
#         all_tokens = all_tokens + pos
#         enhanced = self.transformer(all_tokens)
        
#         L_l, L_z = lang_tokens.shape[1], template_tokens.shape[1]
#         return enhanced[:, :L_l, :], enhanced[:, L_l:L_l+L_z, :], enhanced[:, L_l+L_z:, :]

# class CorrelationModule(nn.Module):
#     """相关性匹配模块（关键改进）"""
#     def __init__(self, dim=256):
#         super().__init__()
#         self.proj = nn.Conv2d(dim, dim//2, 1)
        
#     def forward(self, template_feat, search_feat):
#         """
#         Args:
#             template_feat: [B, C, H, W]
#             search_feat: [B, C, H, W]
#         Returns:
#             corr_map: [B, H*W, H, W] 相关性图
#         """
#         B, C, Ht, Wt = template_feat.shape
#         _, _, Hs, Ws = search_feat.shape
        
#         # 降维
#         template_feat = self.proj(template_feat)  # [B, C/2, Ht, Wt]
#         search_feat = self.proj(search_feat)      # [B, C/2, Hs, Ws]
        
#         # 展平
#         template_flat = template_feat.view(B, -1, Ht*Wt)  # [B, C/2, Ht*Wt]
#         search_flat = search_feat.view(B, -1, Hs*Ws)      # [B, C/2, Hs*Ws]
        
#         # 计算相关性
#         corr = torch.matmul(template_flat.transpose(1, 2), search_flat)  # [B, Ht*Wt, Hs*Ws]
#         corr = corr.view(B, Ht*Wt, Hs, Ws)
        
#         return corr

# class RGBDTextTracker(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.backbone = RGBDBackbone()
#         self.text_encoder = TextEncoder()
        
#         self.rgbd_fusion = nn.Sequential(
#             nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU()
#         )
        
#         # === 核心改进 ===
#         self.visual_prompt = VisualPrompt(num_prompts=10, dim=256)
#         self.msrm = MSRM(dim=256, nhead=8, num_layers=2)
#         self.correlation = CorrelationModule(dim=256)
        
#         self.text_proj = nn.Linear(512, 256)
        
#         # === 关键：分离的bbox回归头 ===
#         self.bbox_head = nn.Sequential(
#             nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 4, 1)  # 输出[cx,cy,w,h]的偏移量
#         )
        
#         # 响应图头
#         self.response_head = nn.Sequential(
#             nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 1, 1), nn.Sigmoid()
#         )
        
#     def forward(self, template_rgb, template_depth, text, search_rgb, search_depth):
#         B = template_rgb.shape[0]
        
#         # === 1. 特征提取 ===
#         temp_rgb, temp_depth = self.backbone(template_rgb, template_depth)
#         temp_fused = self.rgbd_fusion(torch.cat([temp_rgb, temp_depth], dim=1))  # [B,256,H,W]
        
#         search_rgb, search_depth = self.backbone(search_rgb, search_depth)
#         search_fused = self.rgbd_fusion(torch.cat([search_rgb, search_depth], dim=1))
        
#         # === 2. 相关性匹配（关键） ===
#         Ht, Wt = temp_fused.shape[-2:]
#         Hs, Ws = search_fused.shape[-2:]
        
#         # 全局池化模板作为kernel
#         template_kernel = F.adaptive_avg_pool2d(temp_fused, 1)  # [B,256,1,1]
        
#         # 加权搜索特征
#         weighted_search = search_fused * template_kernel  # 广播相乘
        
#         # === 3. MSRM增强（简化） ===
#         search_tokens = weighted_search.flatten(2).transpose(1, 2)  # [B, Hs*Ws, 256]
#         search_tokens = self.visual_prompt(search_tokens)  # 添加prompt
        
#         text_feat = self.text_encoder(text)  # [B, 512]
#         text_proj = self.text_proj(text_feat).unsqueeze(1)  # [B, 1, 256]
        
#         # 简化：只增强搜索区域
#         _, _, enhanced_search = self.msrm(text_proj, text_proj, search_tokens[:, :Hs*Ws, :])
        
#         # 转回特征图
#         enhanced_search_feat = enhanced_search.transpose(1, 2).view(B, 256, Hs, Ws)
        
#         # === 4. 预测 ===
#         response_map = self.response_head(enhanced_search_feat)  # [B,1,Hs,Ws]
#         bbox_map = self.bbox_head(enhanced_search_feat)  # [B,4,Hs,Ws]
        
#         # === 5. 从响应图提取bbox（修复版） ===
#         # 找到响应最大位置
#         response_flat = response_map.view(B, -1)
#         max_idx = response_flat.argmax(dim=1)
        
#         # 提取对应bbox
#         bbox_flat = bbox_map.view(B, 4, -1)
#         bbox_offset = torch.stack([bbox_flat[i, :, max_idx[i]] for i in range(B)])  # [B,4]
        
#         # === 关键修复：转换为绝对坐标 ===
#         # bbox_offset是相对偏移量，需要转换为绝对坐标
#         grid_size = Hs
#         max_y = max_idx // Ws
#         max_x = max_idx % Ws
        
#         # 转换为中心点坐标（归一化到0-1）
#         cx = (max_x.float() + bbox_offset[:, 0]) / Ws
#         cy = (max_y.float() + bbox_offset[:, 1]) / Hs
#         w = torch.sigmoid(bbox_offset[:, 2])  # 归一化宽高
#         h = torch.sigmoid(bbox_offset[:, 3])
        
#         bbox_norm = torch.stack([cx, cy, w, h], dim=1)
        
#         # ✅ 修复：转换为输入图像尺度（256x256）
#         bbox = bbox_norm.clone()
#         bbox[:, 0] = bbox_norm[:, 0] * 256 - bbox_norm[:, 2] * 256 / 2  # x = cx - w/2
#         bbox[:, 1] = bbox_norm[:, 1] * 256 - bbox_norm[:, 3] * 256 / 2  # y = cy - h/2
#         bbox[:, 2] = bbox_norm[:, 2] * 256  # w
#         bbox[:, 3] = bbox_norm[:, 3] * 256  # h
        
#         bbox = torch.clamp(bbox, 0, 256)
        
#         return bbox, response_map

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import RGBDBackbone
from models.text_encoder import TextEncoder
from models.fusion import MultiModalFusion

class DepthGuidedAttention(nn.Module):
    """深度引导的空间注意力（充分利用depth）"""
    def __init__(self, dim=256):
        super().__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, padding=1),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(),
            nn.Conv2d(dim//2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat, depth_feat):
        """
        用深度特征生成空间注意力，引导RGB特征
        """
        depth_attn = self.depth_conv(depth_feat)  # [B, 1, H, W]
        return rgb_feat * depth_attn + rgb_feat  # 残差连接

class TextGuidedQuery(nn.Module):
    """文本引导的查询生成（充分利用text）"""
    def __init__(self, text_dim=512, visual_dim=256):
        super().__init__()
        self.text_to_query = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim)
        )
        self.query_attention = nn.MultiheadAttention(
            visual_dim, num_heads=8, batch_first=True
        )
        
    def forward(self, visual_tokens, text_feat):
        """
        用文本生成query，在视觉特征中查找目标
        """
        query = self.text_to_query(text_feat).unsqueeze(1)  # [B, 1, 256]
        
        # 文本query查询视觉特征
        attn_out, attn_weights = self.query_attention(
            query, visual_tokens, visual_tokens
        )
        return attn_out, attn_weights

class CorrelationLayer(nn.Module):
    """模板匹配相关层"""
    def __init__(self, dim=256):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim//2, 1)
        
    def forward(self, template, search):
        """
        计算模板和搜索区域的相关性
        Args:
            template: [B, C, Ht, Wt]
            search: [B, C, Hs, Ws]
        Returns:
            correlation_map: [B, Ht*Wt, Hs, Ws]
        """
        B, C, Ht, Wt = template.shape
        Hs, Ws = search.shape[-2:]
        
        template = self.conv(template)
        search = self.conv(search)
        
        # Reshape for correlation
        template_flat = template.view(B, -1, Ht * Wt)  # [B, C/2, Ht*Wt]
        search_flat = search.view(B, -1, Hs * Ws)      # [B, C/2, Hs*Ws]
        
        # Correlation
        corr = torch.matmul(template_flat.transpose(1, 2), search_flat)  # [B, Ht*Wt, Hs*Ws]
        corr = corr.view(B, Ht * Wt, Hs, Ws)
        
        return corr

class RGBDTextTracker(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = RGBDBackbone()
        self.text_encoder = TextEncoder()
        self.fusion = MultiModalFusion(visual_dim=256, text_dim=512)
        
        # ===== 充分利用三模态的模块 =====
        # 1. 深度引导注意力
        self.depth_guided_attn = DepthGuidedAttention(dim=256)
        
        # 2. 文本引导查询
        self.text_guided_query = TextGuidedQuery(text_dim=512, visual_dim=256)
        
        # 3. 相关性匹配
        self.correlation = CorrelationLayer(dim=256)
        
        # 4. bbox预测头（基于相关性+文本引导特征）
        self.bbox_head = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),  # 融合相关性特征
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 4),  # [x, y, w, h] 直接预测256尺度
            nn.Sigmoid()  # 限制范围到 [0,1]
        )
        
        # 5. 响应图头（用于可视化和辅助定位）
        self.response_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        
    def forward(self, template_rgb, template_depth, text, search_rgb, search_depth):
        B = template_rgb.shape[0]
        
        # ===== 1. 提取backbone特征 =====
        temp_rgb_feat, temp_depth_feat = self.backbone(template_rgb, template_depth)
        search_rgb_feat, search_depth_feat = self.backbone(search_rgb, search_depth)
        
        # ===== 2. 深度引导的RGB特征增强 =====
        temp_rgb_enhanced = self.depth_guided_attn(temp_rgb_feat, temp_depth_feat)
        search_rgb_enhanced = self.depth_guided_attn(search_rgb_feat, search_depth_feat)
        
        # ===== 3. 文本特征提取 =====
        text_feat = self.text_encoder(text)  # [B, 512]
        
        # ===== 4. 三模态融合 =====
        temp_fused = self.fusion(temp_rgb_enhanced, temp_depth_feat, text_feat)
        search_fused = self.fusion(search_rgb_enhanced, search_depth_feat, text_feat)
        
        # ===== 5. 模板匹配 =====
        corr_map = self.correlation(temp_fused, search_fused)  # [B, Ht*Wt, Hs, Ws]
        
        # 取最大相关性作为响应
        response = corr_map.max(dim=1, keepdim=True)[0]  # [B, 1, Hs, Ws]
        
        # ===== 6. 文本引导的特征查询 =====
        H, W = search_fused.shape[-2:]
        search_tokens = search_fused.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        text_query, _ = self.text_guided_query(search_tokens, text_feat)  # [B, 1, 256]
        
        # 将文本query广播到空间维度
        text_spatial = text_query.transpose(1, 2).view(B, 256, 1, 1).expand(-1, -1, H, W)
        
        # ===== 7. 融合所有信息预测bbox =====
        final_feat = torch.cat([search_fused, text_spatial], dim=1)  # [B, 512, H, W]
        bbox = self.bbox_head(final_feat)  # [B, 4]
        bbox = torch.sigmoid(bbox)

        # 响应图
        response_map = self.response_head(search_fused)
        
        return bbox, response_map