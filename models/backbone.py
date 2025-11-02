# import torch
# import torch.nn as nn
# import timm

# class RGBDBackbone(nn.Module):
#     """RGB-D双分支特征提取器（优先本地权重，没本地就下载）"""
#     def __init__(self, model_name='resnet50', rgb_ckpt=None, depth_ckpt=None):
#         super().__init__()

#         # ---------- RGB ----------
#         if rgb_ckpt and os.path.exists(rgb_ckpt):
#             pretrained_rgb = False
#             rgb_checkpoint = rgb_ckpt
#         else:
#             pretrained_rgb = True
#             rgb_checkpoint = None

#         self.rgb_encoder = timm.create_model(
#             model_name,
#             pretrained=pretrained_rgb,
#             checkpoint_path=rgb_checkpoint,
#             features_only=True,
#             out_indices=[2, 3]
#         )

#         # ---------- Depth ----------
#         if depth_ckpt and os.path.exists(depth_ckpt):
#             pretrained_depth = False
#             depth_checkpoint = depth_ckpt
#         else:
#             pretrained_depth = True
#             depth_checkpoint = None

#         self.depth_encoder = timm.create_model(
#             model_name,
#             pretrained=pretrained_depth,
#             checkpoint_path=depth_checkpoint,
#             features_only=True,
#             out_indices=[2, 3],
#             in_chans=1
#         )

#         # Depth第一层卷积处理单通道权重（如果加载本地权重）
#         if depth_checkpoint:
#             state_dict = torch.load(depth_checkpoint, map_location='cpu')
#             state_dict['conv1.weight'] = state_dict['conv1.weight'].mean(dim=1, keepdim=True)
#             self.depth_encoder.load_state_dict(state_dict, strict=False)

#         self.feat_dim = 1024
#     def forward(self, rgb, depth):
#         """
#         Args:
#             rgb: [B, 3, H, W]
#             depth: [B, 1, H, W]
#         Returns:
#             fused_feat: [B, C, H', W']
#         """
#         rgb_feats = self.rgb_encoder(rgb)
#         depth_feats = self.depth_encoder(depth)
        
#         # 取最后一层特征
#         rgb_feat = rgb_feats[-1]  # [B, 1024, H/16, W/16]
#         depth_feat = depth_feats[-1]
        
#         return rgb_feat, depth_feat

#  lite版

# /root/ost/RGBDTextTracker/models/backbone.py
# import os, torch, torch.nn as nn
# import timm
# from .lite_mono import LiteMono   # ← 你的深度 encoder
# import torch.nn.functional as F

# class RGBDBackbone(nn.Module):
#     def __init__(self, lite_ckpt='/root/ost/RGBDTextTracker/pretrained/lite_mono_pretrained.pth'):
#         super().__init__()
#         # RGB 分支：resnet50
#         self.rgb_encoder = timm.create_model(
#             'resnet50', pretrained=True, features_only=True, out_indices=[2,3]
#         )
#         # 深度分支：LiteMono **encoder**（只取特征）
#         self.depth_encoder = LiteMono()   # ← 这就是 depth_encoder
#         if lite_ckpt and os.path.exists(lite_ckpt):
#             self.depth_encoder.load_state_dict(
#                 torch.load(lite_ckpt, map_location='cpu'), strict=False
#             )

#     # def forward(self, rgb, depth):
#     #     rgb_feats   = self.rgb_encoder(rgb)          # list of tensors
#     #     dep_feat    = self.depth_encoder.forward_features(depth)[-1]  # 只拿最后一层
#     #     return rgb_feats, dep_feat
#     def forward(self, rgb, depth):
#         rgb_feat = self.rgb_encoder(rgb)[-1]          # [B,2048,H,W]
#         dep_3ch  = depth.expand(-1, 3, -1, -1)
#         dep_feat = self.depth_encoder.forward_features(dep_3ch)[-1]   # [B,128,H,W]

#         # 统一空间尺寸 & 降维 RGB → 128
#         if dep_feat.shape[-2:] != rgb_feat.shape[-2:]:
#             dep_feat = F.interpolate(dep_feat, size=rgb_feat.shape[-2:], mode='bilinear', align_corners=False)
#         rgb_128 = nn.Conv2d(rgb_feat.shape[1], 128, 1, bias=False).to(rgb_feat.device)(rgb_feat)   # 1×1 降维

#         return rgb_128, dep_feat      # 都是 128 ch → 融合期望 256


# models/backbone.py 报错中
# import os, torch, torch.nn as nn
# import timm
# from .lite_mono import LiteMono
# import torch.nn.functional as F

# class RGBDBackbone(nn.Module):
#     def __init__(self, lite_ckpt='/root/ost/RGBDTextTracker/pretrained/lite_mono_pretrained.pth'):
#         super().__init__()
        
#         # RGB分支
#         self.rgb_encoder = timm.create_model(
#             'resnet50', pretrained=True, features_only=True, out_indices=[3]  # 只取最后一层
#         )
        
#         # Depth分支
#         self.depth_encoder = LiteMono()
#         if lite_ckpt and os.path.exists(lite_ckpt):
#             self.depth_encoder.load_state_dict(
#                 torch.load(lite_ckpt, map_location='cpu'), strict=False
#             )
        
#         # RGB降维：2048→128
#         self.rgb_proj = nn.Conv2d(2048, 128, 1, bias=False)
        
#     def forward(self, rgb, depth):
#         # RGB特征
#         rgb_feat = self.rgb_encoder(rgb)[0]  # [B,2048,H,W]
#         rgb_128 = self.rgb_proj(rgb_feat)    # [B,128,H,W]
        
#         # Depth特征
#         dep_3ch = depth.expand(-1, 3, -1, -1)
#         dep_feat = self.depth_encoder.forward_features(dep_3ch)[-1]  # [B,128,H,W]
        
#         # 对齐空间尺寸
#         if dep_feat.shape[-2:] != rgb_128.shape[-2:]:
#             dep_feat = F.interpolate(
#                 dep_feat, size=rgb_128.shape[-2:], 
#                 mode='bilinear', align_corners=False
#             )
        
#         return rgb_128, dep_feat

#777
# models/backbone.py
import os, torch, torch.nn as nn
import timm
from .lite_mono import LiteMono
import torch.nn.functional as F

class RGBDBackbone(nn.Module):
    def __init__(self, lite_ckpt='/root/ost/RGBDTextTracker/pretrained/lite_mono_pretrained.pth'):
        super().__init__()
        
        # RGB分支：resnet50，取倒数第二层(1024维)
        self.rgb_encoder = timm.create_model(
            'resnet50', pretrained=True, features_only=True, out_indices=[3]
        )
        
        # Depth分支：LiteMono
        self.depth_encoder = LiteMono()
        if lite_ckpt and os.path.exists(lite_ckpt):
            self.depth_encoder.load_state_dict(
                torch.load(lite_ckpt, map_location='cpu'), strict=False
            )
        
        # 投影到统一维度256
        self.rgb_proj = nn.Conv2d(1024, 256, 1, bias=False)    # ← 改为1024输入
        self.depth_proj = nn.Conv2d(128, 256, 1, bias=False)
        
    def forward(self, rgb, depth):
        # RGB特征
        rgb_feat = self.rgb_encoder(rgb)[0]  # [B,1024,H,W]
        rgb_256 = self.rgb_proj(rgb_feat)    # [B,256,H,W]
        
        # Depth特征
        dep_3ch = depth.expand(-1, 3, -1, -1)
        dep_feat = self.depth_encoder.forward_features(dep_3ch)[-1]  # [B,128,H,W]
        dep_256 = self.depth_proj(dep_feat)  # [B,256,H,W]
        
        # 对齐空间尺寸
        if dep_256.shape[-2:] != rgb_256.shape[-2:]:
            dep_256 = F.interpolate(
                dep_256, size=rgb_256.shape[-2:], 
                mode='bilinear', align_corners=False
            )
        
        return rgb_256, dep_256