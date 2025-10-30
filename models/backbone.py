import torch
import torch.nn as nn
import timm

class RGBDBackbone(nn.Module):
    """RGB-D双分支特征提取器（优先本地权重，没本地就下载）"""
    def __init__(self, model_name='resnet50', rgb_ckpt=None, depth_ckpt=None):
        super().__init__()

        # ---------- RGB ----------
        if rgb_ckpt and os.path.exists(rgb_ckpt):
            pretrained_rgb = False
            rgb_checkpoint = rgb_ckpt
        else:
            pretrained_rgb = True
            rgb_checkpoint = None

        self.rgb_encoder = timm.create_model(
            model_name,
            pretrained=pretrained_rgb,
            checkpoint_path=rgb_checkpoint,
            features_only=True,
            out_indices=[2, 3]
        )

        # ---------- Depth ----------
        if depth_ckpt and os.path.exists(depth_ckpt):
            pretrained_depth = False
            depth_checkpoint = depth_ckpt
        else:
            pretrained_depth = True
            depth_checkpoint = None

        self.depth_encoder = timm.create_model(
            model_name,
            pretrained=pretrained_depth,
            checkpoint_path=depth_checkpoint,
            features_only=True,
            out_indices=[2, 3],
            in_chans=1
        )

        # Depth第一层卷积处理单通道权重（如果加载本地权重）
        if depth_checkpoint:
            state_dict = torch.load(depth_checkpoint, map_location='cpu')
            state_dict['conv1.weight'] = state_dict['conv1.weight'].mean(dim=1, keepdim=True)
            self.depth_encoder.load_state_dict(state_dict, strict=False)

        self.feat_dim = 1024
    def forward(self, rgb, depth):
        """
        Args:
            rgb: [B, 3, H, W]
            depth: [B, 1, H, W]
        Returns:
            fused_feat: [B, C, H', W']
        """
        rgb_feats = self.rgb_encoder(rgb)
        depth_feats = self.depth_encoder(depth)
        
        # 取最后一层特征
        rgb_feat = rgb_feats[-1]  # [B, 1024, H/16, W/16]
        depth_feat = depth_feats[-1]
        
        return rgb_feat, depth_feat