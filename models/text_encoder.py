import torch
import torch.nn as nn
import clip

class TextEncoder(nn.Module):
    """CLIP文本编码器"""
    def __init__(self, clip_model='ViT-B/32'):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model, device='cuda')
        self.clip_model.eval()  # 冻结CLIP
        
    def forward(self, text_list):
        """
        Args:
            text_list: list of strings
        Returns:
            text_feat: [B, 512]
        """
        with torch.no_grad():
            text_tokens = clip.tokenize(text_list).cuda()
            text_feat = self.clip_model.encode_text(text_tokens)
            text_feat = text_feat.float()
        return text_feat
