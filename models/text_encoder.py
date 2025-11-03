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

# ↑能跑通但单卡
# models/text_encoder.py
# import torch
# import torch.nn as nn
# import clip

# class TextEncoder(nn.Module):
#     """CLIP文本编码器 - 延迟到 forward 时把模型移到正确 device（避免 DataParallel 跨卡问题）"""
#     def __init__(self, clip_model='ViT-B/32', device=None):
#         super().__init__()
#         # ✅ 不把 CLIP 强制放到 cuda:0，先加载到 CPU（或默认）
#         self.clip_model, _ = clip.load(clip_model, device='cpu')  # <- load on CPU
#         self.clip_model.eval()
#         # freeze params
#         for p in self.clip_model.parameters():
#             p.requires_grad = False

#     def forward(self, text_list, device=None):
#         """
#         Args:
#             text_list: list[str]
#             device: torch.device or None (如果 None 会从 first token 推断)
#         Returns:
#             text_feat: [B, D] float32 on same device as input tokens
#         """
#         # tokenize -> CPU numpy -> then move to correct device
#         text_tokens = clip.tokenize(text_list)  # cpu tensor (long)
#         if device is None:
#             device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#         text_tokens = text_tokens.to(device)

#         # 确保模型也在相同 device（一次移动即可）
#         # 注意：把模型整体移动到 device（在 DataParallel 里，每个 replica 会把 module.cuda() 再 replicate）
#         # 这里先把 clip_model 移到 device（model parameters are not trainable so this is fine）
#         if next(self.clip_model.parameters()).device != device:
#             self.clip_model = self.clip_model.to(device)

#         with torch.no_grad():
#             text_feat = self.clip_model.encode_text(text_tokens)
#             text_feat = text_feat.float()
#         return text_feat
