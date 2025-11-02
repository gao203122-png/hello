import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos

class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x)
        x = self.bn1(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x

# v1.4
class LGFI(nn.Module):
    """
    Local-Global Feature Interaction with MCA (纯 4D 版)
    - 位置编码直接加在 (B,C,H,W)
    - 通道归一化用 channels_first
    - gamma 正确广播到 (1,C,1,1)
    - Inverted Bottleneck 仍走 channels_last 分支
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()
        self.dim = dim

        # 位置编码：(B,C,H,W)
        self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None

        # 4D 注意力分支
        self.norm_mca = LayerNorm(self.dim, eps=1e-6, data_format="channels_first")
        self.gamma_mca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.mca = MCA(dim=dim)

        # Inverted Bottleneck（channels_last）
        self.norm = LayerNorm(self.dim, eps=1e-6)   # channels_last
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: (B,C,H,W)
        identity = x
        B, C, H, W = x.shape

        # 位置编码
        if self.pos_embd is not None:
            x = x + self.pos_embd(B, H, W)  # (B,C,H,W)

        # 4D 注意力
        x_norm = self.norm_mca(x)
        y = self.mca(x_norm)  # (B,C,H,W)
        # 形状强校验（正常不会触发）
        if y.shape != x.shape:
            raise RuntimeError(f"LGFI: MCA 输出 {y.shape} 与输入 {x.shape} 不一致")

        # 按通道维广播 gamma
        if self.gamma_mca is not None:
            x = x + self.gamma_mca.view(1, -1, 1, 1) * y
        else:
            x = x + y

        # Inverted Bottleneck（与原版一致）
        x = x.permute(0, 2, 3, 1)          # (B,C,H,W) -> (B,H,W,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x             # (B,H,W,C)，沿 C 广播
        x = x.permute(0, 3, 1, 2)          # (B,H,W,C) -> (B,C,H,W)

        # 残差
        x = identity + self.drop_path(x)
        return x

class LiteMono(nn.Module):
    def __init__(self, model='lite-mono', in_chans=3, **k):
        super().__init__()
        # ===== 只留 encoder 部分 =====
        self.dims = [48, 80, 128]          # 根据 model 选
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, self.dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dims[0]), nn.ReLU(inplace=True),
            nn.Conv2d(self.dims[0], self.dims[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dims[0]), nn.ReLU(inplace=True),
        )
        self.downsample_layers.append(stem)

        for i in range(1, 3):
            down = nn.Sequential(
                nn.Conv2d(self.dims[i-1], self.dims[i], 3, 2, 1, bias=False),
                nn.BatchNorm2d(self.dims[i]), nn.ReLU(inplace=True),
            )
            self.downsample_layers.append(down)

        # 3 个 stage（只留 LGFI/DilatedConv）
        self.stages = nn.ModuleList()
        depth = [4, 4, 10]          # lite-mono
        for i in range(3):
            blocks = []
            for j in range(depth[i]):
                blocks.append(
                    LGFI(self.dims[i]) if j > depth[i]-2 else   # 最后 2 层用 LGFI
                    DilatedConv(self.dims[i], k=3, dilation=1+((j)%3), drop_path=0.))
            self.stages.append(nn.Sequential(*blocks))

    def forward_features(self, x):
        x = (x - 0.45) / 0.225
        feats = []
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)      # [1/2, 1/4, 1/8]
        return feats             # 只返回特征图列表



# v1.4 残差 + LayerScale + DropPath + LayerNorm
import math
import torch
from torch import nn

__all__ = ["MCAGate", "MCALayer", "MCA"]

class StdPool(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        return std.view(b, c, 1, 1)

class MCAGate(nn.Module):
    def __init__(self, k_size: int, pool_types=('avg', 'std'), dropout_p=0.1):
        super().__init__()
        assert k_size >= 3 and k_size % 2 == 1
        self.pools = nn.ModuleList()
        for p in pool_types:
            if p == "avg":
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif p == "std":
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size),
                              stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)
        self.weight = nn.Parameter(torch.zeros(len(pool_types)))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        w = torch.sigmoid(self.weight)
        base = sum(w[i] * feats[i] for i in range(len(feats)))

        if len(feats) == 2:
            base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

        y = base.permute(0, 3, 2, 1).contiguous()
        y = self.conv(y)
        y = self.dropout(y)
        y = y.permute(0, 3, 2, 1).contiguous()
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class MCALayer(nn.Module):
    def __init__(self, channels: int, no_spatial: bool = False, drop_path=0.1):
        super().__init__()
        lambd, gamma = 1.5, 1.0
        temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
        k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(k_spatial)

        # === LayerNorm + LayerScale + DropPath ===
        self.norm = nn.LayerNorm(channels)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones(channels))
        self.drop_path = nn.Dropout(drop_path)

    def forward(self, x):
        shortcut = x
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            out = (x_c + x_h + x_w) / 3.0
        else:
            out = (x_h + x_w) / 2.0

        # === 残差 + LayerScale + DropPath + LayerNorm ===
        out = self.drop_path(out)
        out = self.layer_scale.view(1, -1, 1, 1) * out
        out = out + shortcut
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out

class MCA(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.layer = MCALayer(channels=dim, no_spatial=False)

    def forward(self, x):
        assert x.dim() == 4
        y = self.layer(x)
        if y.shape != x.shape:
            raise RuntimeError(f"MCA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
        return y