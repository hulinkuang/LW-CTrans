from functools import partial
from typing import Union, Sequence

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.nn.init import trunc_normal_


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DWMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c', b=B, c=C, d=D, h=H, w=W)
        x = self.fc1(x)
        x = rearrange(x, 'b (d h w) c -> b c d h w', b=B, c=x.shape[-1], d=D, h=H, w=W)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = rearrange(x, 'b (d h w) c -> b c d h w', b=B, c=C, d=D, h=H, w=W)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if dim % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(dim, dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.input_rearrange = rearrange(x, "b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        # self.out_rearrange = rearrange(x, "b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):

        output = rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads)
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim,
                 num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim, data_format="channels_last")
        self.norm2 = norm_layer(dim, data_format="channels_last")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_mixer = SABlock(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout_rate=attn_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c', b=B, c=C, d=D, h=H, w=W)
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'b (d h w) c -> b c d h w', b=B, c=C, d=D, h=H, w=W)
        return x


class GroupConv(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super(GroupConv, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=1e-5)
        self.group_conv3x3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class ConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, Sequence] = 3,
                 stride: Union[int, Sequence] = 1, padding: Union[int, Sequence] = 1, groups=1, drop=0.):
        super(ConvNormNonlin, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm3d(num_features=out_channels)
        self.nonlin = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = self.nonlin(self.norm(x))
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, Sequence] = 3,
                 stride: Union[int, Sequence] = 1, padding: Union[int, Sequence] = 1, groups=1, drop=0.):
        super(BottleNeck, self).__init__()
        hidden_dim = in_channels * 2
        self.conv = nn.Sequential(
            # pw
            nn.Conv3d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv3d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
        )

        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, Sequence] = 3,
                 stride: Union[int, Sequence] = 1, padding: Union[int, Sequence] = 1, groups=1, drop=0.):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvNormNonlin(in_channels, out_channels, kernel_size, stride, padding, groups, drop)
        self.conv2 = ConvNormNonlin(out_channels, out_channels, kernel_size=3, stride=1, padding=1, drop=drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ProjectConvBlock(nn.Module):
    def __init__(self, dim, drop=0.):
        super().__init__()
        self.d_conv = ConvNormNonlin(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=dim,
                                      drop=drop)
        self.h_conv = ConvNormNonlin(dim, dim, kernel_size=(3, 1, 3), padding=(1, 0, 1), groups=dim,
                                      drop=drop)
        self.w_conv = ConvNormNonlin(dim, dim, kernel_size=(3, 3, 1), padding=(1, 1, 0), groups=dim,
                                      drop=drop)
        self.pwconv = ConvNormNonlin(dim * 3, dim, kernel_size=1, stride=1, padding=0, drop=drop)

    def forward(self, x):
        x_d = self.d_conv(x)
        x_h = self.h_conv(x)
        x_w = self.w_conv(x)
        x = torch.cat([x_d, x_h, x_w], dim=1)
        x = self.pwconv(x)
        return x



class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

    def forward(self, x):
        res = x
        x_pooling = [torch.mean(x, dim=i, keepdim=True).expand_as(x) for i in range(2, len(x.shape))]
        x = torch.stack(x_pooling, dim=-1).mean(dim=-1)
        x = torch.sigmoid(x) * res
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv1d(dim // 4, dim, 1),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_ = einops.rearrange(x, 'b c d h w -> b (d h w) c', b=B, c=C, d=D, h=H, w=W)
        B, N, C = x_.shape
        q = self.q(x_).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x_).reshape(B, N, 2, C).permute(2, 0, 3, 1)
        k, v = kv[0], kv[1]
        k = self.se(k).reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn * v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = einops.rearrange(x, 'b (d h w) c -> b c d h w', b=B, c=C, d=D, h=H, w=W)

        return x


class LightTransformerBlock(nn.Module):

    def __init__(self, dim,
                 num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim, data_format="channels_first")
        self.norm2 = norm_layer(dim, data_format="channels_last")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_mixer = PoolingAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            attn_drop=attn_drop, proj_drop=drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = einops.rearrange(x, 'b c d h w -> b (d h w) c', b=B, c=C, d=D, h=H, w=W)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = einops.rearrange(x, 'b (d h w) c -> b c d h w', b=B, c=C, d=D, h=H, w=W)
        return x


def _init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        m.weight = nn.init.kaiming_normal_(m.weight, a=0.01)
        if m.bias is not None:
            m.bias = nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
