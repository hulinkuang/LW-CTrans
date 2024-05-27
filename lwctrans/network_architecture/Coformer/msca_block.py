import torch
import torch.nn as nn
from timm.models.layers import DropPath


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, (3, 1, 7), padding=(1, 0, 3), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (3, 7, 1), padding=(1, 3, 0), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (3, 1, 11), padding=(1, 0, 5), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (3, 11, 1), padding=(1, 5, 0), groups=dim)

        self.conv2_1 = nn.Conv3d(
            dim, dim, (3, 1, 21), padding=(1, 0, 10), groups=dim)
        self.conv2_2 = nn.Conv3d(
            dim, dim, (3, 21, 1), padding=(1, 10, 0), groups=dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class MSCABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 ):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        return x


if __name__ == '__main__':
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    x = torch.randn(2, 128, 16, 80, 80).cuda()
    model = MSCABlock(128).cuda()
    out = model(x)
    print(out.shape)

