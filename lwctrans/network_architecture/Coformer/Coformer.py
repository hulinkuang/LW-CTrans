import torch
import torch.nn as nn
from lwctrans.network_architecture.Coformer.light_transformer_block import ConvNormNonlin, \
    LightTransformerBlock, LayerNorm, ConvBlock, BottleNeck, GroupConv, DWMlp, ProjectConvBlock, TransformerBlock
from lwctrans.network_architecture.Coformer.msca_block import MSCABlock
from lwctrans.network_architecture.Coformer.transformer_block import TransformerBlock
from timm.models.layers import trunc_normal_


class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., num_conv=1, num_trans=1,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=LayerNorm, act_layer=nn.GELU):
        super(BasicBlock, self).__init__()
        self.conv_blocks = nn.Sequential(*[
            ProjectConvBlock(dim=dim // 2, drop=drop)
            for _ in range(num_conv)
        ])
        self.trans_blocks = nn.Sequential(*[
            LightTransformerBlock(
                dim=dim // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer,
                num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop
            )
            for _ in range(num_trans)
        ])

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv_blocks(x1)
        x2 = self.trans_blocks(x2)
        x = torch.cat([x1, x2], dim=1)
        return x


class BasicLayer(nn.Module):

    def __init__(self,
                 block_type,
                 dim,
                 out_dim,
                 depth,
                 num_heads,
                 num_conv=1,
                 num_trans=1,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = None
            if block_type == BasicBlock:
                block = BasicBlock(
                    dim=dim, num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    num_conv=num_conv, num_trans=num_trans,
                    mlp_ratio=mlp_ratio, drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, act_layer=act_layer
                )
            if block_type == MSCABlock:
                block = MSCABlock(dim=dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i], act_layer=act_layer)
            if block_type == GroupConv:
                block = GroupConv(out_channels=dim, head_dim=16)
            if block_type == ProjectConvBlock:
                block = ProjectConvBlock(dim=dim, drop=drop)
            if block_type == TransformerBlock:
                block = TransformerBlock(dim=dim,
                                         num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                         mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i], norm_layer=norm_layer,
                                         act_layer=act_layer)
            if block_type == LightTransformerBlock:
                block = LightTransformerBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop
                )
            self.blocks.append(block)

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(indim=dim, outdim=out_dim,
                                         kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x_down
        return x


class Subsample(nn.Module):
    def __init__(self, indim, outdim, kernel_size, stride, padding):
        super(Subsample, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.norm = nn.BatchNorm3d(outdim)
        self.subsample = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.proj = nn.Conv3d(self.indim, self.outdim, 1)

    def forward(self, x):
        x = self.subsample(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(self, patch_size=(2, 2, 2), in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, D, H, W)
        return x


class Coformer3D(nn.Module):

    def __init__(self,
                 in_chans=1,
                 depths=[2, 1, 1, 3],
                 mlp_ratio=4.,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 num_heads=[1, 2, 4, 8],
                 num_conv=[1, 4, 2, 1],
                 num_trans=[1, 2, 4, 1],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.1,
                 norm_layer=LayerNorm,
                 act_layer=nn.GELU,
                 patch_norm=False,
                 kernel_size=[],
                 stride=[],
                 padding=[],
                 embed_dims=[],
                 block_types=[ProjectConvBlock, BasicBlock, BasicBlock, LightTransformerBlock]
                 ):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        embed_dims.insert(0, in_chans)
        stem_len = len(kernel_size) - 4

        self.stem = nn.ModuleList()
        for i in range(stem_len):
            self.stem.append(ConvBlock(embed_dims[i], embed_dims[i + 1], kernel_size=kernel_size[i],
                                       stride=stride[i], padding=padding[i]))

        self.dims = embed_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(stem_len, stem_len + 4):
            layer = BasicLayer(
                block_type=block_types[i_layer - stem_len],
                dim=self.dims[i_layer],
                out_dim=self.dims[i_layer + 1],
                depth=depths[i_layer - stem_len],
                num_heads=num_heads[i_layer - stem_len],
                kernel_size=self.kernel_size[i_layer],
                stride=self.stride[i_layer],
                padding=self.padding[i_layer],
                num_conv=num_conv[i_layer - stem_len],
                num_trans=num_trans[i_layer - stem_len],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i_layer - stem_len]):sum(depths[:i_layer - stem_len + 1])],
                norm_layer=norm_layer,
                act_layer=act_layer,
                downsample=Subsample,
            )
            self.layers.append(layer)
        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        outs = []
        for stem in self.stem:
            x = stem(x)
            outs.append(x)

        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return outs


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
