from ctypes import Union
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from lwctrans.network_architecture.Coformer.light_transformer_block import ConvBlock, ConvNormNonlin, \
    _init_weights


class nnUNetDecoder(nn.Module):
    def __init__(self, num_class, embed_dims, kernel_size, stride, padding):
        super(nnUNetDecoder, self).__init__()
        self.decoders = nn.ModuleList()
        self.up = nn.ModuleList()
        self.seg_heads = nn.ModuleList()

        for i in range(len(kernel_size)):
            self.decoders.append(ConvBlock(embed_dims[i + 1] * 2, embed_dims[i + 1], kernel_size=kernel_size[i],
                                           padding=padding[i]))
            self.up.append(nn.ConvTranspose3d(embed_dims[i], embed_dims[i + 1], kernel_size=stride[i],
                                              stride=stride[i]))
            self.seg_heads.append(nn.Conv3d(embed_dims[i + 1], num_class, kernel_size=1))

        self.apply(_init_weights)

    def forward(self, inputs):
        outputs = []
        x = inputs.pop()
        for i in range(len(inputs)):
            x = self.up[i](x)
            x = torch.cat([x, inputs[-(i + 1)]], dim=1)
            x = self.decoders[i](x)
            outputs.append(self.seg_heads[i](x))

        return outputs[::-1]


class UNetDecoder(nn.Module):
    def __init__(self, num_class, embed_dims, kernel_size, stride, padding):
        super(UNetDecoder, self).__init__()
        self.decoders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.up = nn.ModuleList()
        self.seg_heads = nn.ModuleList()

        for i in range(4):
            self.convs.append(ConvNormNonlin(embed_dims[i], embed_dims[3], kernel_size=1, stride=1, padding=0))

        for i in range(3, len(kernel_size)):
            self.decoders.append(ConvBlock(embed_dims[i + 1] * 2, embed_dims[i + 1], kernel_size=kernel_size[i],
                                           padding=padding[i]))
            self.up.append(nn.ConvTranspose3d(embed_dims[i], embed_dims[i + 1], kernel_size=stride[i],
                                              stride=stride[i]))
            self.seg_heads.append(nn.Conv3d(embed_dims[i + 1], num_class, kernel_size=1))

        self.seg = nn.Conv3d(embed_dims[3], num_class, 1)
        self.fusion_conv = nn.Conv3d(embed_dims[3] * 4, embed_dims[3], 1)

        self.apply(_init_weights)

    def forward(self, inputs):
        outs = []
        tmp = []
        for idx in range(4):
            x = inputs[-(idx + 1)]
            conv = self.convs[idx]
            tmp.append(
                F.interpolate(conv(x), size=inputs[-4].shape[2:], mode='trilinear', align_corners=False)
            )
        x = self.fusion_conv(torch.cat(tmp, dim=1))
        outs.append(self.seg(x))

        for i in range(3, len(inputs) - 1):
            x = self.up[i - 3](x)
            x = torch.cat([x, inputs[-(i + 2)]], dim=1)
            x = self.decoders[i - 3](x)
            outs.append(self.seg_heads[i - 3](x))
        return outs[::-1]
