# ---------------------------------------------------UCaps3D---------------------
# CoTr
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LWCTrans.network_architecture.neural_network import SegmentationNetwork
from LWCTrans.network_architecture import Coformer3D
from LWCTrans.network_architecture import UNetDecoder as Decoder
from torch.nn.init import trunc_normal_


class InitWeights_He(object):

    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
            module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class U_ResTran3D(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=True,
                 in_channels=1,
                 base_num_features=32, pool_op_kernel_sizes=None, conv_kernel_size=None, max_num_features=320, ):
        super(U_ResTran3D, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        embed_dims = [min(base_num_features * 2 ** i, max_num_features) for i in range(len(conv_kernel_size))]
        output_features = base_num_features
        stride = [[1, 1, 1]] + pool_op_kernel_sizes
        padding = [[1 if i == 3 else 0 for i in krnl] for krnl in conv_kernel_size]

        self.backbone = Coformer3D(in_chans=in_channels,
                                   kernel_size=conv_kernel_size,
                                   stride=stride, padding=padding,
                                   embed_dims=embed_dims)

        self.decoder = Decoder(num_class=num_classes, embed_dims=embed_dims[::-1],
                               kernel_size=conv_kernel_size[:0:-1],
                               stride=stride[:0:-1], padding=padding[:0:-1])
        total = sum([param.nelement() for param in self.backbone.parameters()])
        print('  + Number of Backbone Params: %.2f(e6)' % (total / 1e6))
        self.apply(InitWeights_He())

    def forward(self, inputs):
        # # %%%%%%%%%%%%% LWCTrans
        B, C, D, H, W = inputs.shape
        x = inputs
        x = self.backbone(x)
        x = self.decoder(x)
        return x


class ResTranUnet(SegmentationNetwork):
    """
    ResTran-3D Unet
    """

    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False,
                 in_channels=1,
                 base_num_features=32, pool_op_kernel_sizes=None, conv_kernel_size=None, max_num_features=320,
                 deep_supervision=False):
        super().__init__()
        self.do_ds = False
        self.U_ResTran3D = U_ResTran3D(norm_cfg, activation_cfg, img_size, num_classes, weight_std,
                                       in_channels=in_channels,
                                       base_num_features=base_num_features, pool_op_kernel_sizes=pool_op_kernel_sizes,
                                       conv_kernel_size=conv_kernel_size,
                                       max_num_features=max_num_features)  # U_ResTran3D

        if weight_std == False:
            self.conv_op = nn.Conv3d
        if norm_cfg == 'BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg == 'SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg == 'GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg == 'IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x):
        seg_output = self.U_ResTran3D(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]
