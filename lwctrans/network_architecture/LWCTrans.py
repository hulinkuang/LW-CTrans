# ---------------------------------------------------UCaps3D---------------------
# LWCTrans
# ------------------------------------------------------------------------
import torch.nn as nn
from lwctrans.network_architecture.Coformer import Coformer3D
from lwctrans.network_architecture.Coformer import UNetDecoder as Decoder
from torch.nn.init import trunc_normal_


class InitWeights_He(object):

    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
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


class LWCTrans(nn.Module):
    def __init__(self, in_channels=1, num_classes=None,
                 conv_kernel_size=None, pool_op_kernel_sizes=None,
                 base_num_features=32, max_num_features=320, ):
        super(LWCTrans, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        embed_dims = [min(base_num_features * 2 ** i, max_num_features) for i in range(len(conv_kernel_size))]
        output_features = base_num_features
        stride = pool_op_kernel_sizes
        padding = [[1 if i == 3 else 0 for i in krnl] for krnl in conv_kernel_size]

        self.backbone = Coformer3D(in_chans=in_channels, kernel_size=conv_kernel_size, stride=stride, padding=padding,
                                   embed_dims=embed_dims)

        self.decoder = Decoder(num_class=num_classes, embed_dims=embed_dims[::-1],
                               kernel_size=conv_kernel_size[:0:-1],
                               stride=stride[:0:-1], padding=padding[:0:-1])
        backbone = sum([param.nelement() for param in self.backbone.parameters()])
        total = sum([param.nelement() for param in self.parameters()])
        print('  + Number of Backbone Params: %.2f(e6) M' % (backbone / 1e6))
        print('  + Number of Total Params: %.2f(e6) M' % (total / 1e6))
        self.apply(InitWeights_He())

    def forward(self, inputs):
        # # %%%%%%%%%%%%% LWCTrans
        B, C, D, H, W = inputs.shape
        x = inputs
        x = self.backbone(x)
        x = self.decoder(x)
        return x


class SegmentationNetwork(nn.Module):
    """
    All Segmentation Networks
    """

    def __init__(self, in_channels=1, num_classes=None, img_size=None,
                 conv_kernel_size=None, pool_op_kernel_sizes=None, base_num_features=32, max_num_features=320,
                 deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.network = LWCTrans(in_channels=in_channels, num_classes=num_classes,
                                conv_kernel_size=conv_kernel_size,
                                pool_op_kernel_sizes=pool_op_kernel_sizes,
                                base_num_features=base_num_features, max_num_features=max_num_features)

    def forward(self, x):
        seg_output = self.network(x)
        if self.deep_supervision:
            if not isinstance(seg_output, list) and not isinstance(seg_output, tuple):
                return [seg_output]
            else:
                return seg_output
        else:
            if not isinstance(seg_output, list) and not isinstance(seg_output, tuple):
                return seg_output
            else:
                return seg_output[0]
