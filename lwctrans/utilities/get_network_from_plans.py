import sys
import time

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from lwctrans.utilities.network_initialization import InitWeights_He
from lwctrans.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from lwctrans.network_architecture import SegmentationNetwork
from torch import nn
from thop import profile


def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True,
                           use_nnunet: bool = False):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    if use_nnunet:
        return get_nnunet(plans_manager, dataset_json, configuration_manager, num_input_channels, deep_supervision)

    in_channels = num_input_channels
    num_classes = label_manager.num_segmentation_heads
    img_size = configuration_manager.patch_size
    conv_kernel_size = configuration_manager.conv_kernel_sizes
    pool_op_kernel_sizes = configuration_manager.pool_op_kernel_sizes
    base_num_features = 8 if len(conv_kernel_size) >= 7 else 16
    max_num_features = configuration_manager.unet_max_num_features

    model = SegmentationNetwork(in_channels=in_channels, num_classes=num_classes, img_size=img_size,
                                conv_kernel_size=conv_kernel_size, pool_op_kernel_sizes=pool_op_kernel_sizes,
                                base_num_features=base_num_features, max_num_features=max_num_features,
                                deep_supervision=deep_supervision)

    # x = torch.randn([1] + [1] + list(img_size))
    # compute(model, x)

    return model


def get_nnunet(plans_manager: PlansManager,
               dataset_json: dict,
               configuration_manager: ConfigurationManager,
               num_input_channels: int,
               deep_supervision: bool = True):
    """
        we may have to change this in the future to accommodate other plans -> network mappings

        num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
        trainer rather than inferring it again from the plans here.
        """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model


def compute(model, x):
    model.cuda()
    x = x.cuda()
    flops, params = profile(model, inputs=(x,))
    FLOPs = flops / 1000 ** 3
    # for i in range(50):
    #     model(x)
    # torch.cuda.synchronize()
    # tic1 = time.time()
    # for i in range(200):
    #     model(x)
    # torch.cuda.synchronize()
    # tic2 = time.time()
    # throughput = 200 * 1 / (tic2 - tic1)
    # latency = 1000 * (tic2 - tic1) / 200
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # print(f"throughout = {throughput}")
    # print(f"latency = {latency}ms")
    # print(f"FLOPS = {FLOPs / latency}G")
    sys.exit()
