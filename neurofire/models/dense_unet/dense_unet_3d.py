import torch.nn as nn
from .base import DUNetSkeleton, Xcoder, Gcoder
from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D
from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample
from functools import reduce

#
# TODO make anisotropic sampling optional
#


class AtrousBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_branches=4):
        super(AtrousBlock3D, self).__init__()
        self.branches = nn.ModuleList([Conv3D(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              dilation=(2 ** branch_num))
                                       for branch_num in range(num_branches)])
        self.activation = nn.ELU()

    def forward(self, input):
        # Forward through all branches
        preactivation = reduce(lambda x, y: x + y,
                               [branch(input) for branch in self.branches])
        output = self.activation(preactivation)
        return output


class BNAtrousBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_branches=4):
        super(BNAtrousBlock3D, self).__init__()
        self.branches = nn.ModuleList([Conv3D(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              dilation=(2 ** branch_num))
                                       for branch_num in range(num_branches)])
        self.batchnorms = nn.ModuleList([nn.BatchNorm3d(in_channels)
                                         for _ in range(num_branches)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_branches)])

    def forward(self, input):
        # Forward through all branches
        normed = [bn(input) for bn in self.batchnorms]
        activated = [self.activations[i](norm) for i, norm in enumerate(normed)]
        conved = [self.branches[i](active) for i, active in enumerate(activated)]
        output = reduce(lambda x, y: x + y, [out for out in conved])
        return output


CONV_TYPES = {'vanilla': ConvELU3D,
              'atrous_block': AtrousBlock3D,
              'bn_atrous_block': BNAtrousBlock3D}


# Factory classes bork pickle, so we can't DRY this code.
class EncoderX(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size, sampling_scale,
                 conv_type_key='vanilla'):
        super(EncoderX, self).__init__(previous_in_channels, out_channels, kernel_size,
                                       conv_type=CONV_TYPES[conv_type_key],
                                       pre_output=AnisotropicPool(downscale_factor=sampling_scale))


class EncoderG(Gcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size, sampling_scale,
                 conv_type_key='vanilla'):
        super(EncoderG, self).__init__(previous_in_channels, out_channels, kernel_size,
                                       conv_type=CONV_TYPES[conv_type_key],
                                       pre_output=AnisotropicPool(downscale_factor=sampling_scale))


class DecoderX(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size, sampling_scale,
                 conv_type_key='vanilla'):
        super(DecoderX, self).__init__(previous_in_channels, out_channels, kernel_size,
                                       conv_type=CONV_TYPES[conv_type_key],
                                       pre_output=AnisotropicUpsample(scale_factor=sampling_scale))


class DecoderG(Gcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size, sampling_scale,
                 conv_type_key='vanilla'):
        super(DecoderG, self).__init__(previous_in_channels, out_channels, kernel_size,
                                       conv_type=CONV_TYPES[conv_type_key],
                                       pre_output=AnisotropicUpsample(scale_factor=sampling_scale))


class BaseX(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size,
                 conv_type_key='vanilla'):
        super(BaseX, self).__init__(previous_in_channels, out_channels, kernel_size,
                                    conv_type=CONV_TYPES[conv_type_key],
                                    pre_output=None)


class BaseG(Gcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size,
                 conv_type_key='vanilla'):
        super(BaseG, self).__init__(previous_in_channels, out_channels, kernel_size,
                                    conv_type=CONV_TYPES[conv_type_key],
                                    pre_output=None)


class Output(Conv3D):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        self.in_channels = sum(previous_in_channels)
        self.out_channels = out_channels
        super(Output, self).__init__(self.in_channels, self.out_channels, kernel_size)


ENCODERS = {'X': EncoderX,
            'G': EncoderG}

DECODERS = {'X': DecoderX,
            'G': DecoderG}

BASES = {'X': BaseX,
         'G': BaseG}


# sample_scales: our initial value was 2, but the mala variant is 3
# TODO Replace DUNetSkeleton with DUNet from dense_unet_2d
class DUNet(DUNetSkeleton):
    def __init__(self, in_channels, out_channels,
                 N=16, final_activation='auto',
                 conv_type_key='vanilla', encoder_type_key='X', decoder_type_key='X',
                 base_type_key='X',
                 return_hypercolumns=False, sampling_scale=3):
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.N = N
        self.sampling_scale = sampling_scale
        # Get encoder and decoder types
        Encoder = ENCODERS[encoder_type_key]
        Decoder = DECODERS[decoder_type_key]
        Base = BASES[base_type_key]
        # Build encoders
        encoders = [
            # Hard code the first encoder to vanilla, because this takes the major part of the time
            Encoder([in_channels], N, 3, self.sampling_scale,
                    conv_type_key='vanilla'),
            Encoder([in_channels, N], 2 * N, 3, self.sampling_scale,
                    conv_type_key=conv_type_key),
            Encoder([in_channels, N, 2 * N], 4 * N, 3, self.sampling_scale,
                    conv_type_key=conv_type_key)
        ]

        # different poolings:
        pooling_scales = [self.sampling_scale**(i + 1) for i in range(len(encoders))]

        # Build poolers
        poolers = [AnisotropicPool(downscale_factor=scale) for scale in pooling_scales]
        # Build base
        base = Base([in_channels, N, 2 * N, 4 * N], 4 * N, 3, conv_type_key=conv_type_key)
        # Build upsamplers
        upsamplers = [AnisotropicUpsample(scale_factor=scale) for scale in pooling_scales]
        # Build decoders
        decoders = [
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N], 2 * N, 3, self.sampling_scale,
                    conv_type_key=conv_type_key),
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N], N, 3, self.sampling_scale,
                    conv_type_key=conv_type_key),
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N, N], N, 3, self.sampling_scale,
                    conv_type_key=conv_type_key)
        ]
        # Build output
        output = Output([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N, N, N], out_channels, 3)
        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid()
        assert final_activation != 'Softmax2d'
        assert not return_hypercolumns, "No hypercolumns for 3D networks yet."
        # dundundun
        super(DUNet, self).__init__(encoders=encoders,
                                    poolers=poolers,
                                    base=base,
                                    upsamplers=upsamplers,
                                    decoders=decoders,
                                    output=output,
                                    final_activation=final_activation,
                                    return_hypercolumns=return_hypercolumns)
