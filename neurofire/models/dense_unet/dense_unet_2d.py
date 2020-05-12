import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D
from .base import DUNetSkeleton, Xcoder


class Encoder(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        super(Encoder, self).__init__(previous_in_channels, out_channels, kernel_size,
                                      conv_type=ConvELU2D,
                                      pre_output=nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class Decoder(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        super(Decoder, self).__init__(previous_in_channels, out_channels, kernel_size,
                                      conv_type=ConvELU2D,
                                      pre_output=nn.Upsample(scale_factor=2))


class Base(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        super(Base, self).__init__(previous_in_channels, out_channels, kernel_size,
                                   conv_type=ConvELU2D,
                                   pre_output=None)


class Output(Conv2D):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        self.in_channels = sum(previous_in_channels)
        self.out_channels = out_channels
        super(Output, self).__init__(self.in_channels, self.out_channels, kernel_size)


class DUNet(DUNetSkeleton):
    def __init__(self, in_channels, out_channels, N=16, final_activation='auto',
                 return_hypercolumns=False):
        # Build encoders
        encoders = [
            Encoder([in_channels], N, 3),
            Encoder([in_channels, N], 2 * N, 3),
            Encoder([in_channels, N, 2 * N], 4 * N, 3)
        ]
        # Build poolers
        poolers = [nn.MaxPool2d(stride=stride, kernel_size=stride + 1, padding=1)
                   for stride in [2, 4, 8]]
        # Build base
        base = Base([in_channels, N, 2 * N, 4 * N], 4 * N, 3)
        # Build upsamplers
        upsamplers = [nn.Upsample(scale_factor=scale) for scale in [2, 4, 8]]
        # Build decoders
        decoders = [
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N], 2 * N, 3),
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N], N, 3),
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N, N], N, 3)
        ]
        # Build output
        output = Output([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N, N, N], out_channels, 3)
        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax2d()
        # dundundun
        super(DUNet, self).__init__(encoders=encoders,
                                    poolers=poolers,
                                    base=base,
                                    upsamplers=upsamplers,
                                    decoders=decoders,
                                    output=output,
                                    final_activation=final_activation,
                                    return_hypercolumns=return_hypercolumns)

    def forward(self, input_):
        # CREMI loaders are usually 3D, so we reshape if necessary
        if input_.dim() == 5:
            reshape_to_3d = True
            b, c, _0, _1, _2 = list(input_.size())
            assert _0 == 1
            input_ = input_.view(b, c * _0, _1, _2)
        else:
            reshape_to_3d = False
        output = super(DUNet, self).forward(input_)
        if reshape_to_3d:
            b, c, _0, _1 = list(output.size())
            output = output.view(b, c, 1, _0, _1)
        return output
