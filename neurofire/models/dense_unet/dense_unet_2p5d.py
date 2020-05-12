import torch.nn as nn
from .base import DUNetSkeleton
from .dense_unet_2d import Encoder, Decoder, Base, Output
from inferno.extensions.layers.reshape import As2D, As3D


# TODO Replace DUNetSkeleton with DUNet from dense_unet_2d
class DUNet(DUNetSkeleton):
    def __init__(self, in_channels, out_channels, z_channels,
                 N=16, final_activation='auto', return_hypercolumns=False):
        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_slices = z_channels
        self.N = N
        # Recompute in channels and out channels
        in_channels = self.z_slices * self.in_channels
        out_channels = self.z_slices * self.out_channels
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
            final_activation = nn.Sigmoid() if self.out_channels == 1 else nn.Softmax2d()
        assert not return_hypercolumns, "No hypercolumns for 2.5D networks yet."
        # dundundun
        super(DUNet, self).__init__(encoders=encoders,
                                    poolers=poolers,
                                    base=base,
                                    upsamplers=upsamplers,
                                    decoders=decoders,
                                    output=output,
                                    final_activation=final_activation,
                                    return_hypercolumns=return_hypercolumns)
        # Build reshape layers
        self.as_2d = As2D(z_as_channel=True)
        self.as_3d = As3D(channel_as_z=True, num_channels_or_num_z_slices=self.out_channels)

    # noinspection PyCallingNonCallable
    def forward(self, input_):
        # Make sure input has the right shape
        assert input_.dim() == 5
        assert input_.size(2) == self.z_slices
        # We're expecting 3D inputs - so we convert to 2D
        input_2d = self.as_2d(input_)
        output_2d = super(DUNet, self).forward(input_2d)
        output = self.as_3d(output_2d)
        return output
