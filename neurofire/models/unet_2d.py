from __future__ import print_function, division

import torch
import torch.nn as nn

# Use inferno ConvELU2D which implement 'SAME' padding and use exp linear units
# as well as initialization ('OrthogonalWeightsZeroBias' -> weights are initialized
# with orthogonal, bias with zeros)
from inferno.extensions.layers.convolutional import ConvELU2D

#
# TODO batchnorm and different up-scaling schemes
# TODO affinities and appropriate final activation (make parameter ?!)
#


class DownscaleLayer(nn.Module):
    """
    Down-scale block of 2d unet
    """

    def __init__(self, in_size, out_size, kernel_size=3):
        super(DownscaleLayer, self).__init__()
        # ConvELU2D already applies a ELU, so we don't need another RELU
        self.layer = nn.Sequential(
            ConvELU2D(in_size, out_size, kernel_size),
            ConvELU2D(out_size, out_size, kernel_size),
        )

    def forward(self, x):
        return self.layer(x)


class UpscaleLayer(nn.Module):
    """
    Up-scale block of 2d unet
    """

    def __init__(self, in_size, out_size, kernel_size=3):
        super(UpscaleLayer, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Sequential(
            # NOTE we need in_size here because we are concatenating
            ConvELU2D(in_size, out_size, kernel_size),
            ConvELU2D(out_size, out_size, kernel_size),
        )

    def forward(self, x, skip_input):
        up = self.up(x)
        # NOTE we use 'same' convolutions that's why 'up' and 'skip_input' have the same size
        assert up.size() == skip_input.size()
        return self.conv(torch.cat([up, skip_input], 1))


class UNet2D(nn.Module):
    """
    2d-unet for binary segmentation.
    """

    def __init__(self, n_channels, n_out_channels, n_scale=4, n_features_begin=64):
        super(UNet2D, self).__init__()

        self.n_scale = n_scale

        # hold downscale layers, upscale layers and poolings as module lists
        # to be properly indexed
        self.downscale_layers = nn.ModuleList()
        self.upscale_layers = nn.ModuleList()
        self.poolings = nn.ModuleList()

        n_features = n_features_begin
        n_in = n_channels

        # downscale layers
        for scale in range(self.n_scale):
            self.downscale_layers.append(
                DownscaleLayer(n_in, n_features)
            )
            n_in = n_features
            n_features *= 2
            self.poolings.append(nn.MaxPool2d(2))

        # lowest resolution layer
        self.downscale_layers.append(DownscaleLayer(n_in, n_features))
        n_in = n_features
        n_features //= 2

        # upscale layers
        for scale in range(self.n_scale):
            self.upscale_layers.append(
                UpscaleLayer(n_in, n_features)
            )
            n_in = n_features
            n_features //= 2

        # last 1 x 1 conv layer
        self.last = nn.Conv2d(n_in, n_out_channels, 1)

        # Now, the final activation depends on what you want to do. If you're doing
        # semantic segmentation like in Cityscapes, you'd have 19 categorical output classes,
        # in which case you'd need a softmax. If you're doing a binary segmentation with just one
        # output channel, you need a sigmoid. But: if you're overloading the channel axis for
        # z-context, you're still gonna use sigmoid. Assuming you're not doing the latter,
        if n_out_channels == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax2d()

    # the forward pass - defining the connectivity of layers
    def forward(self, x):
        out_down = []

        out = x
        # apply the downscale layers
        for scale in range(self.n_scale):

            # debug out
            # print("Downscaling layer", scale, "in-shape:", out.size())

            out = self.downscale_layers[scale](out)
            out_down.append(out)  # FIXME do we need to make a copy here somehow ?!
            out = self.poolings[scale](out)

            # debug out
            # print("Downscaling layer", scale, "out-shape:", out.size())

        # apply the lowest res layer
        out = self.downscale_layers[-1](out)

        # debug out
        # print("Lowest layer out-shape:", out.size())
        # print("Saved down outputs")
        # for xx in out_down:
        #     print(xx.size())

        # apply the upscale layers
        for scale in range(self.n_scale):

            # debug out
            # print("Upscaling layer", scale, "in-shape:", out.size())

            # we need to get the correct downscale layer
            down_scale = self.n_scale - scale - 1

            skip_input = out_down[down_scale]

            # debug out
            # print("Concatenating down-scale layer", down_scale)
            # print("with shape", skip_input.size())

            out = self.upscale_layers[scale](out, skip_input)

            # debug out
            # print("Upscaling layer", scale, "out-shape:", out.size())

        return self.final_activation(self.last(out))
