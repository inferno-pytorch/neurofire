from __future__ import print_function, division

import torch
import torch.nn as nn

# Use inferno ConvELU2D which implement 'SAME' padding and use exp linear units
# as well as initialization ('OrthogonalWeightsZeroBias' -> weights are initialized
# with orthogonal, bias with zeros)
from inferno.extensions.layers.convolutional import ConvELU2D


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


# TODO batchnorm and different up-scaling schemes

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

    def crop_skip_input(self, skip_input, target_size):
        """
        Crop the skip input to correct size
        """
        _, _, skip_width, skip_height = skip_input.size()
        _, _, target_width, target_height = target_size
        width_offset = (skip_width - target_width) // 2
        height_offset = (skip_height - target_height) // 2
        return skip_input[:, :, width_offset:(width_offset + target_width),
               height_offset:(height_offset + target_height)]

    def forward(self, x, skip_input):
        up = self.up(x)
        # In the paper, Ronneberger used valid convolutions, which means that `skip_input` would
        # not have the same spatial size as `up`. We therefore need this crop_skip_input to crop
        # `skip_input` to have the same spatial size as `up`.
        # BUT: since we're using same convolutions (ConvELU2D), this is not required.
        # skip = self.crop_skip_input(skip_input, up.size())  # TODO understand this !
        skip = skip_input
        return self.conv(torch.cat([up, skip], 1))


class UNet2D(nn.Module):
    """
    2d-unet for binary segmentation.
    """

    def __init__(self, n_channels, n_out_channels, n_scale=4, n_features_begin=64):
        super(UNet2D, self).__init__()

        self.n_scale = n_scale

        # THIS WOULD NOT WORK!
        # The nn.Module object has to register all "child" modules - this is how it keeps track
        # of all the parameters being used, so when you do Unet2D.parameters(), you get parameter
        # of all downscale and upscale layers (this is also required for GPU transfers -
        # this explains the bug you had). When you're setting a list, modules contained therein
        # are not registered.
        # However, there's a nn.ModuleList, which is a data-structure that walks and talks like a
        # list but is actually a nn.Module, such that all its contents are registered with module.
        # list of layers
        downscale_layers = []
        upscale_layers = []
        poolings = []

        n_features = n_features_begin
        n_in = n_channels

        # downscale layers
        for scale in range(self.n_scale):
            downscale_layers.append(
                DownscaleLayer(n_in, n_features)
            )
            n_in = n_features
            n_features *= 2
            poolings.append(nn.MaxPool2d(2))

        # lowest resolution layer
        downscale_layers.append(DownscaleLayer(n_in, n_features))
        n_in = n_features
        n_features //= 2

        # upscale layers
        for scale in range(self.n_scale):
            upscale_layers.append(
                UpscaleLayer(n_in, n_features)
            )
            n_in = n_features
            n_features //= 2

        # last 1 x 1 conv layer
        self.last = nn.Conv2d(n_in, n_out_channels, 1)
        # Now we register the lists by wrapping them as nn.ModuleList
        self.downscale_layers = nn.ModuleList(downscale_layers)
        self.upscale_layers = nn.ModuleList(upscale_layers)
        self.poolings = nn.ModuleList(poolings)
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
            out_down.append(out) # FIXME do we need to make a copy here somehow ?!
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

        # TODO is log_softmax correct?
        # That depends on what you want to do. :)
        return self.final_activation(self.last(out))
