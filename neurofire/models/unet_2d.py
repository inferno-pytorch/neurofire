from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

# TODO can we use the equivalent of 'SAME' padding somehow to not lose context ?

class DownscaleLayer(nn.Module):
    """
    Down-scale block of 2d unet
    """

    def __init__(self, in_size, out_size, kernel_size=3):
        super(DownscaleLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size),
            nn.ReLU()
        )

    # TODO there should be some inferno functionality for this ?!
    # initialize the weights with xaver initialization
    def initialize_weights(self):
        for ll in self.layer:
            if isinstance(ll, nn.Conv2d):
                weight_init.xavier_normal(ll.weight)

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
            nn.Conv2d(in_size, out_size, kernel_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size),
            nn.ReLU()
        )

    # TODO there should be some inferno functionality for this ?!
    # initialize the weights with xaver initialization
    def initialize_weights(self):
        weight_init.xavier_normal(self.up.weight)
        for ll in self.conv:
            if isinstance(ll, nn.Conv2d):
                weight_init.xavier_normal(ll.weight)

    def crop_skip_input(self, skip_input, target_size):
        """
        Crop the skip input to correct size
        """
        _, _, skip_width, skip_height = skip_input.size()
        _, _, target_width, target_height = target_size
        width_offset = (skip_width - target_width) // 2
        height_offset = (skip_height - target_height) // 2
        return skip_input[:, :, width_offset:(width_offset + target_width), height_offset:(height_offset + target_height)]

    def forward(self, x, skip_input):
        up = self.up(x)
        skip = self.crop_skip_input(skip_input, up.size())  # TODO understand this !
        return self.conv(torch.cat([up, skip], 1))


class UNet2D(nn.Module):
    """
    2d-unet for binary segmentation.
    """

    def __init__(self, n_channels, n_out_channels, n_scale=4, n_features_begin=64):
        super(UNet2D, self).__init__()

        self.n_scale = n_scale

        # list of layers
        self.downscale_layers = []
        self.upscale_layers = []
        self.poolings = []

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

    def initialize_weights(self):
        for scale in range(self.n_scale):
            self.downscale_layers[scale].initialize_weights()
            self.upscale_layers[scale].initialize_weights()
        self.downscale_layers[-1].initialize_weights()

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
            print("Upscaling layer", scale, "out-shape:", out.size())

        # TODO is log_softmax correct?
        return F.log_softmax(self.last(out))
