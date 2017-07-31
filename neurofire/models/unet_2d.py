import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# TODO can we use the equivalent of 'SAME' padding somehow to not lose context ?

class UpscaleLayer(nn.Module):
    """
    up-scale block of 2d unet
    """

    def __init__(self, in_size, out_size, kernel_size=3):
        super(UpscaleLayer, self).__init__()
        self.layer = Sequential(
            nn.Conv2d(in_size, out_size, kernel_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UNet2DDownscaleLayer(nn.Module):
    """
    down-scale block of 2d unet
    """

    def __init__(self, in_size, out_size, kernel_size=3):
        super(DownscaleLayer, self).__init__()
        up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2),
        self.conv = Sequential(
            nn.Conv2d(out_size, out_size, kernel_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size),
            nn.ReLU()
        )


    def crop_skip_input(self, skip_input, target_size):
        """
        Crop the skip input to correct size
        """
        _, _, skip_width, skip_height = skip_input.size()
        _, _, target_width, target_height = target_size
        width_offset = (skip_width - target_width) / 2
        height_offset = (skip_height - target_height) / 2
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
        super(UNet2d, self).__init__()

        # list of layers
        self.downscale_layers = []
        self.upscale_layers = []
        self.poolings = []

        n_features = n_features_begin
        n_in = n_channels

        # downscale layers
        for scale in range(n_scale):
            self.downscale_layers.append(
                UNet2DDownscaleLayer(n_in, n_features)
            )
            n_in = n_features
            n_features *= 2
            self.poolings.append(nn.MaxPool2d(2))

        # lowest resolution layer
        self.downscale_layers.append(UNet2DDownscaleLayer(n_in, n_features))
        n_in = n_features
        n_features /= 2

        # upscale layers
        for scale in range(n_scale):
            self.upscale_layers.append(
                UNet2DUpscaleLayer(n_in, n_features)
            )
            n_in = n_features
            n_features /= 2

        # last 1 x 1 conv layer
        self.last = nn.Conv2d(n_in, n_out_channels, 1)


    # the forward pass - defining the connectivity of layers
    def forward(self, x):
        out_down = []

        out = x
        # apply the downscale layers
        for scale in range(n_scale):
            out = self.downscale_layers[scale](out)
            out_down.append(out) # FIXME do we need to make a copy here somehow ?!
            out = self.poolings[scale](out)

        # apply the lowest res layer
        out = self.downscale_layers[-1](out)

        # apply the upscale layers
        for scale in range(n_scale)[::-1]:
            out = self.upscale_layers(out, out_down[scale])

        # TODO is log_softmax correct?
        return F.log_softmax(self.last(out))
