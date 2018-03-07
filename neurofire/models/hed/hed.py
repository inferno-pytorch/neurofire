# This model was taken from https://github.com/xlliu7/hed.pytorch

import torch.nn as nn
import torch
import torch.nn.functional as F
from inferno.extensions.layers.convolutional import ConvELU2D, ConvELU3D
from inferno.extensions.layers.convolutional import Conv2D, Conv3D
from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample

# def crop(d, g):
#     g_h, g_w = g.size()[2:4]
#     d_h, d_w = d.size()[2:4]
#     d1 = d[:, :, int(math.floor((d_h - g_h)/2.0)):
#                  int(math.floor((d_h - g_h)/2.0)) + g_h,
#                  int(math.floor((d_w - g_w)/2.0)):int(math.floor((d_w - g_w)/2.0)) + g_w]
#     return d1


# conv relu with VALID padding
class DefaultConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, dilation=1):
        super(DefaultConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel,
                              padding=dilation, dilation=dilation)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


# conv relu with VALID padding in 3D
class DefaultConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, dilation=1):
        super(DefaultConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel,
                              padding=dilation, dilation=dilation)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


# NOTE we use 3 convolutions for all blocks
# in the initial implementations, the first 2 blocks only
# consist of 2 convolutions
class DefaultHEDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type, dilation=1, with_maxpool=True):
        super(DefaultHEDBlock, self).__init__()
        self.conv = nn.Sequential(conv_type(in_channels, out_channels, 3, dilation),
                                  conv_type(out_channels, out_channels, 3, dilation),
                                  conv_type(out_channels, out_channels, 3, dilation))
        self.with_maxpool = with_maxpool
        if self.with_maxpool:
            pooling_stride = 1 if dilation > 1 else 2
            ceil_mode = False if dilation > 1 else True
            self.pooler = nn.MaxPool2d(2, stride=pooling_stride, ceil_mode=ceil_mode)

    def forward(self, x):
        if self.with_maxpool:
            return self.conv(self.pooler(x))
        else:
            return self.conv(x)


# NOTE we use 3 convolutions for all blocks
# in the initial implementations, the first 2 blocks only
# consist of 2 convolutions
class DefaultHEDBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type, dilation=1, with_maxpool=True):
        super(DefaultHEDBlock3D, self).__init__()
        self.conv = nn.Sequential(conv_type(in_channels, out_channels, 3, dilation),
                                  conv_type(out_channels, out_channels, 3, dilation),
                                  conv_type(out_channels, out_channels, 3, dilation))
        self.with_maxpool = with_maxpool
        if self.with_maxpool:
            pooling_stride = 1 if dilation > 1 else 2
            ceil_mode = False if dilation > 1 else True
            self.pooler = nn.MaxPool3d(2, stride=pooling_stride, ceil_mode=ceil_mode)

    def forward(self, x):
        if self.with_maxpool:
            return self.conv(self.pooler(x))
        else:
            return self.conv(x)


# NOTE we use 3 convolutions for all blocks
# in the initial implementations, the first 2 blocks only
# consist of 2 convolutions
# TODO make the pooling factor settable !!
class AnisotropicHEDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type, dilation=1,
                 with_maxpool=True, pooling_factor=3):
        assert dilation == 1, "Dilation not supported for anisotropic HED"
        super(AnisotropicHEDBlock, self).__init__()
        self.conv = nn.Sequential(conv_type(in_channels, out_channels, 3, dilation),
                                  conv_type(out_channels, out_channels, 3, dilation),
                                  conv_type(out_channels, out_channels, 3, dilation))
        self.with_maxpool = with_maxpool
        if self.with_maxpool:
            self.pooler = AnisotropicPool(pooling_factor)

    def forward(self, x):
        if self.with_maxpool:
            return self.conv(self.pooler(x))
        else:
            return self.conv(x)


# FIXME bilinear upsampling for 5D input dies not work
class Upsampling3d(nn.Module):
    def __init__(self, scale_factor):
        super(Upsampling3d, self).__init__()
        self.sample = nn.Upsample(scale_factor=scale_factor)  # , mode='bilinear')

    def forward(self, x):
        return self.sample(x)


class HED(nn.Module):
    conv_types = {'default': DefaultConv,
                  'default3d': DefaultConv3D,
                  'same': ConvELU2D,
                  'same3d': ConvELU3D}
    block_types = {'default': DefaultHEDBlock,
                   'default3d': DefaultHEDBlock3D,
                   'anisotropic': AnisotropicHEDBlock}
    output_types = {'default': nn.Conv2d,
                    'default3d': nn.Conv3d,
                    'same': Conv2D,
                    'same3d': Conv3D}
    upsampling_types = {'default': nn.UpsamplingBilinear2d,
                        'default3d': Upsampling3d,
                        'anisotropic': AnisotropicUpsample}

    def __init__(self, in_channels=3,
                 out_channels=1, dilation=1,
                 conv_type_key='default',
                 block_type_key='default',
                 output_type_key='default',
                 upsampling_type_key='default'):
        # validate input keys
        assert conv_type_key in self.conv_types, conv_type_key
        conv_type = self.conv_types[conv_type_key]

        # block types can be single key or list of keys
        assert isinstance(block_type_key, (str, list, tuple))
        if isinstance(block_type_key, (list, tuple)):
            assert len(block_type_key) == 5
            assert all(bkt in self.block_types for bkt in block_type_key), block_type_key
            self.block_type_key = block_type_key
        else:
            assert block_type_key in self.block_types, block_type_key
            self.block_type_key = 5 * (block_type_key,)

        assert output_type_key in self.output_types, output_type_key
        output_type = self.output_types[output_type_key]

        # upsampling types can be single key or list of keys
        if isinstance(upsampling_type_key, (list, tuple)):
            assert len(upsampling_type_key) == 5
            assert all(ukt in self.upsampling_types
                       for ukt in upsampling_type_key), upsampling_type_key
            self.upsampling_type_key = upsampling_type_key
        else:
            assert upsampling_type_key in self.upsampling_types, upsampling_type_key
            self.upsampling_type_key = 5 * (upsampling_type_key,)

        super(HED, self).__init__()
        self.conv1 = self.block_types[self.block_type_key[0]](in_channels, 64, conv_type, with_maxpool=False)
        self.conv2 = self.block_types[self.block_type_key[1]](64, 128, conv_type)
        self.conv3 = self.block_types[self.block_type_key[2]](128, 256, conv_type)
        self.conv4 = self.block_types[self.block_type_key[3]](256, 512, conv_type)
        self.conv5 = self.block_types[self.block_type_key[4]](512, 512, conv_type, dilation=dilation)

        self.dsn1 = output_type(64, out_channels, 1)
        self.dsn2 = output_type(128, out_channels, 1)
        self.dsn3 = output_type(256, out_channels, 1)
        self.dsn4 = output_type(512, out_channels, 1)
        self.dsn5 = output_type(512, out_channels, 1)
        # 6 is the fusion layer -> 5 * out_channels
        self.dsn6 = output_type(5*out_channels, out_channels, 1)

        # last_scale = 8 if dilation > 1 else 16
        # FIXME don't hardcode cremi values
        self.upscore2 = self.upsampling_types[self.upsampling_type_key[0]](scale_factor=3)
        self.upscore3 = self.upsampling_types[self.upsampling_type_key[1]](scale_factor=9)
        self.upscore4 = self.upsampling_types[self.upsampling_type_key[2]](scale_factor=2)
        # self.upscore5 = self.upsampling_types[self.upsampling_type_key[3]](scale_factor=last_scale)
        self.upscore5 = self.upsampling_types[self.upsampling_type_key[3]](scale_factor=4)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # FIXME don't hardcode cremi settings
        # side output
        d5 = self.upscore5(self.upscore2(self.dsn5(conv5)))
        # d5 = crop(dsn5_up, gt)

        d4 = self.upscore4(self.uspcore2(self.dsn4(conv4)))
        # d4 = crop(dsn4_up, gt)

        d3 = self.upscore3(self.upscore2(self.dsn3(conv3)))
        # d3 = crop(dsn3_up, gt)

        d2 = self.upscore2(self.dsn2(conv2))
        # d2 = crop(dsn2_up, gt)

        d1 = self.dsn1(conv1)
        # d1 = crop(dsn1, gt)

        # dsn fusion output
        d6 = self.dsn6(torch.cat((d1, d2, d3, d4, d5), 1))

        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        d6 = F.sigmoid(d6)

        # we return first, because it is usually the one used for stuff
        return d6, d2, d3, d4, d5, d1
