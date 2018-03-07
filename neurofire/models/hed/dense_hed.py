import torch.nn as nn
import torch
import torch.nn.functional as F
from inferno.extensions.layers.convolutional import ConvELU2D, ConvELU3D
from inferno.extensions.layers.convolutional import Conv2D, Conv3D
from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample
from .hed2 import DefaultHEDBlock, DefaultHEDBlock3D, Upsampling3d


# dense block
class DenseHEDBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type1, conv_type2,
                 dilation=1, kernel=3):
        super(DenseHEDBlockBase, self).__init__()
        self.conv1 = conv_type1(in_channels=in_channels,
                                out_channels=out_channels // 2,
                                kernel_size=kernel,
                                dilation=dilation)
        self.conv2 = conv_type2(in_channels=in_channels + out_channels // 2,
                                out_channels=out_channels // 2,
                                kernel_size=kernel,
                                dilation=dilation)

    def forward(self, input_):
        conv1 = self.conv1(input_)
        conv2 = self.conv2(torch.cat((input_, conv1), 1))
        return torch.cat((conv1, conv2), 1)


# We don't support dilation here for now
class DenseHEDBlock(DenseHEDBlockBase):
    def __init__(self, in_channels, out_channels):
        super(DenseHEDBlock, self).__init__(in_channels, out_channels,
                                            ConvELU2D, ConvELU2D)


# We don't support dilation here for now
class DenseHEDBlock3D(DenseHEDBlockBase):
    def __init__(self, in_channels, out_channels):
        super(DenseHEDBlock3D, self).__init__(in_channels, out_channels,
                                              ConvELU3D, ConvELU3D)


class DenseHED(nn.Module):
    block_types = {'default': DefaultHEDBlock,
                   'default3d': DefaultHEDBlock3D,
                   'dense': DenseHEDBlock,
                   'dense3d': DenseHEDBlock3D}
    sampling_types = {'default': (nn.MaxPool2d, nn.UpsamplingBilinear2d),
                      'default3d': (nn.MaxPool3d, Upsampling3d),
                      'anisotropic': (AnisotropicPool, AnisotropicUpsample)}
    output_types = {'default': Conv2D,
                    'default3d': Conv3D}

    def __init__(self,
                 in_channels,
                 out_channels, N=16,
                 scale_factor=2,
                 block_type_key='dense',
                 output_type_key='default',
                 sampling_type_key='default'):

        # scale factors can be list or single value
        assert isinstance(scale_factor, (int, list, tuple))
        if isinstance(scale_factor, (list, tuple)):
            assert len(scale_factor) == 4
            self.scale_factor = scale_factor
        else:
            self.scale_factor = 4 * (scale_factor,)

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

        # sampling types can be single key or list of keys
        if isinstance(sampling_type_key, (list, tuple)):
            assert len(sampling_type_key) == 4
            assert all(skt in self.upsampling_types
                       for skt in sampling_type_key), sampling_type_key
            self.sampling_type_key = sampling_type_key
        else:
            assert sampling_type_key in self.sampling_types, sampling_type_key
            self.sampling_type_key = 4 * (sampling_type_key,)

        super(DenseHED, self).__init__()

        # NOTE in contrast to nasims dense-unet impl, we don't connect the input to higher levels
        # convolutional blocks
        self.conv0 = self.block_types[self.block_type_key[0]](in_channels, N)
        self.conv1 = self.block_types[self.block_type_key[1]](N, 2*N)
        self.conv2 = self.block_types[self.block_type_key[2]](3*N, 4*N)
        self.conv3 = self.block_types[self.block_type_key[3]](7*N, 8*N)
        # we don't change the number of feats in the last conv block
        self.conv4 = self.block_types[self.block_type_key[4]](15*N, 16*N)

        # poolers
        sample_type0 = self.sampling_types[self.sampling_type_key[0]]
        self.pool0 = sample_type0[0](self.scale_factor[0], stride=self.scale_factor[0])

        sample_type1 = self.sampling_types[self.sampling_type_key[1]]
        self.pool1 = sample_type1[0](self.scale_factor[1], stride=self.scale_factor[1])

        sample_type2 = self.sampling_types[self.sampling_type_key[2]]
        self.pool2 = sample_type2[0](self.scale_factor[2], stride=self.scale_factor[2])

        sample_type3 = self.sampling_types[self.sampling_type_key[3]]
        self.pool3 = sample_type3[0](self.scale_factor[3], stride=self.scale_factor[3])

        self.out0 = output_type(N, out_channels, 1)
        self.out1 = output_type(2*N, out_channels, 1)
        self.out2 = output_type(4*N, out_channels, 1)
        self.out3 = output_type(8*N, out_channels, 1)
        self.out4 = output_type(16*N, out_channels, 1)
        # 6 is the fusion layer -> 5 * out_channels
        self.out5 = output_type(5*out_channels, out_channels, 1)

        # FIXME don't hardcode cremi values
        self.upsample0 = sample_type0[1](scale_factor=self.scale_factor[0])
        self.upsample1 = sample_type1[1](scale_factor=self.scale_factor[1])
        self.upsample2 = sample_type2[1](scale_factor=self.scale_factor[2])
        self.upsample3 = sample_type3[1](scale_factor=self.scale_factor[3])

    def forward(self, x):

        # apply convolutions and poolings

        # hed block 0
        conv0 = self.conv0(x)
        # to next level
        conv1 = self.pool0(conv0)
        # dense connections to higher levels
        conv02 = self.pool1(conv1)
        conv03 = self.pool2(conv02)
        conv04 = self.pool3(conv03)

        # hed block 1
        conv1 = self.conv1(conv1)
        # to next level
        conv2 = self.pool1(conv1)
        # dense connections to higher levels
        conv13 = self.pool2(conv2)
        conv14 = self.pool3(conv13)

        # hed block 2
        conv2 = self.conv2(torch.cat((conv02, conv2), 1))
        # to next level
        conv3 = self.pool2(conv2)
        # dense connections to higher levels
        conv24 = self.pool3(conv3)

        # hed block 3
        conv3 = self.conv3(torch.cat((conv03, conv13, conv3), 1))
        # to next level
        conv4 = self.pool3(conv3)

        # hed block 4
        conv4 = self.conv4(torch.cat((conv04, conv14, conv24, conv4), 1))

        # make side output
        # NOTE we may have different pooling schemes, so we need to apply
        # all the samplers in a chained fashion
        out0 = self.out0(conv0)
        out1 = self.upsample0(self.out1(conv1))
        out2 = self.upsample1(self.upsample0(self.out2(conv2)))
        out3 = self.upsample2(self.upsample1(self.upsample0(self.out3(conv3))))
        out4 = self.upsample3(self.upsample2(self.upsample1(self.upsample0(self.out4(conv4)))))

        # make fusion output
        out5 = self.out5(torch.cat((out0, out1, out2, out3, out4), 1))

        # apply activations
        # TODO enable different activations
        out0 = F.sigmoid(out0)
        out1 = F.sigmoid(out1)
        out2 = F.sigmoid(out2)
        out3 = F.sigmoid(out3)
        out4 = F.sigmoid(out4)
        out5 = F.sigmoid(out5)

        # we return first, because it is usually the one used for stuff
        return out5, out4, out3, out2, out1, out0
