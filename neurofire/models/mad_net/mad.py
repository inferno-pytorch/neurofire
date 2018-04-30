import torch.nn as nn
import torch
import torch.nn.functional as F
from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D
# from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample


class MADBlock(nn.Module):
    """
    Convolutional Block of MAD, no pooling

    The number of convolutions can be set
    (usually lower in early blocks to save memory)
    """
    conv_type = ConvELU3D

    def __init__(self, in_channels, out_channels,
                 n_convs=3, kernel_size=3, residual=False):
        super(self, MADBlock).__init__()
        convs = [self.conv_type(in_channels, out_channels, kernel_size)]
        convs += (n_convs - 1) * [self.conv_type(out_channels, in_channels, kernel_size)]
        self.conv = nn.Sqequential(*convs)
        self.residual = residual

    def forward(self, x):
        return x + self.conv(x) if self.residual else self.conv(x)


# TODO enable residual MAD block
# Hardcoded for pixel ratio 10 - 1 - 1 (CREMI) for now
class MAD(nn.Module):
    output_type = Conv3D
    activation = F.sigmoid

    def __init__(self, in_channels, out_channels, initial_num_fmaps=32, fmap_growth=2):
        num_fmaps = [initial_num_fmaps * fmap_growth**i for i in range(5)]
        super(self, MAD).__init__()
        # first convolutional block and pooling.
        # we start from ratio ~ (10, 1 ,1)
        # afterwards, we have pooled to ~ (10, 3, 3)
        self.block0 = MADBlock(in_channels, num_fmaps[0], n_convs=2)
        self.pool0 = nn.MaxPool3d((1, 3, 3))
        # second convolutional block and pooling.
        # we start from ratio ~ (10, 3, 3)
        # afterwards, we have pooled to ~ (10, 9, 9)
        # which is roughly isotropic ~~ (10, 10, 10)
        self.block1 = MADBlock(num_fmaps[0], num_fmaps[1], n_convs=2)
        self.pool1 = nn.MaxPool3d((1, 3, 3))
        # third convolutional block and pooling.
        # we start from isotropic ratio
        # which is roughly isotropic ~~ (10, 10, 10) after pooling we are at
        # ~ (20, 20, 20)
        self.block2 = MADBlock(num_fmaps[1], num_fmaps[2], n_convs=3)
        self.pool2 = nn.MaxPool3d(2)
        # fourth convolutional block and pooling.
        # which is roughly isotropic ~~ (20, 20, 20) after pooling we are at
        # ~ (40, 40, 40)
        self.block3 = MADBlock(num_fmaps[2], num_fmaps[3], n_convs=3)
        self.pool3 = nn.MaxPool3d(2)
        # fifth and last convolutional block / no pooling.
        self.block4 = MADBlock(num_fmaps[3], num_fmaps[4], n_convs=3)

        # convolution for intermediate loss
        self.intermed0 = self.output_type(num_fmaps[0], out_channels, 1)
        self.intermed1 = self.output_type(num_fmaps[1], out_channels, 1)
        self.intermed2 = self.output_type(num_fmaps[2], out_channels, 1)
        self.intermed3 = self.output_type(num_fmaps[3], out_channels, 1)
        self.intermed4 = self.output_type(num_fmaps[4], out_channels, 1)

        # so far so HED. How do we combine the intermediate results in a more clever way
        # to avoid huge sampling artifacts ???
        # maybe we don't have to with multiscale affinities ?

        # default HED:
        # self.upsample1 = nn.Upsample((1, 3, 3), mode='trilinear')
        # self.upsample2 = nn.Upsample((1, 9, 9), mode='trilinear')
        # self.upsample3 = nn.Upsample((2, 18, 18), mode='trilinear')
        # self.upsample4 = nn.Upsample(4, 36, 36), mode='trilinear')
        # self.out = self.output_type(5 * out_channels, out_channels, 1)

        # alternative: small network (upsampling same as above)
        # self.out_block = MADBlock(5 * out_channels, num_fmaps[0], n_convs=2)
        # self.out = self.output_type(num_fmaps[0], out_channels, 1)

        # alternatives:
        # - network path that does the upsampling in several steps for
        #   for each scale. Downside: very expensive
        #   as a compromise, we could just add a few additional layers before the output ?
        # - transposed convolutions instead of trilinear upsampling
        #   might reduce the sampling artifacts ??
        # - spatial transformer network:
        #   learn data dependent transormation
        #   don't know if pytorch supports this in 3d though
        #   also, not sure which transformation to learn (affine or spatial attention)
        # - new, different spatial attention mechanism ?

    def forward(self, x):
        # apply convolutions and pool
        conv0 = self.conv0(x)
        conv1 = self.conv1(self.pool0(conv0))
        conv2 = self.conv2(self.pool1(conv1))
        conv3 = self.conv3(self.pool2(conv2))
        conv4 = self.conv4(self.pool3(conv3))
        conv5 = self.conv5(self.pool4(conv4))

        # get intermediate results
        out0 = self.intermed0(conv0)
        out1 = self.intermed1(conv1)
        out2 = self.intermed2(conv2)
        out3 = self.intermed3(conv3)
        out4 = self.intermed4(conv4)
        out5 = self.intermed5(conv5)

        # default HED:
        # upsample all outputs and apply final convolution
        # up1 = self.upsample1(out1)
        # up2 = self.upsample2(out2)
        # up3 = self.upsample3(out3)
        # up4 = self.upsample4(out4)
        # out = self.out(torch.cat((out0, up1, up2, up3, up4), 1))
        # TODO more clever alternative ?!
        out = ''

        return (self.activation(out), self.activation(out0),
                self.activation(out1), self.activation(out2),
                self.activation(out3), self.activation(out4),
                self.activation(out5))
