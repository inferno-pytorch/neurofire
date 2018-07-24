import torch.nn as nn
import torch
import torch.nn.functional as F
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D
# from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample


class MADBlock2D(nn.Module):
    """
    Convolutional Block of MAD, no pooling

    The number of convolutions can be set
    (usually lower in early blocks to save memory)
    """
    conv_type = ConvELU2D

    def __init__(self, in_channels, out_channels,
                 n_convs=3, kernel_size=3, residual=False):
        super(MADBlock2D, self).__init__()
        convs = [self.conv_type(in_channels, out_channels, kernel_size)]
        convs += (n_convs - 1) * [self.conv_type(out_channels, out_channels, kernel_size)]
        self.conv = nn.Sequential(*convs)
        self.residual = residual

    def forward(self, x):
        return x + self.conv(x) if self.residual else self.conv(x)


# TODO enable residual MAD block
class MAD2D(nn.Module):
    output_type = Conv2D
    # activation = F.sigmoid

    def __init__(self, in_channels, out_channels, initial_num_fmaps=32, fmap_growth=2):
        num_fmaps = [initial_num_fmaps * fmap_growth**i for i in range(4)]
        super(MAD2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # first convolutional block and pooling.
        self.block0 = MADBlock2D(in_channels, num_fmaps[0], n_convs=2)
        self.pool0 = nn.MaxPool2d(2)
        # second convolutional block and pooling.
        self.block1 = MADBlock2D(num_fmaps[0], num_fmaps[1], n_convs=2)
        self.pool1 = nn.MaxPool2d(2)
        # third convolutional block and pooling.
        self.block2 = MADBlock2D(num_fmaps[1], num_fmaps[2], n_convs=3)
        self.pool2 = nn.MaxPool2d(2)
        # fourth convolutional block and pooling.
        self.block3 = MADBlock2D(num_fmaps[2], num_fmaps[3], n_convs=3)
        self.pool3 = nn.MaxPool2d(2)
        # fifth and last convolutional block / no pooling.
        # NOTE we don't increase the number of fmaps any further in this layer
        # to save some memory
        self.block4 = MADBlock2D(num_fmaps[3], num_fmaps[3], n_convs=3)

        # convolution for intermediate loss
        self.intermed0 = self.output_type(num_fmaps[0], out_channels, 1)
        self.intermed1 = self.output_type(num_fmaps[1], out_channels, 1)
        self.intermed2 = self.output_type(num_fmaps[2], out_channels, 1)
        self.intermed3 = self.output_type(num_fmaps[3], out_channels, 1)
        self.intermed4 = self.output_type(num_fmaps[3], out_channels, 1)

        # so far so HED. How do we combine the intermediate results in a more clever way
        # to avoid huge sampling artifacts ???
        # maybe we don't have to with multiscale affinities ?

        # default HED:
        # added the align_corners=False to prevent annoying warning
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.out = self.output_type(5 * out_channels, out_channels, 1)

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
        # some loaders are usually 3D, so we reshape if necessary
        if x.dim() == 5:
            reshape_to_3d = True
            b, c, _0, _1, _2 = list(x.size())
            assert _0 == 1, "%i" % _0
            x = x.view(b, c * _0, _1, _2)
        else:
            reshape_to_3d = False

        # apply convolutions and pool
        conv0 = self.block0(x)
        conv1 = self.block1(self.pool0(conv0))
        conv2 = self.block2(self.pool1(conv1))
        conv3 = self.block3(self.pool2(conv2))
        conv4 = self.block4(self.pool3(conv3))

        # get intermediate results
        out0 = self.intermed0(conv0)
        out1 = self.intermed1(conv1)
        out2 = self.intermed2(conv2)
        out3 = self.intermed3(conv3)
        out4 = self.intermed4(conv4)

        # default HED:
        # upsample all outputs and apply final convolution
        up1 = self.upsample1(out1)
        up2 = self.upsample2(out2)
        up3 = self.upsample3(out3)
        up4 = self.upsample4(out4)

        out = self.out(torch.cat((out0, up1, up2, up3, up4), 1))

        # TODO more clever alternative ?!
        # out = ''
        activation = F.sigmoid
        outputs = [activation(out), activation(out0),
                   activation(out1), activation(out2),
                   activation(out3), activation(out4)]

        if reshape_to_3d:
            outsize = [list(out.size()) for out in outputs]
            outputs = [out.view(osize[0], osize[1], 1, osize[2], osize[3])
                       for out, osize in zip(outputs, outsize)]
        # print("Net output sizes:", out.size(), out0.size(), out1.size())
        return outputs
