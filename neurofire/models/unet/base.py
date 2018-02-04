import torch
import torch.nn as nn


class Xcoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type, pre_output):
        super(Xcoder, self).__init__()
        assert out_channels % 2 == 0
        # make sure that conv-type
        assert isinstance(conv_type, type)
        # the in-channels we get from the top-level / bottom level layer + skip connections
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.conv1 = conv_type(in_channels=self.in_channels,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size)
        self.conv2 = conv_type(in_channels=self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size)
        self.pre_output = pre_output

    #
    # noinspection PyCallingNonCallable
    def forward(self, input_):
        conv1_out = self.conv1(input_)
        conv2_out = self.conv2(conv1_out)
        if self.pre_output is not None:
            out = self.pre_output(conv2_out)
        else:
            out = conv2_out
        return out


class XcoderResidual(Xcoder):
    """
    Inspired by arXiv:1706.00120

    """

    def __init__(self, *super_args, **super_kwargs):
        super(XcoderResidual, self).__init__(*super_args, **super_kwargs)

        # Add initial 2D convolution:
        if isinstance(self.kernel_size, int):
            self.kernel_size_2d = (1, self.kernel_size, self.kernel_size)
        else:
            assert isinstance(self.kernel_size, tuple)
            assert len(self.kernel_size) == 3
            self.kernel_size_2d = (1, self.kernel_size[1], self.kernel_size[2])

        # change block 1 to 2d conv
        self.conv1 = self.conv_type(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=self.kernel_size_2d)

        # add an additional conv layer
        self.conv3 = self.conv_type(in_channels=self.out_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=self.kernel_size)

    def forward(self, input_):
        conv1_out = self.conv1(input_)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)

        # Add skip connection:
        out = conv1_out + conv3_out

        if self.pre_output is not None:
            out = self.pre_output(out)

        return out


class UNetSkeleton(nn.Module):
    def __init__(self, encoders, base, decoders, output, final_activation=None):
        super(UNetSkeleton, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)

        # why do we hard-code this to 3 ? wouldn't it be enough to check that they are
        # all of the same length
        assert len(encoders) == len(decoders) == 3
        assert isinstance(base, nn.Module)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.base = base
        self.output = output
        if isinstance(final_activation, str):
            self.final_activation = getattr(nn, final_activation)()
        elif isinstance(final_activation, nn.Module):
            self.final_activation = final_activation
        elif final_activation is None:
            self.final_activation = None
        else:
            raise NotImplementedError

    def forward(self, input_):

        # all spatial sizes (ssize) for 512 input size
        # and donwscaling by a factor of 2
        # apply first decoder
        # e0.ssize = 256 (ssize / scale_factor)
        e0 = self.encoders[0](input_)

        # apply second encoder
        # e1.ssize = 128 (ssize / (scale_factor**2))
        e1 = self.encoders[1](e0)

        # apply third encoder
        # e2.ssize = 64 (ssize / (scale_factor**3))
        e2 = self.encoders[2](e1)

        # apply the base
        # b.ssize = 64
        b = self.base(e2)

        # apply the third / lowest decoder with input from base
        # and encoder 2
        # d2.ssize = 128
        d2 = self.decoders[0](
            torch.cat((b, e2), 1)
        )

        # apply the second decoder with input from the third decoder
        # and the second encoder
        # d1.ssize = 256
        d1 = self.decoders[1](
            torch.cat((d2, e1), 1)
        )

        # apply the first decoder with input from the second decoder
        # and the first encoder
        # d0.ssize = 512
        d0 = self.decoders[2](
            torch.cat((d1, e0), 1)
        )

        # out.ssize = 512
        out = self.output(d0)

        if self.final_activation is not None:
            out = self.final_activation(out)
        return out
