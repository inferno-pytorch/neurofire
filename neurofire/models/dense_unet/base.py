import torch
import torch.nn as nn


class Xcoder(nn.Module):
    def __init__(self, previous_in_channels, out_channels, kernel_size, conv_type, pre_output):
        super(Xcoder, self).__init__()
        print("Using Xcoder!")
        assert out_channels % 2 == 0
        assert isinstance(conv_type, type)
        self.in_channels = sum(previous_in_channels)
        self.out_channels = out_channels
        self.conv1 = conv_type(in_channels=self.in_channels,
                               out_channels=self.out_channels // 2,
                               kernel_size=kernel_size)
        self.conv2 = conv_type(in_channels=self.in_channels + (self.out_channels // 2),
                               out_channels=self.out_channels // 2,
                               kernel_size=kernel_size)
        self.pre_output = pre_output

    # noinspection PyCallingNonCallable
    def forward(self, input_):
        conv1_out = self.conv1(input_)
        conv2_inp = torch.cat((input_, conv1_out), 1)
        conv2_out = self.conv2(conv2_inp)
        conv_out = torch.cat((conv1_out, conv2_out), 1)
        if self.pre_output is not None:
            out = self.pre_output(conv_out)
        else:
            out = conv_out
        return out


class Gcoder(nn.Module):
    def __init__(self, previous_in_channels, out_channels, kernel_size, conv_type, pre_output):
        super(Gcoder, self).__init__()
        print("Using Gcode!")
        self.in_channels = sum(previous_in_channels)
        self.out_channels = out_channels
        self.conv1 = conv_type(in_channels=self.in_channels,
                               out_channels=self.out_channels // 2,
                               kernel_size=kernel_size)
        self.conv2 = conv_type(in_channels=self.out_channels // 2,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size)
        self.pre_output = pre_output

    def forward(self, input_):
        conv1_out = self.conv1(input_)
        conv2_out = self.conv2(conv1_out)
        if self.pre_output is not None:
            out = self.pre_output(conv2_out)
        else:
            out = conv2_out
        return out


class DUNetSkeleton(nn.Module):
    def __init__(self, encoders, poolers, base, upsamplers, decoders, output,
                 final_activation=None, return_hypercolumns=False):
        super(DUNetSkeleton, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)
        assert len(encoders) == len(decoders) == 3
        assert len(poolers) == 3
        assert len(upsamplers) == 3
        assert isinstance(base, nn.Module)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.poolx2 = poolers[0]
        self.poolx4 = poolers[1]
        self.poolx8 = poolers[2]
        self.upx2 = upsamplers[0]
        self.upx4 = upsamplers[1]
        self.upx8 = upsamplers[2]
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
        self.return_hypercolumns = return_hypercolumns

    def forward(self, input_):

        # Say input_ spatial size is 512, i.e. input_.ssize = 512
        # Pre-pool to save computation
        input_2ds = self.poolx2(input_)
        input_4ds = self.poolx4(input_)
        input_8ds = self.poolx8(input_)

        # e0.ssize = 256
        e0 = self.encoders[0](input_)
        e0_2ds = self.poolx2(e0)
        e0_4ds = self.poolx4(e0)
        e0_2us = self.upx2(e0)

        # e1.ssize = 128
        e1 = self.encoders[1](torch.cat((input_2ds, e0), 1))
        e1_2ds = self.poolx2(e1)
        e1_2us = self.upx2(e1)
        e1_4us = self.upx4(e1)

        # e2.ssize = 64
        e2 = self.encoders[2](torch.cat((input_4ds,
                                         e0_2ds,
                                         e1), 1))
        e2_2us = self.upx2(e2)
        e2_4us = self.upx4(e2)
        e2_8us = self.upx8(e2)

        # b.ssize = 64
        b = self.base(torch.cat((input_8ds,
                                 e0_4ds,
                                 e1_2ds,
                                 e2), 1))
        b_2us = self.upx2(b)
        b_4us = self.upx4(b)
        b_8us = self.upx8(b)

        # d2.ssize = 128
        d2 = self.decoders[0](torch.cat((input_8ds,
                                         e0_4ds,
                                         e1_2ds,
                                         e2,
                                         b), 1))
        d2_2us = self.upx2(d2)
        d2_4us = self.upx4(d2)

        # d1.ssize = 256
        d1 = self.decoders[1](torch.cat((input_4ds,
                                         e0_2ds,
                                         e1,
                                         e2_2us,
                                         b_2us,
                                         d2), 1))
        d1_2us = self.upx2(d1)

        # d0.ssize = 512
        d0 = self.decoders[2](torch.cat((input_2ds,
                                         e0,
                                         e1_2us,
                                         e2_4us,
                                         b_4us,
                                         d2_2us,
                                         d1), 1))

        # out.ssize = 512
        out = self.output(torch.cat((input_,
                                     e0_2us,
                                     e1_4us,
                                     e2_8us,
                                     b_8us,
                                     d2_4us,
                                     d1_2us,
                                     d0), 1))

        if self.final_activation is not None:
            out = self.final_activation(out)

        if not self.return_hypercolumns:
            return out
        else:
            out = torch.cat((input_,
                             e0_2us,
                             e1_4us,
                             e2_8us,
                             b_8us,
                             d2_4us,
                             d1_2us,
                             d0,
                             out), 1)
            return out
