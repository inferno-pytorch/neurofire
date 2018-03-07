from itertools import chain
import torch
import torch.nn as nn
from .hed import HED


# TODO reference to paper
class M2FCN(nn.Module):
    def __init__(self, in_channels=1,
                 out_channels=1, dilation=1,
                 conv_type_key='default',
                 block_type_key='default',
                 output_type_key='default',
                 upsampling_type_key='default'):
        super(M2FCN, self).__init__()
        self.hed1 = HED(in_channels=in_channels,
                        out_channels=out_channels,
                        dilation=dilation,
                        conv_type_key=conv_type_key,
                        block_type_key=block_type_key,
                        output_type_key=output_type_key,
                        upsampling_type_key=upsampling_type_key)
        self.hed2 = HED(in_channels=out_channels + 1,
                        out_channels=out_channels,
                        dilation=dilation,
                        conv_type_key=conv_type_key,
                        block_type_key=block_type_key,
                        output_type_key=output_type_key,
                        upsampling_type_key=upsampling_type_key)
        self.hed3 = HED(in_channels=out_channels + 1,
                        out_channels=out_channels,
                        dilation=dilation,
                        conv_type_key=conv_type_key,
                        block_type_key=block_type_key,
                        output_type_key=output_type_key,
                        upsampling_type_key=upsampling_type_key)

    # TODO concate all outputs instead of only fusion ?!
    # TODO consider applying hed2 iteratively
    def forward(self, x):
        x1 = self.hed1(x)
        x2 = self.hed2(torch.cat((x, x1[0]), 1))
        # only hed 2 ?!
        x3 = self.hed3(torch.cat((x, x2[0]), 1))
        # again, we return such that the last fusion is the first output
        return chain(x3, x2, x1)
