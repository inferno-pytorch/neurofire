from itertools import chain
import torch
import torch.nn as nn
from .dense_hed import DenseHED


# TODO reference to paper
# TODO choice between hed and dense hed
class M2FCN(nn.Module):
    def __init__(self, in_channels, out_channels, N,
                 scale_factor=2,
                 block_type_key='default',
                 output_type_key='default',
                 sampling_type_key='default'):

        assert isinstance(N, (int, list, tuple))
        if isinstance(N, int):
            self.N = (N,) * 3
        else:
            assert len(N) == 3
            self.N = N
        super(M2FCN, self).__init__()
        self.hed1 = DenseHED(in_channels=in_channels,
                             out_channels=out_channels, N=self.N[0],
                             scale_factor=scale_factor,
                             block_type_key=block_type_key,
                             output_type_key=output_type_key,
                             sampling_type_key=sampling_type_key)
        self.hed2 = DenseHED(in_channels=out_channels + 1,
                             out_channels=out_channels, N=self.N[1],
                             scale_factor=scale_factor,
                             block_type_key=block_type_key,
                             output_type_key=output_type_key,
                             sampling_type_key=sampling_type_key)
        self.hed3 = DenseHED(in_channels=out_channels + 1,
                             out_channels=out_channels, N=self.N[2],
                             scale_factor=scale_factor,
                             block_type_key=block_type_key,
                             output_type_key=output_type_key,
                             sampling_type_key=sampling_type_key)

    # TODO concate all outputs instead of only fusion ?!
    # TODO consider applying hed2 iteratively
    def forward(self, x):
        x1 = self.hed1(x)
        x2 = self.hed2(torch.cat((x, x1[0]), 1))
        # only hed 2 ?!
        x3 = self.hed3(torch.cat((x, x2[0]), 1))
        # again, we return such that the last fusion is the first output
        return tuple(chain(x3, x2, x1))
