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

        assert isinstance(N, (list, tuple))
        assert len(N) > 1
        super(M2FCN, self).__init__()
        self.heds = nn.ModuleList([DenseHED(in_channels=in_channels,
                                            out_channels=out_channels, N=n,
                                            scale_factor=scale_factor,
                                            block_type_key=block_type_key,
                                            output_type_key=output_type_key,
                                            sampling_type_key=sampling_type_key) for n in N])

    # TODO concate all outputs instead of only fusion ?!
    # TODO consider applying hed2 iteratively
    def forward(self, x):
        out = []
        for i, hed in enumerate(self.heds):
            if i == 0:
                out.append(hed(x))
            else:
                out.append(hed(out[-1][0]))
        return tuple(chain(*out))
