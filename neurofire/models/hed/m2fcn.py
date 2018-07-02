from itertools import chain
import torch
import torch.nn as nn
from .dense_hed import DenseHED
from .hed2 import HED


# TODO reference to paper
class M2FCN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 parameter,
                 scale_factor=2,
                 block_type_key='default',
                 output_type_key='default',
                 sampling_type_key='default',
                 fuse_sub_networks=False):

        assert isinstance(parameter, (list, tuple))
        assert len(parameter) > 1

        super(M2FCN, self).__init__()

        # need to have out channels as member for inference engine
        self.out_channels = out_channels
        self.fuse_sub_networks = fuse_sub_networks

        self.heds = nn.ModuleList([])
        for i, param in enumerate(parameter):
            expected_in_channels = in_channels if i == 0 else out_channels
            assert isinstance(param, (list, int, tuple))
            if isinstance(param, int):
                self.heds.append(DenseHED(in_channels=expected_in_channels,
                                          out_channels=out_channels, N=param,
                                          scale_factor=scale_factor,
                                          block_type_key=block_type_key,
                                          output_type_key=output_type_key,
                                          sampling_type_key=sampling_type_key))
            else:
                assert len(param) == 2
                self.heds.append(HED(in_channels=expected_in_channels,
                                     out_channels=out_channels,
                                     initial_num_fmaps=param[0],
                                     fmap_growth=param[1],
                                     scale_factor=scale_factor,
                                     block_type_key=block_type_key,
                                     output_type_key=output_type_key,
                                     sampling_type_key=sampling_type_key))
        if self.fuse_sub_networks:
            self.fuse = HED.output_types[output_type_key](6 * len(self.heds) * out_channels,
                                                          out_channels, 1)

    # TODO consider applying hed2 iteratively
    def forward(self, x):
        out = []
        for i, hed in enumerate(self.heds):
            if i == 0:
                out.append(hed(x))
            else:
                out.append(hed(out[-1][0]))

        out = tuple(chain(*out))
        if self.fuse_sub_networks:
            fused = F.sigmoid(self.fuse(torch.cat(out, dim=1)))
            out = (fused,) + out
        return out
