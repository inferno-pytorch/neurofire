import torch.nn as nn
import torch
import torch.nn.functional as F
from .hed2 import HED
from .dense_hed import DenseHED


class FusionHED(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 parameter,
                 block_type_key='default',
                 output_type_key='default',
                 sampling_type_key='default',
                 scaling_type_key='default'):
        assert isinstance(parameter, (int, list, tuple))
        if isinstance(parameter, (list, tuple)):
            assert len(parameter) == 2
            use_vamilla_hed = True
        else:
            use_vamilla_hed = False

        super(FusionHED, self).__init__()

        # need to have out channels as member for inference engine
        self.out_channels = out_channels

        if use_vamilla_hed:
            self.hed0 = HED(in_channels=in_channels, out_channels=out_channels,
                            initial_num_fmaps=parameter[0],
                            fmap_growth=parameter[1],
                            block_type_key=block_type_key,
                            output_type_key=output_type_key,
                            sampling_type_key=sampling_type_key)
            self.hed1 = HED(in_channels=in_channels, out_channels=out_channels,
                            initial_num_fmaps=parameter[0],
                            fmap_growth=parameter[1],
                            block_type_key=block_type_key,
                            output_type_key=output_type_key,
                            sampling_type_key=sampling_type_key)
            self.hed2 = HED(in_channels=in_channels, out_channels=out_channels,
                            initial_num_fmaps=parameter[0],
                            fmap_growth=parameter[1],
                            block_type_key=block_type_key,
                            output_type_key=output_type_key,
                            sampling_type_key=sampling_type_key)
        else:
            self.hed0 = DenseHED(in_channels=in_channels, out_channels=out_channels,
                                 N=parameter,
                                 block_type_key=block_type_key,
                                 output_type_key=output_type_key,
                                 sampling_type_key=sampling_type_key)
            self.hed1 = DenseHED(in_channels=in_channels, out_channels=out_channels,
                                 N=parameter,
                                 block_type_key=block_type_key,
                                 output_type_key=output_type_key,
                                 sampling_type_key=sampling_type_key)
            self.hed2 = DenseHED(in_channels=in_channels, out_channels=out_channels,
                                 N=parameter,
                                 block_type_key=block_type_key,
                                 output_type_key=output_type_key,
                                 sampling_type_key=sampling_type_key)

        self.downscale = HED.sampling_types[scaling_type_key][0]
        self.upscale = HED.sampling_types[scaling_type_key][1]
        self.fusion = HED.output_types[output_type_key](3*out_channels, out_channels, 1)

    def forward(self, x):
        ds = self.downscale
        us = self.upscale

        # upscaled branch
        d11, d12, d13, d14, d15, d16 = self.hed0(self.upscale(x))
        d11, d12, d13, d14, d15, d16 = ds(d11), ds(d12), ds(d13), ds(d14), ds(d15), ds(d16)

        # normal branch
        d21, d22, d23, d24, d25, d26 = self.hed1(x)

        # downscaled branch
        d31, d32, d33, d34, d35, d36 = self.hed2(ds(x))
        d31, d32, d33, d34, d35, d36 = us(d31), us(d32), us(d33), us(d34), us(d35), us(d36)

        d_final = self.fusion(torch.cat((d16, d26, d36), 1))
        out = F.sigmoid(d_final)

        return (out, d11, d12, d13, d14, d15, d16,
                d21, d22, d23, d24, d25, d26,
                d31, d32, d33, d34, d35, d36)
