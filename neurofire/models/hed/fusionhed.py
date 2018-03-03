import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from .hed import HED

class FusionHED(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dilation=1):
        super(FusionHED, self).__init__()
        self.out_channels = out_channels
        self.hed1 = HED(in_channels=in_channels, out_channels=out_channels, dilation=dilation)
        self.hed2 = HED(in_channels=in_channels, out_channels=out_channels, dilation=dilation)
        self.hed3 = HED(in_channels=in_channels, out_channels=out_channels, dilation=dilation)
        self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.downscale = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.fusion = nn.Conv2d(3*out_channels, out_channels, 1)

    def forward(self, x):
        ds = self.downscale
        us = self.upscale

        # upscaled branch
        d11, d12, d13, d14, d15, d16 = self.hed1(self.upscale(x))
        d11, d12, d13, d14, d15, d16 = ds(d11), ds(d12), ds(d13), ds(d14), ds(d15), ds(d16)

        # normal branch
        d21, d22, d23, d24, d25, d26 = self.hed2(x)

        # downscaled branch
        d31, d32, d33, d34, d35, d36 = self.hed3(ds(x))
        d31, d32, d33, d34, d35, d36 = us(d31), us(d32), us(d33), us(d34), us(d35), us(d36)

        d_final = self.fusion(torch.cat((d16, d26, d36), 1))
        self.output = F.sigmoid(d_final)
        
        return d11, d12, d13, d14, d15, d16, d21, d22, d23, d24, d25, d26, d31, d32, d33, d34, d35, d36, self.output
