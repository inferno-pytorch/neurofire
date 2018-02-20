import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from .hed import HED

class FusionHED(nn.Module):
    def __init__(self, out_channels=1, dilation=1, scale_factor=(2, 2, 2)):
        self.hed1 = HED(out_channels=out_channels, dilation=dilation)
        self.hed2 = HED(out_channels=out_channels, dilation=dilation)
        self.hed3 = HED(out_channels=out_channels, dilation=dilation)
        self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.ds = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.fusion = nn.Conv2d(3*out_channels, out_channels, 1)

    def forward(x)
        d11, d12, d13, d14, d15, d16 = self.ds(self.hed1(self.upscale(x)))
        d21, d22, d23, d24, d25, d26 = self.hed2(x)
        d31, d32, d33, d34, d35, d36 = self.upscale(self.hed2(self.ds(x)))
        d_final = self.fusion(torch.cat((d16, d26, d36), 1))

        return d11, d12, d13, d14, d15, d16, d21, d22, d23, d24, d25, d26, d31, d32, d33, d34, d35, d36, d_final
