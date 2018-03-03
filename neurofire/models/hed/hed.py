# This model was taken from https://github.com/xlliu7/hed.pytorch

import torch.nn as nn
import math
import torch
import torch.nn.functional as F

# def crop(d, g):
#     g_h, g_w = g.size()[2:4]
#     d_h, d_w = d.size()[2:4]
#     d1 = d[:, :, int(math.floor((d_h - g_h)/2.0)):int(math.floor((d_h - g_h)/2.0)) + g_h, int(math.floor((d_w - g_w)/2.0)):int(math.floor((d_w - g_w)/2.0)) + g_w]
#     return d1

class HED(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dilation=1):
        super(HED, self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv2 = nn.Sequential(
            # conv2
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv3 = nn.Sequential(
            # conv3
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv4 = nn.Sequential(
            # conv4
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        
        self.dsn1 = nn.Conv2d( 64, out_channels, 1)
        self.dsn2 = nn.Conv2d(128, out_channels, 1)
        self.dsn3 = nn.Conv2d(256, out_channels, 1)
        self.dsn4 = nn.Conv2d(512, out_channels, 1)
        self.dsn5 = nn.Conv2d(512, out_channels, 1)
        self.dsn6 = nn.Conv2d(  5*out_channels, out_channels, 1)
        
        self.upscore2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upscore4 = nn.UpsamplingBilinear2d(scale_factor=8)
        
        if dilation > 1:
            self.conv5 = nn.Sequential(
                # conv5
                nn.MaxPool2d(2, stride=1,  padding=1, ceil_mode=False),  # 1/8
                nn.Conv2d(512, 512, 3, padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True),
            )
            self.upscore5 = nn.UpsamplingBilinear2d(scale_factor=8) 
        else:
            self.conv5 = nn.Sequential(
                # conv5
                nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.upscore5 = nn.UpsamplingBilinear2d(scale_factor=16)  

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        ## side output
        d5 = self.upscore5(self.dsn5(conv5))
        # d5 = crop(dsn5_up, gt)
        
        d4 = self.upscore4(self.dsn4(conv4))
        # d4 = crop(dsn4_up, gt)
        
        d3 = self.upscore3(self.dsn3(conv3))
        # d3 = crop(dsn3_up, gt)
        
        d2 = self.upscore2(self.dsn2(conv2))
        # d2 = crop(dsn2_up, gt)
        
        d1 = self.dsn1(conv1)
        # d1 = crop(dsn1, gt)

        # dsn fusion output
        d6 = self.dsn6(torch.cat((d1, d2, d3, d4, d5), 1))
        
        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        d6 = F.sigmoid(d6)

        return d1, d2, d3, d4, d5, d6
