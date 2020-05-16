import torch
import torch.nn as nn

#
# resnets for 3d data. Adapted from @constantinpape adaptation of
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#


__all__ = ['ResNet3D', 'resnet3d18', 'resnet3d34', 'resnet3d50', 'resnet3d101',
           'resnet3d152', 'resnext3d50_32x4d', 'resnext3d101_32x8d',
           'wide_resnet3d50_2', 'wide_resnet3d101_2']


def get_norm(norm_funct, channels, groups=None):
    if isinstance(norm_funct, str):
        norm_funct = getattr(nn, norm_funct)
    if norm_funct == nn.GroupNorm:
        assert groups is not None, "GroupNorm requires specifying the number of groups"
        # fails if groups > channels
        return norm_funct(min(groups, channels), channels)
    else:
        return norm_funct(channels)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, norm_groups=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = get_norm(norm_layer, planes, norm_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = get_norm(norm_layer, planes, norm_groups)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, norm_groups=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.norm1 = get_norm(norm_layer, width, norm_groups)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.norm2 = get_norm(norm_layer, width, norm_groups)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.norm3 = get_norm(norm_layer, planes * self.expansion, norm_groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self, block, layers, inchannels=1, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, norm_groups=None):
        super(ResNet3D, self).__init__()
        if block == "BasicBlock":
            block = BasicBlock
        elif block == "Bottleneck":
            block = Bottleneck
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.norm_groups = norm_groups
        self.inchannels = inchannels
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # TODO not a good idea in 3d ?!
        self.conv1 = nn.Conv3d(self.inchannels, self.inplanes,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.norm1 = get_norm(norm_layer, self.inplanes, self.norm_groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.norm3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        num_groups = self.norm_groups
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                get_norm(norm_layer, planes * block.expansion, num_groups),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, num_groups))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, norm_groups=num_groups))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet3D(block, layers, **kwargs)
    return model


def resnet3d18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

    Adapted for 3d data.
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet3d34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

    Adapted for 3d data.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet3d50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

    Adapted for 3d data.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet3d101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

    Adapted for 3d data.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet3d152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`

    Adapted for 3d data.
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext3d50_32x4d(**kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks"
    <https://arxiv.org/pdf/1611.05431.pdf>`

    Adapted for 3d data.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext3d101_32x8d(**kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`

    Adapted for 3d data.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet3d50_2(**kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Adapted for 3d data.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet3d101_2(**kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Adapted for 3d data.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
