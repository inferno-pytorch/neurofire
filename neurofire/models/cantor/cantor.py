import torch.nn as nn
import inferno.extensions.layers.convolutional as conv
import inferno.extensions.layers.reshape as shape
from inferno.extensions.containers.graph import Graph
from inferno.extensions.containers.sequential import Sequential2
from ..base import Model


# Presets
class Upsample(nn.UpsamplingNearest2d):
    def __init__(self):
        super(Upsample, self).__init__(scale_factor=2)


class Pool(nn.MaxPool2d):
    def __init__(self):
        super(Pool, self).__init__(kernel_size=3, stride=2, padding=1)


# Modules
class L3SA(nn.Module):
    def __init__(self, base_width, kernel_size=3):
        super(L3SA, self).__init__()
        self.in_channels = 4 * base_width
        self.out_channels = 4 * base_width
        self.conv = conv.ConvELU2D(self.in_channels, self.out_channels, kernel_size)

    def forward(self, input):
        # noinspection PyCallingNonCallable
        return self.conv(input)


class L3SB(nn.Module):
    def __init__(self, base_width, kernel_size=3):
        super(L3SB, self).__init__()
        self.in_channels = (4 + 3) * base_width
        self.out_channels = 4 * base_width
        self.cat = shape.Concatenate()
        self.upsample = Upsample()
        self.conv = conv.ConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 2
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        upsampled = self.upsample(conved)
        return conved, upsampled


class L3SX(nn.Module):
    def __init__(self, base_width, kernel_size=3):
        super(L3SX, self).__init__()
        self.in_channels = 3 * base_width
        self.out_channels = 4 * base_width
        self.upsample = Upsample()
        self.conv = conv.ConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, input):
        conved = self.conv(input)
        upsampled = self.upsample(conved)
        return conved, upsampled


class L3SY(nn.Module):
    def __init__(self, base_width, kernel_size=3):
        super(L3SY, self).__init__()
        self.in_channels = (4 + 3) * base_width
        self.out_channels = 4 * base_width
        self.cat = shape.Concatenate()
        self.upsample = Upsample()
        self.conv = conv.ConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 2
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        upsampled = self.upsample(conved)
        return upsampled


class L2SA(nn.Module):
    def __init__(self, base_width, kernel_size=3):
        super(L2SA, self).__init__()
        self.in_channels = (4 + 3) * base_width
        self.out_channels = 3 * base_width
        self.cat = shape.Concatenate()
        self.pool = Pool()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 2
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        pooled = self.pool(conved)
        return pooled, conved


class L2SB(nn.Module):
    def __init__(self, base_width, kernel_size=3):
        super(L2SB, self).__init__()
        self.in_channels = (4 + 3 + 2) * base_width
        self.out_channels = 3 * base_width
        self.cat = shape.Concatenate()
        self.upsample = Upsample()
        self.pool = Pool()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 3
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        upsampled = self.upsample(conved)
        pooled = self.pool(conved)
        return pooled, conved, upsampled


class L2SX(nn.Module):
    def __init__(self, base_width, kernel_size=3):
        super(L2SX, self).__init__()
        self.in_channels = 2 * base_width
        self.out_channels = 3 * base_width
        self.upsample = Upsample()
        self.pool = Pool()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, input):
        conved = self.conv(input)
        pooled = self.pool(conved)
        upsampled = self.upsample(conved)
        return pooled, conved, upsampled


class L2SY(nn.Module):
    def __init__(self, base_width, kernel_size=3):
        super(L2SY, self).__init__()
        self.in_channels = (4 + 3 + 2) * base_width
        self.out_channels = 3 * base_width
        self.cat = shape.Concatenate()
        self.upsample = Upsample()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 3
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        upsampled = self.upsample(conved)
        return upsampled


class L1SA(nn.Module):
    def __init__(self, base_width, kernel_size=5):
        super(L1SA, self).__init__()
        self.in_channels = (3 + 2) * base_width
        self.out_channels = 2 * base_width
        self.cat = shape.Concatenate()
        self.pool = Pool()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 2
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        pooled = self.pool(conved)
        return pooled, conved


class L1SB(nn.Module):
    def __init__(self, base_width, kernel_size=5):
        super(L1SB, self).__init__()
        self.in_channels = (3 + 2 + 1) * base_width
        self.out_channels = 2 * base_width
        self.cat = shape.Concatenate()
        self.upsample = Upsample()
        self.pool = Pool()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 3
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        pooled = self.pool(conved)
        upsampled = self.upsample(conved)
        return pooled, conved, upsampled


class L1SX(nn.Module):
    def __init__(self, base_width, kernel_size=5):
        super(L1SX, self).__init__()
        self.in_channels = 1 * base_width
        self.out_channels = 2 * base_width
        self.upsample = Upsample()
        self.pool = Pool()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, input):
        conved = self.conv(input)
        pooled = self.pool(conved)
        upsampled = self.upsample(conved)
        return pooled, conved, upsampled


class L1SY(nn.Module):
    def __init__(self, base_width, kernel_size=5):
        super(L1SY, self).__init__()
        self.in_channels = (3 + 2 + 1) * base_width
        self.out_channels = 2 * base_width
        self.cat = shape.Concatenate()
        self.upsample = Upsample()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 3
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        upsampled = self.upsample(conved)
        return upsampled


class L0SA(nn.Module):
    def __init__(self, base_width, kernel_size=7):
        super(L0SA, self).__init__()
        self.in_channels = (2 + 1) * base_width
        self.out_channels = base_width
        self.cat = shape.Concatenate()
        self.pool = Pool()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 2
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        pooled = self.pool(conved)
        return pooled, conved


class L0SX(nn.Module):
    def __init__(self, in_channels, base_width, kernel_size=7):
        super(L0SX, self).__init__()
        self.in_channels = in_channels
        self.out_channels = base_width
        self.pool = Pool()
        self.conv = conv.DilatedConvELU2D(self.in_channels, self.out_channels, kernel_size)

    # noinspection PyCallingNonCallable
    def forward(self, input):
        conved = self.conv(input)
        pooled = self.pool(conved)
        return pooled, conved


class L0SY(nn.Module):
    def __init__(self, out_channels, base_width, kernel_size=7, activation=None):
        super(L0SY, self).__init__()
        self.in_channels = (2 + 1) * base_width
        self.out_channels = out_channels
        self.cat = shape.Concatenate()
        self.conv = conv.Conv2D(self.in_channels, self.out_channels, kernel_size)
        assert activation in [None, 'Sigmoid', 'Softmax']
        if activation is None:
            self.activation = None
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax2d()
        else:
            raise NotImplementedError

    # noinspection PyCallingNonCallable
    def forward(self, *inputs):
        assert len(inputs) == 2
        cated = self.cat(*inputs)
        conved = self.conv(cated)
        if self.activation is not None:
            activated = self.activation(conved)
        else:
            activated = conved
        return activated


class CantorModule(Graph):
    # noinspection PyTypeChecker
    def __init__(self, base_width):
        super(CantorModule, self).__init__()
        self.base_width = base_width
        # Build graph
        # IO
        # Add input nodes
        self.add_input_node('l3i3')
        self.add_input_node('l2i3')
        self.add_input_node('l2i2')
        self.add_input_node('l1i2')
        self.add_input_node('l1i1')
        self.add_input_node('l0i1')
        self.add_input_node('l0i0')
        # Add output nodes
        self.add_output_node('l3o3')
        self.add_output_node('l2o3')
        self.add_output_node('l2o2')
        self.add_output_node('l1o2')
        self.add_output_node('l1o1')
        self.add_output_node('l0o1')
        self.add_output_node('l0o0')

        # Stage 0
        # Add stage 0 nodes
        self.add_node('l3s0', L3SA(self.base_width))
        self.add_node('l2s0', L2SA(self.base_width))
        self.add_node('l1s0', L1SA(self.base_width))
        self.add_node('l0s0', L0SA(self.base_width))
        # Add stage 0 edges
        self.add_edge('l3i3', 'l3s0')
        self.add_edge('l2i3', 'l2s0')
        self.add_edge('l2i2', 'l2s0')
        self.add_edge('l1i2', 'l1s0')
        self.add_edge('l1i1', 'l1s0')
        self.add_edge('l0i1', 'l0s0')
        self.add_edge('l0i0', 'l0s0')

        # Stage 1
        # Add stage 1 nodes
        self.add_node('l3s1', L3SB(self.base_width))
        # Add stage 1 edges
        self.add_edge('l3s0', 'l3s1')
        self.add_edge('l2s0', 'l3s1')

        # Stage 2
        # Add stage 2 nodes
        self.add_node('l3s2', L3SA(self.base_width))
        self.add_node('l2s2', L2SB(self.base_width))
        # Add stage 2 edges
        self.add_edge('l3s1', 'l3s2')
        self.add_edge('l3s1', 'l2s2')
        self.add_edge('l2s0', 'l2s2')
        self.add_edge('l1s0', 'l2s2')

        # Stage 3
        # Add stage 3 nodes
        self.add_node('l3s3', L3SB(self.base_width))
        # Add stage 3 edges
        self.add_edge('l3s2', 'l3s3')
        self.add_edge('l2s2', 'l3s3')

        # Stage 4
        # Add stage 4 nodes
        self.add_node('l3s4', L3SA(self.base_width))
        self.add_node('l2s4', L2SA(self.base_width))
        self.add_node('l1s4', L1SB(self.base_width))
        # Add stage 4 edges
        self.add_edge('l3s3', 'l3s4')
        self.add_edge('l3s3', 'l2s4')
        self.add_edge('l2s2', 'l2s4')
        self.add_edge('l2s2', 'l1s4')
        self.add_edge('l1s0', 'l1s4')
        self.add_edge('l0s0', 'l1s4')

        # Stage 5
        # Add stage 5 nodes
        self.add_node('l3s5', L3SB(self.base_width))
        # Add stage 5 edges
        self.add_edge('l3s4', 'l3s5')
        self.add_edge('l2s4', 'l3s5')

        # Stage 6
        # Add stage 6 nodes
        self.add_node('l3s6', L3SA(self.base_width))
        self.add_node('l2s6', L2SB(self.base_width))
        # Add stage 2 edges
        self.add_edge('l3s5', 'l3s6')
        self.add_edge('l3s5', 'l2s6')
        self.add_edge('l2s4', 'l2s6')
        self.add_edge('l1s4', 'l2s6')

        # Stage 7
        # Add stage 7 nodes
        self.add_node('l3s7', L3SB(self.base_width))
        # Add stage 7 edges
        self.add_edge('l3s6', 'l3s7')
        self.add_edge('l2s6', 'l3s7')

        # Stage Out
        # Add stage out edges
        self.add_edge('l3s7', 'l3o3')
        self.add_edge('l3s7', 'l2o3')
        self.add_edge('l2s6', 'l2o2')
        self.add_edge('l2s6', 'l1o2')
        self.add_edge('l1s4', 'l1o1')
        self.add_edge('l1s4', 'l0o1')
        self.add_edge('l0s0', 'l0o0')


class CantorInitiator(Graph):
    # noinspection PyTypeChecker
    def __init__(self, in_channels, base_width):
        super(CantorInitiator, self).__init__()
        self.base_width = base_width
        # Build graph
        # IO
        # Add input nodes
        self.add_input_node('input')
        # Add output nodes
        self.add_output_node('l3o3')
        self.add_output_node('l2o3')
        self.add_output_node('l2o2')
        self.add_output_node('l1o2')
        self.add_output_node('l1o1')
        self.add_output_node('l0o1')
        self.add_output_node('l0o0')

        # Add nodes
        self.add_node('l3sx', L3SX(self.base_width))
        self.add_node('l2sx', L2SX(self.base_width))
        self.add_node('l1sx', L1SX(self.base_width))
        self.add_node('l0sx', L0SX(in_channels, self.base_width))

        # Add edges
        self.add_edge('input', 'l0sx')
        self.add_edge('l0sx', 'l1sx')
        self.add_edge('l1sx', 'l2sx')
        self.add_edge('l2sx', 'l3sx')

        self.add_edge('l3sx', 'l3o3')
        self.add_edge('l3sx', 'l2o3')
        self.add_edge('l2sx', 'l2o2')
        self.add_edge('l2sx', 'l1o2')
        self.add_edge('l1sx', 'l1o1')
        self.add_edge('l1sx', 'l0o1')
        self.add_edge('l0sx', 'l0o0')


class CantorTerminator(Graph):
    # noinspection PyTypeChecker
    def __init__(self, out_channels, base_width, activation=None):
        super(CantorTerminator, self).__init__()
        self.base_width = base_width
        # Build graph
        # Add input nodes
        self.add_input_node('l3i3')
        self.add_input_node('l2i3')
        self.add_input_node('l2i2')
        self.add_input_node('l1i2')
        self.add_input_node('l1i1')
        self.add_input_node('l0i1')
        self.add_input_node('l0i0')

        # Add stage 0 nodes
        self.add_node('l3s0', L3SA(self.base_width))
        self.add_node('l2s0', L2SA(self.base_width))
        self.add_node('l1s0', L1SA(self.base_width))
        self.add_node('l0s0', L0SA(self.base_width))
        # Add stage 0 edges
        self.add_edge('l3i3', 'l3s0')
        self.add_edge('l2i3', 'l2s0')
        self.add_edge('l2i2', 'l2s0')
        self.add_edge('l1i2', 'l1s0')
        self.add_edge('l1i1', 'l1s0')
        self.add_edge('l0i1', 'l0s0')
        self.add_edge('l0i0', 'l0s0')

        # Add exit nodes
        self.add_node('l3sy', L3SY(self.base_width))
        self.add_node('l2sy', L2SY(self.base_width))
        self.add_node('l1sy', L1SY(self.base_width))
        self.add_node('l0sy', L0SY(out_channels, self.base_width,
                                   activation=activation))

        # Add edges
        self.add_edge('l3s0', 'l3sy')
        self.add_edge('l2s0', 'l3sy')
        self.add_edge('l3sy', 'l2sy')
        self.add_edge('l2s0', 'l2sy')
        self.add_edge('l1s0', 'l2sy')
        self.add_edge('l2sy', 'l1sy')
        self.add_edge('l1s0', 'l1sy')
        self.add_edge('l0s0', 'l1sy')
        self.add_edge('l1sy', 'l0sy')
        self.add_edge('l0s0', 'l0sy')

        # Add output
        self.add_output_node('output', previous='l0sy')


class Cantor(Sequential2, Model):
    """Construct a Cantor network."""
    def __init__(self, in_channels, out_channels, base_width, num_modules,
                 output_activation='Sigmoid', dim=3):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels to the __2D__ layer - for 3D inputs, this is the number
            of input channels times the number of slices. For example if the input shape is
            [1, 2, 3, 512, 512], in_channels = 2 * 3 = 6.
        out_channels : int
            Number of output channels from the __2D__ layer. For 3D inputs, this is the number
            of z slices, i.e. multi-channel 3D outputs is not supported.
        base_width : int
            Number of units in the lower-most (i.e. highest resolution) level.
        num_modules : int
            Number of cantor blocks. Does not include the initiator and terminator.
        output_activation : {'Sigmoid', 'Softmax', None}
            Output activation,
        dim : {2, 3}
            Dimensionality of the input and output.
        """
        assert dim in [2, 3]
        modules = ([shape.As2D(z_as_channel=True)] if dim == 3 else []) + \
                   [CantorInitiator(in_channels, base_width)] + \
                   [CantorModule(base_width) for _ in range(num_modules)] + \
                   [CantorTerminator(out_channels, base_width, activation=output_activation)] + \
                  ([shape.As3D(channel_as_z=True)] if dim == 3 else [])
        super(Cantor, self).__init__(*modules)

    @classmethod
    def from_shape(cls, input_shape, base_width, num_modules, output_activation='Sigmoid',
                   output_shape=None):
        # Check if 2D or 3D
        if len(input_shape) == 5:
            num_input_channels = input_shape[1]
            num_z_slices = input_shape[2]
            assert None not in [num_input_channels, num_z_slices]
            in_channels = num_input_channels * num_z_slices
            out_channels = num_z_slices
            if output_shape is not None:
                assert len(output_shape) == 5
                assert output_shape[1] == 1
                assert output_shape[2] == num_z_slices
            return cls(in_channels, out_channels, base_width,
                       num_modules, output_activation, dim=3)
        elif len(input_shape) == 4:
            assert output_shape is not None
            in_channels = input_shape[1]
            out_channels = output_shape[1]
            assert None not in [in_channels, out_channels]
            return cls(in_channels, out_channels, base_width,
                       num_modules, output_activation, dim=2)
        else:
            raise NotImplementedError

    @classmethod
    def from_config(cls, **config):
        return cls.from_shape(**config)
