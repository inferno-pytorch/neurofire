import torch
import torch.nn as nn


class UNetSkeletonMultiscale(nn.Module):
    def __init__(self, encoders, base, decoders, predictors, final_activation=None,
                 return_inner_feature_layers=False):
        super(UNetSkeletonMultiscale, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)

        assert isinstance(return_inner_feature_layers, bool)
        self._return_inner_feature_layers = return_inner_feature_layers

        # why do we hard-code this to 3 ? wouldn't it be enough to check that they are
        # all of the same length
        assert len(encoders) == len(decoders) == 3
        assert isinstance(base, nn.Module)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.predictors = nn.ModuleList(predictors)

        self.base = base
        if isinstance(final_activation, str):
            self.final_activation = getattr(nn, final_activation)()
        elif isinstance(final_activation, nn.Module):
            self.final_activation = final_activation
        elif final_activation is None:
            self.final_activation = None
        else:
            raise NotImplementedError

    def apply_act(self, input_):
        if self.final_activation is not None:
            return self.final_activation(input_)
        else:
            return input_

    def forward(self, input_):

        # all spatial sizes (ssize) for 512 input size
        # and donwscaling by a factor of 2
        # apply first decoder
        # e0.ssize = 256 (ssize / scale_factor)
        e0 = self.encoders[0](input_)

        # apply second encoder
        # e1.ssize = 128 (ssize / (scale_factor**2))
        e1 = self.encoders[1](e0)

        # apply third encoder
        # e2.ssize = 64 (ssize / (scale_factor**3))
        e2 = self.encoders[2](e1)

        # apply the base
        # b.ssize = 64
        b = self.base(e2)

        # apply the third / lowest decoder with input from base
        # and encoder 2
        # d2.ssize = 128
        p3 = self.predictors[3](b)
        out3 = self.apply_act(p3)

        d2 = self.decoders[0](
            torch.cat((b, e2, p3), 1)
        )
        # apply the second decoder with input from the third decoder
        # and the second encoder
        # d1.ssize = 256
        p2 = self.predictors[2](d2)
        out2 = self.apply_act(p2)
        d1 = self.decoders[1](
            torch.cat((d2, e1, p2), 1)
        )
        # apply the first decoder with input from the second decoder
        # and the first encoder
        # d0.ssize = 512
        p1 = self.predictors[1](d1)
        out1 = self.apply_act(p1)
        d0 = self.decoders[2](
            torch.cat((d1, e0, p1), 1)
        )

        # out.ssize = 512
        p0 = self.predictors[0](d0)
        out0 = self.apply_act(p0)

        if self._return_inner_feature_layers:
            return (out0, out1, out2, out3), (e0, e1, e2, b, d2, d1, d0)
        else:
            return out0, out1, out2, out3
