import torch
import torch.nn as nn


class WNetSkeletonMultiscale(nn.Module):
    def __init__(self, encoders1, decoders1, base1,
                 encoders2, decoders2, base2, samplers,
                 predictors2, predictors1=None,
                 final_activation=None):
        super(WNetSkeletonMultiscale, self).__init__()
        assert isinstance(encoders1, list)
        assert isinstance(decoders1, list)
        assert isinstance(encoders2, list)
        assert isinstance(decoders2, list)
        assert isinstance(samplers, list)
        assert isinstance(base2, nn.Module)

        assert len(encoders1) == len(encoders2) == len(decoders1) == len(decoders2)
        assert len(encoders1) + 1 == len(predictors2)
        if predictors1 is not None:
            assert len(predictors1) == len(predictors2)
        self.encoders1 = nn.ModuleList(encoders1)
        self.decoders1 = nn.ModuleList(decoders1)
        self.encoders2 = nn.ModuleList(encoders2)
        self.decoders2 = nn.ModuleList(decoders2)
        self.samplers = nn.ModuleList(samplers)

        self.base1 = base1
        self.base2 = base2

        self.predictors2 = nn.ModuleList(predictors2)
        if predictors1 is not None:
            self.predict_first_decoders = True
            self.predictors1 = nn.ModuleList(predictors1)
        else:
            self.predict_first_decoders = False

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

    def first_decoder_pass(self, x, encoder_out1):
        # apply first base and remember its output
        x = self.base1(x)
        decoder_out1 = [self.samplers[0](x)]

        # apply decoders of the first decoder path
        # and remember their outputs
        max_level = len(self.decoders1) - 1
        for level, decoder in enumerate(self.decoders1):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(torch.cat((decoder_out1[-1],
                                   encoder_out1[max_level - level]), 1))
            decoder_out1.append(self.samplers[level](x))
        # reverse decoder output
        return x, decoder_out1[::-1]

    def first_decoder_pass_with_prediction(self, x, encoder_out1):
        # apply first base and remember its output
        x = self.base1(x)
        decoder_out1 = [self.samplers[0](x)]
        decoder_pred = [self.predictors1[-1](x)]

        # apply decoders of the first decoder path
        # and remember their outputs
        max_level = len(self.decoders1) - 1
        for level, decoder in enumerate(self.decoders1):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(torch.cat((decoder_out1[-1],
                                   encoder_out1[max_level - level],
                                   self.samplers[level](decoder_pred[-1])), 1))
            decoder_out1.append(self.samplers[level](x))
            decoder_pred.append(self.predictors1[max_level - level](x))
        # reverse decoder output
        return x, decoder_out1[::-1], decoder_pred[::-1]

    def forward(self, input_):
        x = input_
        encoder_out1 = []
        # apply encoders of the first encoder path
        # and remember their outputs
        for encoder in self.encoders1:
            x = encoder(x)
            encoder_out1.append(x)

        if self.predict_first_decoders:
            x, decoder_out1, decoder_pred1 = self.first_decoder_pass_with_prediction(x, encoder_out1)
        else:
            x, decoder_out1 = self.first_decoder_pass(x, encoder_out1)

        # apply the encoders of the second encoder path
        # they get additional input from the decoder path 1 and
        # their output is added to the output of the first encoders
        for level, encoder in enumerate(self.encoders2):
            if level == 0:
                x = encoder(x)
            else:
                x = encoder(torch.cat((x, decoder_out1[level]), 1))
            # add the output oup with the output from the previous encoder path
            encoder_out1[level] = encoder_out1[level] + x

        # apply the second base
        x = self.base2(torch.cat((x, decoder_out1[-1]), 1))
        outputs = [self.predictors2[-1](x)]

        # apply decoders of the second decoder path
        # and remember their outputs
        max_level = len(self.decoders1) - 1
        for level, decoder in enumerate(self.decoders2):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level, as well as the prediction of the previous decoder
            x = decoder(torch.cat((self.samplers[level](x),
                                   encoder_out1[max_level - level],
                                   self.samplers[level](outputs[-1])), 1))
            outputs.append(self.predictors2[max_level - level](x))

        outputs = [self.apply_act(out) for out in reversed(outputs)]
        if self.predict_first_decoders:
            outputs += [self.apply_act(dp1) for dp1 in decoder_pred1]
        return outputs
