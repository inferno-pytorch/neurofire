import torch.nn as nn

from inferno.extensions.layers.sampling import AnisotropicPool
from .loss_wrapper import LossWrapper
from .loss_transform import InvertTarget


class MultiScaleLoss(nn.Module):
    # Steffen's arguments `offsets` and `scale_facter` were never used
    def __init__(self, loss, n_scales=4, scale_weights=None):
        super(MultiScaleLoss, self).__init__()
        assert isinstance(loss, LossWrapper)
        self.loss = loss
        self.n_scales = n_scales
        # per default, we weight each scale's loss with 1 / 4**scale_level i. (1, 1/4, 1/16, 1/128, ...)
        if scale_weights is None:
            self.scale_weights = [4**scale for scale in range(n_scales)]
        else:
            assert isinstance(scale_weights, (list, tuple))
            assert len(scale_weights) == n_scales
            self.scale_weights = scale_weights

        # For reference: Steffen's defaults
        # aff_trafo was never used
        # self.aff_trafo = ManySegmentationsToFuzzyAffinities(dim=2,
        #                             offsets=offsets, retain_segmentation=True)

        # self.loss = LossWrapper(criterion=SorensenDiceLoss(),
        #                     transforms=Compose(MaskTransitionToIgnoreLabel(offsets, ignore_label=0),
        #                                        RemoveSegmentationFromTarget(),
        #                                        InvertTarget()))

    def forward(self, predictions, targets):
        assert isinstance(predictions, (list, tuple))
        assert len(targets) == len(predictions)
        assert len(predictions) == self.n_scales

        # TODO make sure that this actually checks out with pytorch logics
        # calculate and add up the loss for each scale, weighted by the corresponding factor
        # (weighting factor 0 disables the scale)
        loss = sum([self.loss(ps, ts) / ws
                    for ps, ts, ws in zip(predictions, targets, self.scale_weights) if ws > 0])
        return loss


class MultiScaleLossMaxPool(MultiScaleLoss):
    # NOTE if invert target is activated here, it should not be added in the loss transforms
    # otherwise it will be applied twice
    def __init__(self, loss, n_scales=4, scale_weights=None, scaling_factor=3, invert_target=True):
        super(MultiScaleLossMaxPool, self).__init__(loss, n_scales, scale_weights)
        assert isinstance(scaling_factor, (list, tuple, int))
        if isinstance(scaling_factor, int):
            self.scaling_factor = self.n_scales * [scaling_factor]
        else:
            assert len(scaling_factor) == self.n_scales
            assert all(isinstance(sf, int) for sf in scaling_factor)
            self.scaling_factor = scaling_factor
        self.poolers = [AnisotropicPool(downscale_factor=sf) for sf in self.scaling_factor]

        # in case of affinities, we need to invert the targets BEFORE passing to
        # the loss to have the correct input to the max pooling
        if invert_target:
            self.inverter = InvertTarget()
        else:
            self.iverter = None

        # For reference: Alberto's defaults
        # self.transforms = Compose(MaskTransitionToIgnoreLabel(offsets, ignore_label=0),
        #                           RemoveSegmentationFromTarget(),
        #                           InvertTarget())
        # self.loss = SorensenDiceLoss()

    def forward(self, predictions, target):
        # TODO this is ALbertp's comment, which I don't get ...
        # Here the targets are affinities (i.e. 0 == split and 1 == merge).
        # And anyway for the max-pool we need the opposite (preserve all boundaries).
        assert isinstance(predictions, (list, tuple))
        assert len(predictions) == self.n_scales

        if self.inverter is not None:
            target = self.inverter(target)

        targets = []
        for scale in range(self.n_scales):
            target = self.poolers[scale](target)
            targets.append(target.clone())

        return super(MultiScaleLossMaxPool, self).forward(predictions, targets)
