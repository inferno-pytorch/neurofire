import torch.nn as nn


class LossWrapper(nn.Module):
    """
    Wrapper around a torch criterion.
    Enables transforms before applying the criterion.
    Should be subclassed for implementation.
    """
    def __init__(self, criterion, transforms=None):
        # validate: the criterion needs to inherit from nn.Module
        assert isinstance(criterion, nn.Module)
        self.criterion = criterion
        # validate: transforms need to be callable
        if transforms is not None:
            assert callable(transforms)
        self.transforms = transforms

    def forward(self, prediction, target):
        # we apply the transformations to the prediction
        # TODO in https://github.com/nasimrahaman/neuro-skunkworks/blob/more-cremi/skunkworks/datasets/cremi/criteria/euclidean.py#L36
        # we make a new variable `masked_prediction`. Is there a special reason for it (no gradients for inplace operations)
        # or could we also write `prediction = prediction`?
        if self.transforms is None:
            transformed_prediction = self.transforms(prediction, target)
        else:
            transformed_prediction = prediction

        loss = self.criterion(transformed_prediction, target)
        return loss


# TODO We need to have `loss transforms` analogous to the inferno `transforms`,
# but which take `prediction` and `target` as arguments and return
# `transformed_prediction`
# I don't really know how to best implement this.
# Should we subclass inferno transforms or should we make a new
# base class for this


# TODO something analogous to `AsSegmentationCriterion` from neuro-skunkworks to
# move loss preprocessing to the gpu ?!
