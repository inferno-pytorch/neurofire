import torch.nn as nn


class LossWrapper(nn.Module):
    """
    Wrapper around a torch criterion.
    Enables transforms before applying the criterion.
    Should be subclassed for implementation.
    """
    def __init__(self,
                 criterion,
                 transforms=None,
                 weight_function=None):
        super(LossWrapper, self).__init__()
        # validate: the criterion needs to inherit from nn.Module
        assert isinstance(criterion, nn.Module)
        self.criterion = criterion
        # validate: transforms need to be callable
        if transforms is not None:
            assert callable(transforms)
        self.transforms = transforms
        if weight_function is not None:
            assert callable(weight_function)
        self.weight_function = weight_function

    def forward(self, prediction, target):
        # calculate the weight based on prediction and target
        if self.weight_function is not None:
            weight = self.weight_function(prediction, target)
            self.loss.weight = weight

        if self.transforms is not None:
            transformed_prediction, transformed_target = self.transforms(prediction, target)
        else:
            transformed_prediction, transformed_target = prediction, target

        loss = self.criterion(transformed_prediction, transformed_target)
        return loss


# TODO something analogous to `AsSegmentationCriterion` from neuro-skunkworks to
# move loss preprocessing to the gpu ?!
