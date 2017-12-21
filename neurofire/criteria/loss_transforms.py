import numbers

from torch.autograd import Variable

from inferno.io.transform import Transform

# TODO
# Implement loss transforms
# (prediction, target) -> (transformed_prediction, transformed_target)
# Implement by subclassing transform and providing batch or tensor function
# TODO how do we keep trafos for prediction / target optional ?

# TODO implement all the different masking functions

# TODO provide functionality to do trafos on gpu ?!?
# (for affinity trafos on gpu)


class MaskIgnoreLabel(Transform):
    """
    """
    def __init__(self, ignore_label, **super_kwargs):
        super(MaskIgnoreLabel, self).__init__(**super_kwargs)
        assert isinstance(ignore_label, numbers.Integral)
        self.ignore_label = ignore_label

    # for all batch requests, we assume that
    # we are passed prediction and target in `tensors`
    def batch_function(self, tensors):
        prediction, target = tensors
        # FIXME I am not sure if this does the right thing, need test !
        mask_variable = Variable(target.data.clone().eq(float(self.ignore_label)).float(),
                                 requires_grad=False).expand_as(prediction)
        masked_prediction = prediction * mask_variable
        return masked_prediction, target


class MaskTransitionToIgnoreLabel(Transform):
    """
    """
    pass
