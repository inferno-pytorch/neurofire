import numbers

import torch
from torch.autograd import Variable
from torch.nn.functional import conv3d

from inferno.io.transform import Transform

# TODO provide functionality to do trafos on gpu ?!?
# (for affinity trafos on gpu)


# TODO expect retain segmentation
class MaskIgnoreLabel(Transform):
    """
    """
    def __init__(self, ignore_label=0, **super_kwargs):
        super(MaskIgnoreLabel, self).__init__(**super_kwargs)
        assert isinstance(ignore_label, numbers.Integral)
        self.ignore_label = ignore_label

    # for all batch requests, we assume that
    # we are passed prediction and target in `tensors`
    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        # FIXME I am not sure if this does the right thing, need test !
        mask_variable = Variable(target.data.clone().ne(float(self.ignore_label)).float(),
                                 requires_grad=False).expand_as(prediction)
        masked_prediction = prediction * mask_variable
        return masked_prediction, target


class MaskTransitionToIgnoreLabel(Transform):
    """Applies a mask where the transition to zero label is masked for the respective offsets."""
    def __init__(self, offsets, ignore_label=0, **super_kwargs):
        super(MaskTransitionToIgnoreLabel, self).__init__(**super_kwargs)
        assert isinstance(offsets, (list, tuple))
        self.offsets = offsets
        assert isinstance(ignore_label, numbers.Integral)
        self.ignore_label = ignore_label

    # TODO explain what the hell is going on here ...
    @staticmethod
    def mask_shift_kernels(kernel, dim, offset):
        if dim == 3:
            assert len(offset) == 3
            s_z = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            s_x = 1 if offset[2] == 0 else (2 if offset[2] > 0 else 0)
            kernel[0, 0, s_z, s_y, s_x] = 1.
        elif dim == 2:
            assert len(offset) == 2
            s_x = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            kernel[0, 0, s_x, s_y] = 1.
        else:
            raise NotImplementedError
        return kernel

    def mask_tensor_for_offset(self, segmentation, offset):
        """
        Generate mask where a pixel is 1 if it's NOT a transition to ignore label
        AND not a ignore label itself.

        Example (ignore label 0)
        -------
        For
            offset       = 2,
            segmentation = 0 0 0 1 1 1 1 2 2 2 2 0 0 0
            affinity     = 0 1 1 1 1 0 0 1 1 0 0 1 1 0
            shift_mask   = 0 0 0 0 0 1 1 1 1 1 1 1 1 0
        --> final_mask   = 0 0 0 0 0 1 1 1 1 1 1 0 0 0
        """
        # expecting target to be segmentation of shape (N, 1, z, y, x)
        assert segmentation.size(1) == 1, str(segmentation.size())
        dim = 3  # TODO implement for 2d

        # Get mask where we don't have ignore label
        dont_ignore_labels_mask_variable = Variable(segmentation.data.clone().ne_(self.ignore_label),
                                                    requires_grad=False, volatile=True)
        shift_kernels = self.mask_shift_kernels(segmentation.data.new(1, 1, 3, 3, 3).zero_(), dim, offset)
        shift_kernels = Variable(shift_kernels, requires_grad=False)
        # Convolve
        abs_offset = tuple(max(1, abs(off)) for off in offset)
        mask_shifted = conv3d(input=dont_ignore_labels_mask_variable,
                              weight=shift_kernels,
                              padding=abs_offset, dilation=abs_offset)
        # Mask the mask tehe
        final_mask_tensor = (dont_ignore_labels_mask_variable
                             .expand_as(mask_shifted)
                             .data
                             .mul_(mask_shifted.data))
        return final_mask_tensor

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        # validate the prediction
        assert prediction.dim() == 5, prediction.dim()
        assert prediction.size(1) == len(self.offsets), "%i, %i" % (prediction.size(1), len(self.offsets))

        # validate target and extract segmentation from the target
        assert target.size(1) == len(self.offsets) + 1, "%i, %i" % (target.size(1), len(self.offsets) + 1)
        segmentation = target[:, 0:1]

        # get the individual mask for the offsets
        masks = [self.mask_tensor_for_offset(segmentation, offset) for offset in self.offsets]

        # Concatenate to one tensor
        master_mask = torch.cat(tuple(masks), 1)
        # Convert tensor to variable
        master_mask_variable = Variable(master_mask, requires_grad=False)
        # Mask prediction with master mask
        masked_prediction = prediction * master_mask_variable
        return masked_prediction, target
