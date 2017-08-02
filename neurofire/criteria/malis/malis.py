import numpy as np
import torch
from torch.autograd import Function

from .malis_impl.bld._malis_impl import malis_impl, constrained_malis_impl

# malis loss and cpp impl adapted from:
# https://github.com/naibaf7/caffe/blob/master/include/caffe/layers/malis_loss_layer.hpp
# https://github.com/naibaf7/caffe/blob/master/src/caffe/layers/malis_loss_layer.cpp

# TODO: the caffee implementation parallelizes over the batches in the backward / forward pass
# I have no idea how batches are handled in pytorch, so I haven't done any parallelization yet
# However the gil for 'malis_impl' is lifted, so this CAN be used in multiple threads

# TODO as far as I can tell, this is normal MALIS. I can't see, where in the caffe impl
# constrained MALIS (aka MALA) comes into play

# To create new loss, we inherit from Function:
# https://github.com/pytorch/pytorch/blob/master/torch/autograd/function.py#L123
# see also
# http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html


class MalisLoss(Function):
    """
    Malis Loss
    """

    def __init__(self):
        super(MalisLoss, self).__init__()
        self._intermediates = {}

    def forward(self, affinities, groundtruth):
        """
        Apply malis forward pass to get the loss.

        Parameters
        ----------
        affinities : torch.Variable wrapping the affinity tensor
        groundtruth : torch.Variable wrapping the groundtruth tensor

        Returns
        -------
        loss: malis loss
        """
        # Convert input to numpy
        affinities = affinities.numpy()
        groundtruth = groundtruth.numpy()
        # Store shapes for backward
        self._intermediates.update({'affinities_shape': affinities.shape,
                                    'groundtruth_shape': groundtruth.shape})

        # fist, compute the positive loss and gradients
        pos_gradients, pos_loss, _, _ = malis_impl(
            affinities, groundtruth, True
        )

        # next, compute the negative loss and gradients
        neg_gradients, neg_loss, _, _ = malis_impl(
            affinities, groundtruth, False
        )

        # save the combined gradient for the backward pass
        combined_gradient = -(neg_gradients + pos_gradients) / 2.
        self._intermediates.update({'combined_gradient': combined_gradient})

        # get the combined loss
        combined_loss = (neg_loss + pos_loss) / 2.
        # return
        return torch.from_numpy(np.asarray([combined_loss]))

    def backward(self, grad_output):
        """
        Apply malis backward pass to get the gradients.
        """
        # Fetch gradient from intermediates
        gradients = self._intermediates.get('combined_gradient')
        assert gradients is not None
        # Make sure the shape is correct
        assert gradients.shape == self._intermediates.get('affinities_shape')
        # Make a zero variable for the target
        target_gradient = np.zeros(shape=self._intermediates.get('groundtruth_shape'))
        # Clear intermediates
        self._intermediates.clear()
        return torch.from_numpy(gradients), torch.from_numpy(target_gradient)


class ConstrainedMalisLoss(Function):
    """
    Constrained Malis Loss
    """

    def forward(self, affinities, groundtruth):
        """
        Apply constrained malis forward pass to get the loss.

        Parameters
        ----------
        affinities : torch.Variable wrapping the affinity tensor
        groundtruth : torch.Variable wrapping the groundtruth tensor

        Returns
        -------
        loss: malis loss
        """

        gradients, loss = constrained_malis_impl(affinities, groundtruth)
        self.save_for_backward(-gradients / 2.)
        return loss

    def backward(self, grad_output):
        """
        Apply malis backward pass to get the gradients.
        """

        gradients, = self.saved_tensors
        return gradients
