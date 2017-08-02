import numpy as np
import torch
import torch.nn as nn
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
        Apply malis forward pass to get the loss gradient. The final loss function is then
        the sum of this function's output.

        Parameters
        ----------
        affinities : torch.Tensor wrapping the affinity tensor
        groundtruth : torch.Tensor wrapping the groundtruth tensor

        Returns
        -------
        loss: malis loss gradients
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
        # the trailing .mul(1) makes a copy of the numpy tensor, without which pytorch segfaults
        # yes, i've aged figuring this out
        combined_gradient = torch.from_numpy(-(neg_gradients + pos_gradients) / 2.).mul(1)
        # Short story: No combined loss. Long story: torch doesn't allow saving a non-input or
        # non-output variable for backward (boohoo). If we save the numpy array as a python
        # variable (i.e. avoid save_for_backward), the backward pass segfaults.
        self.save_for_backward(combined_gradient)
        # return
        return combined_gradient

    def backward(self, grad_output):
        """
        Apply malis backward pass to get the gradients.
        """
        # Fetch gradient from intermediates
        gradients, = self.saved_tensors
        target_gradient = gradients.new(*self._intermediates.get('groundtruth_shape')).zero_()
        # Clear intermediates
        self._intermediates.clear()
        return gradients, target_gradient


class ConstrainedMalisLoss(Function):
    """
    Constrained Malis Loss
    """

    def __init__(self):
        super(ConstrainedMalisLoss, self).__init__()
        # Store shapes for backward
        self._intermediates = {}

    def forward(self, affinities, groundtruth):
        """
        Apply constrained malis forward pass to get the loss gradients.

        Parameters
        ----------
        affinities : torch.Tensor wrapping the affinity tensor
        groundtruth : torch.Tensor wrapping the groundtruth tensor

        Returns
        -------
        loss: malis loss
        """
        # Convert to numpy
        affinities = affinities.numpy()
        groundtruth = groundtruth.numpy()
        # Store shapes for backward
        self._intermediates.update({'affinities_shape': affinities.shape,
                                    'groundtruth_shape': groundtruth.shape})
        # Compute gradients
        gradients, loss = constrained_malis_impl(affinities, groundtruth)
        gradients = torch.from_numpy(gradients).mul(-0.5)
        self.save_for_backward(gradients)
        return gradients

    def backward(self, grad_output):
        """
        Apply malis backward pass to get the gradients.
        """
        gradients, = self.saved_tensors
        target_gradient = gradients.new(*self._intermediates.get('groundtruth_shape')).zero_()
        return gradients, target_gradient


class Malis(nn.Module):
    """
    This computes the Malis pseudo-loss, which is defined such that the backprop
    deltas are correct.
    """
    def __init__(self, constrained=True):
        super(Malis, self).__init__()
        self.constrained = constrained
        if constrained:
            self.malis_loss = ConstrainedMalisLoss()
        else:
            self.malis_loss = MalisLoss()

    def forward(self, input, target):
        loss_gradients = self.malis_loss(input, target)
        pseudo_loss = loss_gradients.sum()
        return pseudo_loss
