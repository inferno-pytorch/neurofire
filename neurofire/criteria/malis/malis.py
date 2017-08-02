import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from concurrent.futures import ThreadPoolExecutor
from inferno.extensions.layers.device import DeviceTransfer
import os

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

    def _wrapper(self, affinities, groundtruth):
        # fist, compute the positive loss and gradients
        pos_gradients, _, _, _ = malis_impl(
            affinities, groundtruth, True)

        # next, compute the negative loss and gradients
        neg_gradients, _, _, _ = malis_impl(
            affinities, groundtruth, False)
        return pos_gradients, neg_gradients

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
        # TODO insert asserts
        # Convert input to numpy
        affinities = affinities.numpy()
        groundtruth = groundtruth.numpy()
        # Store shapes for backward
        self._intermediates.update({'affinities_shape': affinities.shape,
                                    'groundtruth_shape': groundtruth.shape})

        # Parallelize over the leading batch axis
        all_affinities = list(affinities)
        all_groundtruth = list(groundtruth)

        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1)) as executor:
            all_pos_and_neg_gradients = list(executor.map(self._wrapper,
                                                          all_affinities,
                                                          all_groundtruth))
        all_pos_gradients, all_neg_gradients = zip(*all_pos_and_neg_gradients)
        pos_gradients = np.array(all_pos_gradients)
        neg_gradients = np.array(all_neg_gradients)
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

    def _wrapper(self, affinities, groundtruth):
        gradients, _ = constrained_malis_impl(affinities, groundtruth)
        return gradients

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
        all_affinities = list(affinities)
        all_groundtruth = list(groundtruth)
        # Distribute over threads (GIL is lifted)
        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1)) as executor:
            all_gradients = list(executor.map(self._wrapper, all_affinities, all_groundtruth))
        # Build arrays and tensors
        gradients = np.array(all_gradients)
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

    Notes
    -----
    Malis is only implemented on the CPU. Variables on the GPU will be transfered to the CPU in
    the forward pass, and their gradients back to the GPU in the backward pass.
    Also, the resulting pseudo loss is on the CPU.
    """
    def __init__(self, constrained=True):
        """
        Parameters
        ----------
        constrained : bool
            Whether to use constrained MALIS.
        """
        super(Malis, self).__init__()
        self.constrained = constrained
        self.device_transfer = DeviceTransfer('cpu')
        if constrained:
            self.malis_loss = ConstrainedMalisLoss()
        else:
            self.malis_loss = MalisLoss()

    # noinspection PyCallingNonCallable
    def forward(self, input, target):
        input, target = self.device_transfer(input, target)
        loss_gradients = self.malis_loss(input, target)
        pseudo_loss = loss_gradients.sum()
        return pseudo_loss
