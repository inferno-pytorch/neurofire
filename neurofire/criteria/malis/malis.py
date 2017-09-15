import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from concurrent.futures import ThreadPoolExecutor
from inferno.extensions.layers.device import DeviceTransfer
import os

try:
    from custom_malis import malis, constrained_malis, constrained_malis_custom_nh
except ImportError:
    raise ImportError("Malis loss is only supported when the custom malis module is available")


class _MalisBase(Function):
    def __init__(self, dim=None, malis_dim='auto'):
        super(_MalisBase, self).__init__()
        self._dim = dim
        self._malis_dim = malis_dim
        self._intermediates = {}

    def parse_dim(self, affinities_ndim):
        if self._dim is None:
            self._dim = {5: 3, 4: 2}.get(affinities_ndim)

    @property
    def dim(self):
        return self._dim

    @property
    def tensor_ndim(self):
        return {2: 4, 3: 5}.get(self._dim)

    @property
    def malis_dim(self):
        if self._malis_dim == 'auto':
            return self._dim
        else:
            return self._malis_dim


class MalisLoss(_MalisBase):
    """
    Malis Loss
    """

    def _wrapper(self, affinities, groundtruth):
        # The malis wrapper expects 2D or 3D groundtruth, so we ought to get rid of the
        # leading channel axis
        # fist, compute the positive loss and gradients
        pos_gradients, _, _, _ = malis(
            np.require(affinities, requirements='C'),
            np.require(groundtruth[0], requirements='C'), True)

        # next, compute the negative loss and gradients
        neg_gradients, _, _, _ = malis(
            np.require(affinities, requirements='C'),
            np.require(groundtruth[0], requirements='C'), False)
        return pos_gradients, neg_gradients

    def forward(self, affinities, groundtruth):
        """
        Apply malis forward pass to get the loss gradient. The final loss function is then
        the sum of this function's output.

        Parameters
        ----------
        affinities : torch.Tensor
            The affinity tensor (the network output). Must be a 4D NCHW tensor where C = 2
            or a 5D NCDHW tensor where C = 3 or C = 2.
        groundtruth : torch.Tensor
            The ground truth tensor. Must be a 4D NCHW tensor or a 5D NCDHW tensor,
            where C = 1.

        Returns
        -------
        loss: malis loss gradients
        """

        # Convert input to numpy
        affinities = affinities.numpy()
        groundtruth = groundtruth.numpy()
        # Validate
        if self.dim is None:
            assert affinities.ndim in [4, 5], "Affinity tensor must have 4 or 5 dimensions."
            # Parse dim
            self.parse_dim(affinities.ndim)
        else:
            assert affinities.ndim == self.tensor_ndim, \
                "Affinity tensor must have {} dimensions, got {} instead." \
                    .format(self.tensor_ndim, affinities.ndim)
        assert groundtruth.ndim == self.tensor_ndim, \
            "Groundtruth tensor must have {} dimensions, got {} instead." \
                .format(self.tensor_ndim, groundtruth.ndim)
        assert groundtruth.shape[1] == 1, "Groundtruth must have a single channel."
        assert affinities.shape[-self.dim:] == groundtruth.shape[-self.dim:], \
            "Spatial shapes of affinities and groundtruth do not match."
        assert affinities.shape[1] == self.malis_dim, \
            "Affinities must have 2 or 3 channels (for 2D and 3D affinities, respectively)"
        # Store shapes for backward
        self._intermediates.update({'affinities_shape': affinities.shape,
                                    'groundtruth_shape': groundtruth.shape})
        # We might want to use 2D malis for 3D volumes. This is especially true for non-realigned
        # data where we want to use 3D context but evaluate MALIS only in 2D. In this case, we must
        # consolidate the z and batch axes
        z_as_batches = self.malis_dim == 2 and self.dim == 3

        affinities_shape = affinities.shape
        groundtruth_shape = groundtruth.shape
        if z_as_batches:
            affinities = affinities.swapaxes(1, 2).reshape((-1,
                                                            affinities_shape[1],
                                                            affinities_shape[3],
                                                            affinities_shape[4]))
            groundtruth = groundtruth.swapaxes(1, 2).reshape((-1,
                                                             groundtruth_shape[1],
                                                             groundtruth_shape[3],
                                                             groundtruth_shape[4]))

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

        # Now we might need to bring bring back the z axis
        if z_as_batches:
            pos_gradients = pos_gradients\
                .reshape((affinities_shape[0],
                          affinities_shape[2],
                          affinities_shape[1],
                          affinities_shape[3],
                          affinities_shape[4]))\
                .swapaxes(1, 2)
            neg_gradients = neg_gradients \
                .reshape((affinities_shape[0],
                          affinities_shape[2],
                          affinities_shape[1],
                          affinities_shape[3],
                          affinities_shape[4])) \
                .swapaxes(1, 2)

        # save the combined gradient for the backward pass
        # the trailing .mul(1) makes a copy of the numpy tensor, without which pytorch segfaults
        # yes, i've aged figuring this out
        combined_gradient = torch.from_numpy((neg_gradients + pos_gradients) / 2.)
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


class ConstrainedMalisLoss(_MalisBase):
    """
    Constrained Malis Loss
    """

    def _wrapper(self, affinities, groundtruth):
        # The groundtruth has a leading channel axis which we need to get rid of
        gradients = constrained_malis(np.require(affinities, requirements='C'),
                                      np.require(groundtruth[0], requirements='C'))
        return gradients

    def forward(self, affinities, groundtruth):
        """
        Apply constrained malis forward pass to get the loss gradients.

        Parameters
        ----------
        affinities : torch.Tensor
            The affinity tensor (the network output). Must be a 4D NCHW tensor, where C in {2, 3}.
        groundtruth : torch.Tensor
            The ground truth tensor. Can be a 3D NHW tensor or a 4D NCHW tensor, where C = 1.

        Returns
        -------
        loss: malis loss
        """
        # Convert to numpy
        affinities = affinities.numpy()
        groundtruth = groundtruth.numpy()

        # Validate
        if self.dim is None:
            assert affinities.ndim in [4, 5], "Affinity tensor must have 4 or 5 dimensions."
            # Parse dim
            self.parse_dim(affinities.ndim)
        else:
            assert affinities.ndim == self.tensor_ndim, \
                "Affinity tensor must have {} dimensions, got {} instead." \
                    .format(self.tensor_ndim, affinities.ndim)
        assert groundtruth.ndim == self.tensor_ndim, \
            "Groundtruth tensor must have {} dimensions, got {} instead." \
                .format(self.tensor_ndim, groundtruth.ndim)
        assert groundtruth.shape[1] == 1, "Groundtruth must have a single channel."
        assert affinities.shape[-self.dim:] == groundtruth.shape[-self.dim:], \
            "Spatial shapes of affinities and groundtruth do not match."
        assert affinities.shape[1] == self.malis_dim, \
            "Affinities must have 2 or 3 channels (for 2D and 3D affinities, respectively)"
        # Store shapes for backward
        self._intermediates.update({'affinities_shape': affinities.shape,
                                    'groundtruth_shape': groundtruth.shape})

        # We might want to use 2D malis for 3D volumes. This is especially true for non-realigned
        # data where we want to use 3D context but evaluate MALIS only in 2D. In this case, we must
        # consolidate the z and batch axes
        z_as_batches = self.malis_dim == 2 and self.dim == 3

        affinities_shape = affinities.shape
        groundtruth_shape = groundtruth.shape
        if z_as_batches:
            affinities = affinities.swapaxes(1, 2).reshape((-1,
                                                            affinities_shape[1],
                                                            affinities_shape[3],
                                                            affinities_shape[4]))
            groundtruth = groundtruth.swapaxes(1, 2).reshape((-1,
                                                              groundtruth_shape[1],
                                                              groundtruth_shape[3],
                                                              groundtruth_shape[4]))
        # Compute gradients
        all_affinities = list(affinities)
        all_groundtruth = list(groundtruth)
        # Distribute over threads (GIL is lifted)
        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1)) as executor:
            all_gradients = list(executor.map(self._wrapper, all_affinities, all_groundtruth))
        # Build array
        gradients = np.array(all_gradients)

        # Now we might need to bring bring back the z axis
        if z_as_batches:
            gradients = gradients \
                .reshape((affinities_shape[0],
                          affinities_shape[2],
                          affinities_shape[1],
                          affinities_shape[3],
                          affinities_shape[4])) \
                .swapaxes(1, 2)

        gradients = torch.from_numpy(gradients)
        self.save_for_backward(gradients)
        return gradients

    def backward(self, grad_output):
        """
        Apply malis backward pass to get the gradients.
        """
        gradients, = self.saved_tensors
        target_gradient = gradients.new(*self._intermediates.get('groundtruth_shape')).zero_()
        return gradients, target_gradient


class CustomNHConstrainedMalisLoss(_MalisBase):
    """
    Constrained Malis Loss with custom neighborhood
    """

    def __init__(self, ranges, axes, dim=None, malis_dim='auto'):
        assert len(ranges) == len(axes)
        super(CustomNHConstrainedMalisLoss, self).__init__(dim=dim, malis_dim=malis_dim)
        self.ranges = ranges
        self.axes = axes

    def _wrapper(self, affinities, groundtruth):
        assert affinities.shape[0] == len(self.ranges)
        # The groundtruth has a leading channel axis which we need to get rid of
        gradients = constrained_malis_custom_nh(np.require(affinities, requirements='C'),
                                                np.require(groundtruth[0], requirements='C'),
                                                self.ranges, self.axes
                                               )
        return gradients

    def forward(self, affinities, groundtruth):
        """
        Apply constrained malis forward pass to get the loss gradients.

        Parameters
        ----------
        affinities : torch.Tensor
            The affinity tensor (the network output). Must be a 4D NCHW tensor, where C in {2, 3}.
        groundtruth : torch.Tensor
            The ground truth tensor. Can be a 3D NHW tensor or a 4D NCHW tensor, where C = 1.

        Returns
        -------
        loss: malis loss
        """
        # Convert to numpy
        affinities = affinities.numpy()
        groundtruth = groundtruth.numpy()

        # Validate


        if self.dim is None:
            assert affinities.ndim in [4, 5], "Affinity tensor must have 4 or 5 dimensions."
            # Parse dim
            self.parse_dim(affinities.ndim)
        else:
            assert affinities.ndim == self.tensor_ndim, \
                "Affinity tensor must have {} dimensions, got {} instead." \
                    .format(self.tensor_ndim, affinities.ndim)
        assert groundtruth.ndim == self.tensor_ndim, \
            "Groundtruth tensor must have {} dimensions, got {} instead." \
                .format(self.tensor_ndim, groundtruth.ndim)
        assert groundtruth.shape[1] == 1, "Groundtruth must have a single channel."
        assert affinities.shape[-self.dim:] == groundtruth.shape[-self.dim:], \
            "Spatial shapes of affinities and groundtruth do not match."
        assert affinities.shape[1] == self.malis_dim, \
            "Affinities must have 2 or 3 channels (for 2D and 3D affinities, respectively)"
        # Store shapes for backward
        self._intermediates.update({'affinities_shape': affinities.shape,
                                    'groundtruth_shape': groundtruth.shape})

        # We might want to use 2D malis for 3D volumes. This is especially true for non-realigned
        # data where we want to use 3D context but evaluate MALIS only in 2D. In this case, we must
        # consolidate the z and batch axes
        z_as_batches = self.malis_dim == 2 and self.dim == 3

        affinities_shape = affinities.shape
        groundtruth_shape = groundtruth.shape
        if z_as_batches:
            affinities = affinities.swapaxes(1, 2).reshape((-1,
                                                            affinities_shape[1],
                                                            affinities_shape[3],
                                                            affinities_shape[4]))
            groundtruth = groundtruth.swapaxes(1, 2).reshape((-1,
                                                              groundtruth_shape[1],
                                                              groundtruth_shape[3],
                                                              groundtruth_shape[4]))
        # Compute gradients
        all_affinities = list(affinities)
        all_groundtruth = list(groundtruth)
        # Distribute over threads (GIL is lifted)
        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1)) as executor:
            all_gradients = list(executor.map(self._wrapper, all_affinities, all_groundtruth))
        # Build array
        gradients = np.array(all_gradients)

        # Now we might need to bring bring back the z axis
        if z_as_batches:
            gradients = gradients \
                .reshape((affinities_shape[0],
                          affinities_shape[2],
                          affinities_shape[1],
                          affinities_shape[3],
                          affinities_shape[4])) \
                .swapaxes(1, 2)

        gradients = torch.from_numpy(gradients)
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
    def __init__(self, constrained=True, malis_dim='auto', ranges=None, axes=None, combined=False):
        """
        Parameters
        ----------
        constrained : bool
            Whether to use constrained MALIS.
        malis_dim : {2, 3, 'auto'}
            Dimensionality of the MALIS computation. This enables that that MALIS loss can
            be computed plane-wise for 3D volumes.
        """
        super(Malis, self).__init__()
        # Validate
        assert malis_dim in ['auto', 2, 3], "`malis_dim` must be one of {3, 2, 'auto'}."
        # check if we have axes and ranges for custom nh
        if ranges is not None:
            assert axes is not None
            assert len(axes) == len(ranges)
            self._ranges = ranges
            self._axes = axes
            self._custom_nh = True
            self._combined = combined
            # we override constrained for custom nhs
            self._constrained = False
        else:
            self._custom_nh = False
            self._combined = False
            self._ranges = None
            self._axes = None
            self._constrained = constrained

        self._malis_dim = malis_dim
        self.device_transfer = DeviceTransfer('cpu')

    @property
    def constrained(self):
        return self._constrained

    @property
    def custom_nh(self):
        return self._custom_nh

    @property
    def combined(self):
        return self._combined

    @property
    def ranges(self):
        return self._ranges

    @property
    def axes(self):
        return self._axes

    @property
    def malis_dim(self):
        return self._malis_dim

    # noinspection PyCallingNonCallable
    def forward(self, input, target):
        input, target = self.device_transfer(input, target)
        if self.constrained:
            loss_gradients = ConstrainedMalisLoss(malis_dim=self.malis_dim)(input, target)
        elif self.custom_nh:
            # TODO if combined, we compute normal constrained malis for the 1-range affinities
            # and constrained malis with custom nh for the long range affinities
            if self._combined:
                raise NotImplementedError("Combined malis loss is not implemented yet")
            else:
                loss_gradients = CustomNHConstrainedMalisLoss(malis_dim=self.malis_dim,
                                                              ranges=self.ranges,
                                                              axes=self.axes
                                                             )(input, target)
        else:
            loss_gradients = MalisLoss(malis_dim=self.malis_dim)(input, target)
        pseudo_loss = loss_gradients.sum()
        return pseudo_loss
