import numpy as np
import torch

from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import label

from inferno.io.transform import Transform
import inferno.utils.python_utils as pyu
import inferno.utils.torch_utils as tu

import logging
logger = logging.getLogger(__name__)

try:
    import vigra
    with_vigra = True
except ImportError:
    logger.warn("Vigra was not found, connected components will not be available")
    vigra = None
    with_vigra = False


class DtypeMapping(object):
    DTYPE_MAPPING = {'float32': 'float32',
                     'float': 'float32',
                     'double': 'float64',
                     'float64': 'float64',
                     'half': 'float16',
                     'float16': 'float16'}

    INVERSE_DTYPE_MAPPING = {'float32': 'float',
                             'float64': 'double',
                             'float16': 'half',
                             'int64': 'long'}


class Segmentation2Membranes(Transform, DtypeMapping):
    """Convert dense segmentation to boundary-maps (or membranes)."""
    def __init__(self, dtype='float32', **super_kwargs):
        super(Segmentation2Membranes, self).__init__(**super_kwargs)
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dtype = self.DTYPE_MAPPING.get(dtype)

    def image_function(self, image):
        gx = convolve(np.float32(image), np.array([-1., 0., 1.]).reshape(1, 3))
        gy = convolve(np.float32(image), np.array([-1., 0., 1.]).reshape(3, 1))
        return getattr(np, self.dtype)((gx ** 2 + gy ** 2) > 0)


class NegativeExponentialDistanceTransform(Transform):
    """'Smooth' e.g. membranes by applying a negative exponential on the distance transform."""
    def __init__(self, gain=1., invert=True, **super_kwargs):
        super(NegativeExponentialDistanceTransform, self).__init__(**super_kwargs)
        self.invert = invert
        self.gain = gain

    def image_function(self, image):
        if self.invert:
            image = 1. - image
            return np.exp(-self.gain * distance_transform_edt(image))
        else:
            # for ISBI the labels are inverted
            return 1 - np.exp(-self.gain * distance_transform_edt(image))


class Segmentation2Affinities(Transform, DtypeMapping):
    """Convert dense segmentation to affinity-maps of arbitrary order."""
    def __init__(self, dim, order=1, dtype='float32', add_singleton_channel_dimension=False,
                 retain_segmentation=False, diagonal_affinities=False, use_gpu=False,
                 **super_kwargs):
        super(Segmentation2Affinities, self).__init__(**super_kwargs)
        # Privates
        self._shift_kernels = None
        # Validate and register args
        assert dim in [2, 3]
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dim = dim
        self.dtype = self.DTYPE_MAPPING.get(dtype)
        self.add_singleton_channel_dimension = bool(add_singleton_channel_dimension)
        self.order = order if isinstance(order, int) else tuple(order)
        self.retain_segmentation = retain_segmentation
        self.use_gpu = use_gpu
        self.diagonal_affinities = diagonal_affinities
        # Build kernels
        self._shift_kernels = self.build_shift_kernels(dim=self.dim,
                                                       dtype=self.dtype,
                                                       diagonal_affinities=self.diagonal_affinities)
        # This will be filled in later if required
        self._cuda_shift_kernels = None

    @staticmethod
    def build_shift_kernels(dim, dtype, diagonal_affinities):
        if dim == 3:
            if diagonal_affinities:
                raise NotImplementedError("Diagonal affinities are not supported for 3d input yet.")
            # The kernels have a shape similar to conv kernels in torch. We have 3 output channels,
            # corresponding to (depth, height, width)
            shift_combined = np.zeros(shape=(3, 1, 3, 3, 3), dtype=dtype)
            # Shift depth
            shift_combined[0, 0, 0, 1, 1] = 1.
            shift_combined[0, 0, 1, 1, 1] = -1.
            # Shift height
            shift_combined[1, 0, 1, 0, 1] = 1.
            shift_combined[1, 0, 1, 1, 1] = -1.
            # Shift width
            shift_combined[2, 0, 1, 1, 0] = 1.
            shift_combined[2, 0, 1, 1, 1] = -1.
            # Set
            return shift_combined
        elif dim == 2:
            # Again, the kernels are similar to conv kernels in torch. We now have 2 output
            # channels, corresponding to (height, width)
            shift_combined = np.zeros(shape=(2, 1, 3, 3), dtype=dtype)

            # Shift height
            if diagonal_affinities:
                shift_combined[0, 0, 0, 0] = 1.
            else:
                shift_combined[0, 0, 0, 1] = 1.
            shift_combined[0, 0, 1, 1] = -1.

            # Shift width
            if diagonal_affinities:
                shift_combined[1, 0, 2, 0] = 1.
            else:
                shift_combined[1, 0, 1, 0] = 1.
            shift_combined[1, 0, 1, 1] = -1.
            # Set
            return shift_combined
        else:
            raise NotImplementedError

    def convolve_with_shift_kernel(self, tensor):
        if isinstance(tensor, np.ndarray):
            return self._convolve_with_shift_kernel_numpy(tensor)
        elif torch.is_tensor(tensor):
            return self._convolve_with_shift_kernel_torch(tensor)
        else:
            raise NotImplementedError

    def _convolve_with_shift_kernel_numpy(self, tensor):
        if self.dim == 3:
            # Make sure the tensor is contains 3D volumes (i.e. is 4D) with the first axis
            # being channel
            assert tensor.ndim == 4, "Tensor must be 4D for dim = 3."
            assert tensor.shape[0] == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv3d
        elif self.dim == 2:
            # Make sure the tensor contains 2D images (i.e. is 3D) with the first axis
            # being channel
            assert tensor.ndim == 3, "Tensor must be 3D for dim = 2."
            assert tensor.shape[0] == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv2d
        else:
            raise NotImplementedError
        # Cast tensor to the right datatype
        if tensor.dtype != self.dtype:
            tensor = tensor.astype(self.dtype)
        # Build torch variables of the right shape (i.e. with a leading singleton batch axis)
        torch_tensor = torch.autograd.Variable(torch.from_numpy(tensor[None, ...]))
        torch_kernel = torch.autograd.Variable(torch.from_numpy(self._shift_kernels))
        # Apply convolution (with zero padding). To obtain higher order features,
        # we apply a dilated convolution.
        torch_convolved = conv(input=torch_tensor,
                               weight=torch_kernel,
                               padding=self.order,
                               dilation=self.order)
        # Extract numpy array and get rid of the singleton batch dimension
        convolved = torch_convolved.data.numpy()[0, ...]
        return convolved

    def _convolve_with_shift_kernel_torch(self, tensor):
        if self.dim == 3:
            # Make sure the tensor is contains 3D volumes (i.e. is 4D) with the first axis
            # being channel
            assert tensor.dim() == 4, "Tensor must be 4D for dim = 3."
            assert tensor.size(0) == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv3d
        elif self.dim == 2:
            # Make sure the tensor contains 2D images (i.e. is 3D) with the first axis
            # being channel
            assert tensor.dim() == 3, "Tensor must be 3D for dim = 2."
            assert tensor.size(0) == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv2d
        else:
            raise NotImplementedError
        # Cast tensor to the right datatype (no-op if it's the right dtype already)
        tensor = getattr(tensor, self.INVERSE_DTYPE_MAPPING.get(self.dtype))()
        # Move tensor to GPU if required
        if self.use_gpu:
            tensor = tensor.cuda()
            # Assign kernel tensor
            if self._cuda_shift_kernels is None:
                self._cuda_shift_kernels = torch.from_numpy(self._shift_kernels).cuda()
            kernel_tensor = self._cuda_shift_kernels
        else:
            kernel_tensor = torch.from_numpy(self._shift_kernels)
        # Build torch variables of the right shape (i.e. with a leading singleton batch axis)
        torch_tensor = torch.autograd.Variable(tensor[None, ...])
        torch_kernel = torch.autograd.Variable(kernel_tensor)
        # Apply convolution (with zero padding). To obtain higher order features,
        # we apply a dilated convolution.
        torch_convolved = conv(input=torch_tensor,
                               weight=torch_kernel,
                               padding=self.order,
                               dilation=self.order)
        # Get rid of the singleton batch dimension (keep cuda tensor as is)
        convolved = torch_convolved.data[0, ...]
        return convolved

    def tensor_function(self, tensor):
        if isinstance(tensor, np.ndarray):
            return self._tensor_function_numpy(tensor)
        elif torch.is_tensor(tensor):
            return self._tensor_function_torch(tensor)
        else:
            raise NotImplementedError

    def _tensor_function_numpy(self, tensor):
        # Add singleton channel dimension if requested
        if self.add_singleton_channel_dimension:
            tensor = tensor[None, ...]
        if tensor.ndim not in [3, 4]:
            raise NotImplementedError("Affinity map generation is only supported in 2D and 3D. "
                                      "Did you mean to set add_singleton_channel_dimension to "
                                      "True?")
        if (tensor.ndim == 3 and self.dim == 2) or (tensor.ndim == 4 and self.dim == 3):
            # Convolve tensor with a shift kernel
            convolved_tensor = self.convolve_with_shift_kernel(tensor)
        elif tensor.ndim == 4 and self.dim == 2:
            # Tensor contains 3D volumes, but the affinity maps are computed in 2D. So we loop over
            # all z-planes and concatenate the results together
            convolved_tensor = np.stack([self.convolve_with_shift_kernel(tensor[:, z_num, ...])
                                         for z_num in range(tensor.shape[1])], axis=1)
        else:
            raise NotImplementedError
        # Threshold convolved tensor
        binarized_affinities = np.where(convolved_tensor == 0., 1., 0.)
        # Cast to be sure
        if not binarized_affinities.dtype == self.dtype:
            binarized_affinities = binarized_affinities.astype(self.dtype)

        # FIXME remove legacy
        # if we have a grown boundary, change affinities where we have the boundary label to 1
        # if self.grow_boundary:
        #     boundary_mask = (tensor == self.boundary_value).squeeze()
        #     binarized_affinities[:, boundary_mask] = 1

        # We might want to carry the segmentation along (e.g. when combining MALIS with
        # euclidean loss higher-order affinities). If this is the case, we insert the segmentation
        # as the *first* channel.
        if self.retain_segmentation:
            if tensor.dtype != self.dtype:
                tensor = tensor.astype(self.dtype)
            output = np.concatenate((tensor, binarized_affinities), axis=0)
        else:
            output = binarized_affinities
        return output

    def _tensor_function_torch(self, tensor):
        # Add singleton channel dimension if requested
        if self.add_singleton_channel_dimension:
            tensor = tensor[None, ...]
        if tensor.dim() not in [3, 4]:
            raise NotImplementedError("Affinity map generation is only supported in 2D and 3D. "
                                      "Did you mean to set add_singleton_channel_dimension to "
                                      "True?")
        if (tensor.dim() == 3 and self.dim == 2) or (tensor.dim() == 4 and self.dim == 3):
            # Convolve tensor with a shift kernel
            convolved_tensor = self.convolve_with_shift_kernel(tensor)
        elif tensor.dim() == 4 and self.dim == 2:
            # Tensor contains 3D volumes, but the affinity maps are computed in 2D. So we loop over
            # all z-planes and concatenate the results together
            convolved_tensor = torch.stack([self.convolve_with_shift_kernel(tensor[:, z_num, ...])
                                            for z_num in range(tensor.size(1))], dim=1)
        else:
            raise NotImplementedError
        # Threshold convolved tensor
        binarized_affinities = tu.where(convolved_tensor == 0.,
                                        convolved_tensor.new(*convolved_tensor.size()).fill_(1.),
                                        convolved_tensor.new(*convolved_tensor.size()).fill_(0.))

        # We might want to carry the segmentation along (e.g. when combining MALIS with
        # euclidean loss higher-order affinities). If this is the case, we insert the segmentation
        # as the *first* channel.
        if self.retain_segmentation:
            tensor = getattr(tensor, self.INVERSE_DTYPE_MAPPING.get(self.dtype))()
            output = torch.cat((tensor, binarized_affinities), 0)
        else:
            output = binarized_affinities
        return output


class Segmentation2MultiOrderAffinities(Transform):
    """
    Generate affinity maps of multiple order affinities given a segmentation. The resulting maps
    are concatenated along the leading (= channel) axis.
    """
    def __init__(self, dim, orders, dtype='float32', add_singleton_channel_dimension=False,
                 retain_segmentation=False, diagonal_affinities=False, **super_kwargs):
        super(Segmentation2MultiOrderAffinities, self).__init__(**super_kwargs)
        assert pyu.is_listlike(orders), "`orders` must be a list or a tuple."
        assert len(orders) > 0, "`orders` must not be empty."
        # Build Segmentation2Affinity objects
        if diagonal_affinities:
            self._segmentation2affinities_objects = [
                Segmentation2Affinities(dim, order=order, dtype=dtype,
                                        diagonal_affinities=use_diag,
                                        add_singleton_channel_dimension=add_singleton_channel_dimension)
                for order in orders for use_diag in (False, True)]

        else:
            self._segmentation2affinities_objects = [
                Segmentation2Affinities(dim, order=order, dtype=dtype,
                                        add_singleton_channel_dimension=add_singleton_channel_dimension)
                for order in orders]
        # Have the first segmentation2affinities object retain the segmentation if required
        if retain_segmentation:
            self._segmentation2affinities_objects[0].retain_segmentation = True

    @property
    def retain_segmentation(self):
        return self._segmentation2affinities_objects[0].retain_segmentation

    @property
    def dim(self):
        return self._segmentation2affinities_objects[0].dim

    @property
    def add_singleton_channel_dimension(self):
        return self._segmentation2affinities_objects[0].add_singleton_channel_dimension

    @property
    def dtype(self):
        return self._segmentation2affinities_objects[0].dtype

    @property
    def orders(self):
        return [seg2aff.order for seg2aff in self._segmentation2affinities_objects]

    def tensor_function(self, tensor):
        higher_order_affinity_tensor = \
            np.concatenate([seg2aff(tensor) for seg2aff in self._segmentation2affinities_objects],
                           axis=0)
        return higher_order_affinity_tensor


class Segmentation2AffinitiesFromOffsets(Transform, DtypeMapping):
    def __init__(self, dim, offsets, dtype='float32',
                 add_singleton_channel_dimension=False,
                 use_gpu=False,
                 retain_segmentation=False, **super_kwargs):
        super(Segmentation2AffinitiesFromOffsets, self).__init__(**super_kwargs)
        assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        assert len(offsets) > 0, "`offsets` must not be empty."

        # TODO implement for 3d as well
        assert dim == 2, "Not implemented in 3d yet"
        self.dim = dim
        self.dtype = self.DTYPE_MAPPING.get(dtype)
        self.add_singleton_channel_dimension = bool(add_singleton_channel_dimension)
        self.offsets = offsets if isinstance(offsets, int) else tuple(offsets)
        self.retain_segmentation = retain_segmentation
        self.use_gpu = use_gpu

    def convolve_with_shift_kernel(self, tensor, offset):
        if isinstance(tensor, np.ndarray):
            return self._convolve_with_shift_kernel_numpy(tensor, offset)
        elif torch.is_tensor(tensor):
            # TODO implement for torch input as well
            assert False, "Not implemented for torch tensors yet"
            return self._convolve_with_shift_kernel_torch(tensor, offset)
        else:
            raise NotImplementedError

    def build_shift_kernels(self, offset):
        # TODO
        if self.dim == 3:
            raise NotImplementedError("Not implemented yet!")
        elif self.dim == 2:
            # Again, the kernels are similar to conv kernels in torch.
            # We now have 2 output
            # channels, corresponding to (height, width)
            shift_combined = np.zeros(shape=(1, 1, 3, 3), dtype=self.dtype)

            assert len(offset) == 2
            assert np.sum(np.abs(offset)) > 0

            shift_combined[0, 0, 1, 1] = -1.
            s_x = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            shift_combined[0, 0, s_x, s_y] = 1.
            return shift_combined
        else:
            raise NotImplementedError

    def _convolve_with_shift_kernel_numpy(self, tensor, offset):
        if self.dim == 3:
            # Make sure the tensor is contains 3D volumes (i.e. is 4D) with the first axis
            # being channel
            assert tensor.ndim == 4, "Tensor must be 4D for dim = 3."
            assert tensor.shape[0] == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv3d
        elif self.dim == 2:
            # Make sure the tensor contains 2D images (i.e. is 3D) with the first axis
            # being channel
            assert tensor.ndim == 3, "Tensor must be 3D for dim = 2."
            assert tensor.shape[0] == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv2d
        else:
            raise NotImplementedError
        # Cast tensor to the right datatype
        if tensor.dtype != self.dtype:
            tensor = tensor.astype(self.dtype)
        # Build torch variables of the right shape (i.e. with a leading singleton batch axis)
        torch_tensor = torch.autograd.Variable(torch.from_numpy(tensor[None, ...]))
        shift_kernel = self.build_shift_kernels(offset)
        torch_kernel = torch.autograd.Variable(torch.from_numpy(shift_kernel))
        # Apply convolution (with zero padding). To obtain higher order features,
        # we apply a dilated convolution.
        abs_offset = tuple(max(1, abs(off)) for off in offset)
        # abs_offset = int(max(1, np.max(np.abs(offset))))
        torch_convolved = conv(input=torch_tensor,
                               weight=torch_kernel,
                               padding=abs_offset,
                               dilation=abs_offset)
        # Extract numpy array and get rid of the singleton batch dimension
        convolved = torch_convolved.data.numpy()[0, ...]
        return convolved

    def tensor_function(self, tensor):
        if isinstance(tensor, np.ndarray):
            return self._tensor_function_numpy(tensor)
        elif torch.is_tensor(tensor):
            # TODO
            assert False, "Not implemented yet"
            return self._tensor_function_torch(tensor)
        else:
            raise NotImplementedError

    def _tensor_function_numpy(self, tensor):
        # Add singleton channel dimension if requested
        if self.add_singleton_channel_dimension:
            tensor = tensor[None, ...]
        if tensor.ndim not in [3, 4]:
            raise NotImplementedError("Affinity map generation is only supported in 2D and 3D. "
                                      "Did you mean to set add_singleton_channel_dimension to "
                                      "True?")
        if (tensor.ndim == 3 and self.dim == 2) or (tensor.ndim == 4 and self.dim == 3):
            # Convolve tensor with a shift kernel
            convolved_tensor = np.concatenate(
                [self.convolve_with_shift_kernel(tensor, offset)
                 for offset in self.offsets], axis=0)
        elif tensor.ndim == 4 and self.dim == 2:
            # Tensor contains 3D volumes, but the affinity maps are computed in 2D. So we loop over
            # all z-planes and concatenate the results together
            # TODO
            assert False, "Not implemented yet"
            convolved_tensor = np.stack([self.convolve_with_shift_kernel(tensor[:, z_num, ...])
                                         for z_num in range(tensor.shape[1])], axis=1)
        else:
            raise NotImplementedError
        # Threshold convolved tensor
        binarized_affinities = np.where(convolved_tensor == 0., 1., 0.)
        # Cast to be sure
        if not binarized_affinities.dtype == self.dtype:
            binarized_affinities = binarized_affinities.astype(self.dtype)

        if self.retain_segmentation:
            if tensor.dtype != self.dtype:
                tensor = tensor.astype(self.dtype)
            output = np.concatenate((tensor, binarized_affinities), axis=0)
        else:
            output = binarized_affinities
        return output


class ConnectedComponents2D(Transform):
    """
    Apply connected components on segmentation in 2D.
    """
    def __init__(self, label_segmentation=True, **super_kwargs):
        """
        Parameters
        ----------
        label_segmentation : bool
            Whether the input is a segmentation. If True (default), this computes a
            connected components on both segmentation and binary images (instead of just binary
            images, when this is set to False). However, this would require vigra as a dependency.
        super_kwargs : dict
            Keyword arguments to the super class.
        """
        super(ConnectedComponents2D, self).__init__(**super_kwargs)
        self.label_segmentation = label_segmentation

    def image_function(self, image):
        if not with_vigra and self.label_segmentation:
            raise NotImplementedError("Connected components is not supported without vigra "
                                      "if label_segmentation is set to True.")
        if self.label_segmentation:
            connected_components = vigra.analysis.labelImageWithBackground(image)
        else:
            connected_components, _ = label(image)
        return connected_components


class ConnectedComponents3D(Transform):
    """
    Apply connected components on segmentation in 3D.
    """
    def __init__(self, label_segmentation=True, **super_kwargs):
        """
        Parameters
        ----------
        label_segmentation : bool
            Whether the input is a segmentation. If True (default), this computes a
            connected components on both segmentation and binary images (instead of just binary
            images, when this is set to False). However, this would require vigra as a dependency.
        super_kwargs : dict
            Keyword arguments to the super class.
        """
        super(ConnectedComponents3D, self).__init__(**super_kwargs)
        self.label_segmentation = label_segmentation

    def volume_function(self, volume):
        if not with_vigra and self.label_segmentation:
            raise NotImplementedError("Connected components is not supported without vigra "
                                      "if label_segmentation is set to True.")
        if self.label_segmentation:
            connected_components = vigra.analysis.labelVolumeWithBackground(volume.astype('uint32'))
        else:
            connected_components, _ = label(volume)
        return connected_components


class SegmentationToBinaryLabels(Transform):
    """
    Transform a segmentation to a binary 0-1 labeling.
    """

    def __init__(self, background_label=0, ignore_label=None, **super_kwargs):
        """
        Parameters
        ----------
        background_label: int
        Label of the background class (default = 0)
        ignore_label: int
        Ignore label that will be kept at its value (default = None).
        """
        super(SegmentationToBinaryLabels, self).__init__(**super_kwargs)
        self.background_label = background_label
        self.ignore_label = ignore_label

    def volume_function(self, volume):
        foreground_mask = volume != self.background_label

        if self.ignore_label is not None:
            ignore_mask = volume == self.ignore_label
            foreground_mask[ignore_mask] = False

        volume[foreground_mask] = 1
        return volume
