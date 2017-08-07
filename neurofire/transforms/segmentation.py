import numpy as np
import torch

from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
try:
    import vigra
    with_vigra = True
except ImportError:
    print("Vigra was not found, connected components will not be available")
    with_vigra = False

from inferno.io.transform import Transform


class Segmentation2Membranes(Transform):
    DTYPE_MAPPING = {'float32': 'float32',
                     'float': 'float32',
                     'double': 'float64',
                     'float64': 'float64',
                     'half': 'float16',
                     'float16': 'float16'}

    def __init__(self, dtype='float32', **super_kwargs):
        super(Segmentation2Membranes, self).__init__(**super_kwargs)
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dtype = self.DTYPE_MAPPING.get(dtype)

    def image_function(self, image):
        gx = convolve(np.float32(image), np.array([-1., 0., 1.]).reshape(1, 3))
        gy = convolve(np.float32(image), np.array([-1., 0., 1.]).reshape(3, 1))
        return getattr(np, self.dtype)((gx ** 2 + gy ** 2) > 0)


class NegativeExponentialDistanceTransform(Transform):
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
            return 1-np.exp(-self.gain * distance_transform_edt(image))


class Segmentation2Affinities(Transform):
    DTYPE_MAPPING = {'float32': 'float32',
                     'float': 'float32',
                     'double': 'float64',
                     'float64': 'float64',
                     'half': 'float16',
                     'float16': 'float16'}

    def __init__(self, dim, dtype='float32', add_singleton_channel_dimension=False, **super_kwargs):
        super(Segmentation2Affinities, self).__init__(**super_kwargs)
        # Privates
        self._shift_kernels = None
        # Validate and register args
        assert dim in [2, 3]
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dim = dim
        self.dtype = self.DTYPE_MAPPING.get(dtype)
        self.add_singleton_channel_dimension = bool(add_singleton_channel_dimension)
        # Build kernels
        self.build_shift_kernels()

    def build_shift_kernels(self):
        if self.dim == 3:
            # The kernels have a shape similar to conv kernels in torch. We have 3 output channels,
            # corresponding to (depth, height, width)
            shift_combined = np.zeros(shape=(3, 1, 3, 3, 3), dtype=self.dtype)
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
            self._shift_kernels = shift_combined
        elif self.dim == 2:
            # Again, the kernels are similar to conv kernels in torch. We now have 2 output
            # channels, corresponding to (height, width)
            shift_combined = np.zeros(shape=(2, 1, 3, 3), dtype=self.dtype)
            # Shift height
            shift_combined[0, 0, 0, 1] = 1.
            shift_combined[0, 0, 1, 1] = -1.
            # Shift width
            shift_combined[1, 0, 1, 0] = 1.
            shift_combined[1, 0, 1, 1] = -1.
            # Set
            self._shift_kernels = shift_combined
        else:
            raise NotImplementedError

    def convolve_with_shift_kernel(self, tensor):
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
        # Apply convolution (with zero padding)
        torch_convolved = conv(input=torch_tensor,
                               weight=torch_kernel,
                               padding=1)
        # Extract numpy array and get rid of the singleton batch dimension
        convolved = torch_convolved.data.numpy()[0, ...]
        return convolved

    def tensor_function(self, tensor):
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
        return binarized_affinities


class ConnectedComponents2D(Transform):
    """
    Apply connected components on segmentation in 2D.
    """
    def image_function(self, image):
        if not with_vigra:
            raise ImportError("Connected components is not supported without vigra")
        connected_components = vigra.analysis.labelImageWithBackground(image)
        return connected_components


class ConnectedComponents3D(Transform):
    """
    Apply connected components on segmentation in 3D.
    """
    def volume_function(self, volume):
        if not with_vigra:
            raise ImportError("Connected components is not supported without vigra")
        connected_components = vigra.analysis.labelVolumeWithBackground(image)
        return connected_components
