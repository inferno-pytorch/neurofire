import numpy as np
import torch
from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
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
        else: # for ISBI the labels are inverted
            return 1-np.exp(-self.gain * distance_transform_edt(image))


class Segmentation2Affinities(Transform):
    DTYPE_MAPPING = {'float32': 'float32',
                     'float': 'float32',
                     'double': 'float64',
                     'float64': 'float64',
                     'half': 'float16',
                     'float16': 'float16'}

    def __init__(self, dtype='float32', **super_kwargs):
        super(Segmentation2Affinities, self).__init__(**super_kwargs)
        # Privates
        self._shift_kernels = None
        # Register dtype
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dtype = self.DTYPE_MAPPING.get(dtype)
        # Build kernels
        self.build_shift_kernels()

    def build_shift_kernels(self):
        # The kernels have a shape similar to conv kernels in torch. We have 3 output channels,
        # corresponding to (depth, height, width)
        shift_combined = np.zeros(shape=(3, 1, 3, 3, 3), dtype=self.dtype)
        # Shift depth
        shift_combined[0, 0, 0, 1, 1] = 1.
        # Shift height
        shift_combined[1, 0, 1, 0, 1] = 1.
        # Shift width
        shift_combined[2, 0, 1, 1, 0] = 1.
        # Set
        self._shift_kernels = shift_combined

    def convolve_with_shift_kernel(self, tensor):
        # Make sure the tensor is contains 3D volumes (i.e. is 4D) with the first axis
        # being channel
        assert tensor.ndim == 4, "Tensor must be 4D."
        assert tensor.shape[0] == 1, "Tensor must have only one channel."
        # Cast tensor to the right datatype
        if tensor.dtype != self.dtype:
            tensor = tensor.astype(self.dtype)
        # Build torch variables of the right shape
        torch_tensor = torch.autograd.Variable(torch.from_numpy(tensor[None, ...]))
        torch_kernel = torch.autograd.Variable(torch.from_numpy(self._shift_kernels))
        # Apply convolution (with zero padding)
        torch_convolved = torch.nn.functional.conv3d(input=torch_tensor,
                                                     weight=torch_kernel,
                                                     padding=1)
        # Extract numpy array and get rid of the singleton batch dimension
        convolved = torch_convolved.data.numpy()[0, ...]
        # The shape shouldn't have changed
        assert convolved.shape[-3:] == tensor.shape[-3:]
        return convolved

    def tensor_function(self, tensor):
        if tensor.ndim != 4:
            raise NotImplementedError("Affinity map generation is only supported in 3D.")
        # Convolve tensor with a shift kernel
        convolved_tensor = self.convolve_with_shift_kernel(tensor)
        # Threshold convolved tensor
        binarized_affinities = np.where(convolved_tensor == 0., 1., 0.)
        # Cast to be sure
        if not binarized_affinities.dtype == self.dtype:
            binarized_affinities = binarized_affinities.astype(self.dtype)
        return binarized_affinities
