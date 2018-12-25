import numpy as np
import torch

from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import label

from inferno.io.transform import Transform

import logging
logger = logging.getLogger(__name__)


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


# TODO rename to Segmentation2Edges ?!
# TODO implement retain segmentation
# TODO test for torch and np
class Segmentation2Membranes(Transform, DtypeMapping):
    """Convert dense segmentation to boundary-maps (or membranes)."""
    def __init__(self, dtype='float32', **super_kwargs):
        super(Segmentation2Membranes, self).__init__(**super_kwargs)
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dtype = self.DTYPE_MAPPING.get(dtype)

    def image_function(self, image):
        if isinstance(image, np.ndarray):
            return self._apply_numpy_tensor(image)
        elif torch.is_tensor(image):
            return self._apply_torch_tensor(image)
        else:
            raise NotImplementedError("Only support np.ndarray and torch.tensor, got %s" % type(image))

    def _apply_numpy_tensor(self, image):
        gx = convolve(np.float32(image), np.array([-1., 0., 1.]).reshape(1, 3))
        gy = convolve(np.float32(image), np.array([-1., 0., 1.]).reshape(3, 1))
        return getattr(np, self.dtype)((gx ** 2 + gy ** 2) > 0)

    # TODO implement and test
    # def _apply_torch_tensor(self, image):
    #     conv = torch.nn.functional.conv2d
    #     kernel = image.new(1, 3, 3).zero_()
    #     return


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


class ConnectedComponents2D(Transform):
    """
    Apply connected components on segmentation in 2D.
    """
    def __init__(self, **super_kwargs):
        """
        Parameters
        ----------
        super_kwargs : dict
            Keyword arguments to the super class.
        """
        super(ConnectedComponents2D, self).__init__(**super_kwargs)

    def image_function(self, image):
        return label(image)


class ConnectedComponents3D(Transform):
    """
    Apply connected components on segmentation in 3D.
    """
    def __init__(self, **super_kwargs):
        """
        Parameters
        ----------
        super_kwargs : dict
            Keyword arguments to the super class.
        """
        super(ConnectedComponents3D, self).__init__(**super_kwargs)

    def volume_function(self, volume):
        return label(volume)
