import vigra
import numpy as np
import torch

import inferno.utils.python_utils as pyu
from inferno.extensions.metrics.arand import ArandError

try:
    from affogato.segmentation import connected_components
    HAVE_AFFOGATO = True
except ImportError as e:
    HAVE_AFFOGATO = False


class ArandFromSegmentationBase(ArandError):
    """
    Base class for segmentation based arand error.
    Subclasses must implement `input_to_segmentation`.

    Arguments:
        parameters [list]: list of different parameters for segmentation algorithm
            that should be evaluated (only single parameter supported for now)
        average_slices [bool]: evaluate the score as average over 2d slices (default: True)
    """

    def __init__(self, parameters, average_slices=True):
        super(ArandFromSegmentationBase, self).__init__(average_slices=average_slices)
        self.parameters = pyu.to_iterable(parameters)

    def input_to_segmentation(self, input_batch, parameter):
        raise NotImplementedError("Implement `input_to_segmentation` in subclass")

    def forward(self, prediction, target):
        assert prediction.dim() in [4, 5], \
            "`prediction` must be a 4D batch of 2D images or a 5D batch of 3D volumes."

        input_batch = prediction.cpu().numpy()
        if(np.isnan(input_batch).any()):
            raise RuntimeError("Have nans in prediction")

        # iterate over the different parameters we have
        arand_errors = []
        for param in self.parameters:
            # compute the mean arand erros for all batches for the given threshold
            gt_seg = target[:, 0:1]
            cc_seg = self.input_to_segmentation(input_batch, param)
            arand_errors.append(super(ArandFromSegmentationBase, self).forward(cc_seg, gt_seg))
        # return the best arand errror
        return min(arand_errors)


class ArandErrorFromConnectedComponentsOnAffinities(ArandFromSegmentationBase):

    def __init__(self, thresholds=0.5, invert_affinities=False,
                 normalize_affinities=False, average_slices=True):
        assert HAVE_AFFOGATO, "Couldn't find 'affogato' module, affinity calculation is not available"
        super(ArandFromSegmentationBase, self).__init__(thresholds, average_slices)
        self.invert_affinities = invert_affinities
        self.normalize_affinities = normalize_affinities

    def input_to_segmentation(self, input_batch, thresh):
        dim = input_batch.ndim - 2
        # if specified, invert and / or normalize the affinities
        if self.invert_affinities:
            input_batch = 1. - input_batch
        if self.normalize_affinities:
            input_batch = input_batch / input_batch.max()

        if np.isnan(input_batch).any():
            raise RuntimeError("Have nan affinities!")
        # Compute the segmentation via connected components on the affinities
        ccs = np.array([connected_components(batch[:dim], thresh)[0] for batch in input_batch])
        # NOTE: we add a singleton channel axis here, which is expected by the arand metrics
        return torch.from_numpy(ccs[:, None].astype('int32'))


class ArandErrorFromConnectedComponents(ArandFromSegmentationBase):

    def __init__(self, thresholds=0.5, invert_input=False,
                 average_input=False, normalize_input=False, average_slices=True):
        super(ArandErrorFromConnectedComponents, self).__init__(thresholds, average_slices=average_slices)
        self.invert_input = invert_input
        self.normalize_input = normalize_input
        self.average_input = average_input

    # Compute the segmentation via connected components on the input
    def input_to_segmentation(self, input_batch, thresh):
        dim = input_batch.ndim - 2

        # if specified, invert and / or normalize the input
        if self.invert_input:
            input_batch = 1. - input_batch
        if self.normalize_input:
            input_batch -= input_batch.min()
            input_batch = input_batch / input_batch.max()

        if input_batch.shape[1] > 1:
            assert self.average_input, "Need to allow average input for multi-scale predictions"
            # we only average over the nearest neighbor affinities
            thresholded = (np.mean(input_batch[:, :dim], axis=1) >= thresh).astype('uint8')
        else:
            thresholded = (input_batch[:, 0] >= thresh).astype('uint8')
        ccs = np.array([vigra.analysis.labelMultiArray(threshd) for threshd in thresholded])
        # NOTE: we add a singleton channel axis here, which is expected by the arand metrics
        return torch.from_numpy(ccs[:, None].astype('int32'))
