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


# TODO we can subsume these under the same base class that implements most of
# the functionality

class ArandErrorFromConnectedComponentsOnAffinities(ArandError):
    NP_DTYPE = 'float32'

    def __init__(self, thresholds=0.5, invert_affinities=False,
                 normalize_affinities=False, average_slices=True):
        assert HAVE_AFFOGATO, "Couldn't find 'affogato' module, affinity calculation is not available"
        super(ArandErrorFromConnectedComponentsOnAffinities, self).__init__(average_slices=average_slices)
        self.thresholds = pyu.to_iterable(thresholds)
        self.invert_affinities = invert_affinities
        self.normalize_affinities = normalize_affinities

    def affinity_to_segmentation(self, affinity_batch, thresh):
        assert affinity_batch.dim() in [4, 5], \
            "`affinity_batch` must be a 4D batch of 2D images or a 5D batch of 3D volumes."
        dim = affinity_batch.dim() - 2
        affinity_batch = affinity_batch.cpu().numpy()
        # if specified, invert and / or normalize the affinities
        if self.invert_affinities:
            affinity_batch = 1. - affinity_batch
        if self.normalize_affinities:
            affinity_batch = affinity_batch / affinity_batch.max()

        # Compute the segmentation via connected components on the affinities
        if(np.isnan(affinity_batch).any()):
            raise RuntimeError("Have nans in prediction")

        # FIXME sometimes this segfaults, but I can't reproduce the segfault (with exactly the
        # same input !!!!) outside of training

        # import h5py
        # with h5py.File('val_input.h5', 'w') as f:
        #     f.create_dataset('data', data=affinity_batch)
        connected_components = np.array([connected_components(batch[:dim], thresh)[0]
                                         for batch in affinity_batch])
        return torch.from_numpy(connected_components[:, None].astype('int32'))

    def forward(self, prediction, target):
        # iterate over the different thresholds we have
        arand_errors = []
        for thresh in self.thresholds:
            # compute the mean arand erros for all batches for the given threshold
            gt_seg = target[:, 0:1]
            cc_seg = self.affinity_to_segmentation(prediction, thresh)
            arand_errors.append(super(ArandErrorFromConnectedComponentsOnAffinities,
                                      self).forward(cc_seg, gt_seg))
        # return the minimum arand errror
        return min(arand_errors)


class ArandErrorFromConnectedComponents(ArandError):
    NP_DTYPE = 'float32'

    def __init__(self, thresholds=0.5, invert_input=False,
                 average_input=False, normalize_input=False, average_slices=True):
        super(ArandErrorFromConnectedComponents, self).__init__(average_slices=average_slices)
        self.thresholds = pyu.to_iterable(thresholds)
        self.invert_input = invert_input
        self.normalize_input = normalize_input
        self.average_input = average_input

    def input_to_segmentation(self, input_batch, thresh):
        assert input_batch.dim() in [4, 5], \
            "`input_batch` must be a 4D batch of 2D images or a 5D batch of 3D volumes."
        dim = input_batch.dim() - 2
        input_batch = input_batch.cpu().numpy()
        # if specified, invert and / or normalize the input
        if self.invert_input:
            input_batch = 1. - input_batch
        if self.normalize_input:
            input_batch = input_batch / input_batch.max()

        # Compute the segmentation via connected components on the input
        if(np.isnan(input_batch).any()):
            raise RuntimeError("Have nans in prediction")

        if input_batch.shape[1] > 1:
            assert self.average_input, "Need to allow average input for multi-scale predictions"
            thresholded = (np.mean(input_batch, axis=1) >= thresh).astype('uint8')
        else:
            thresholded = (input_batch[:, 0] >= thresh).astype('uint8')
        connected_components = np.array([vigra.analysis.labelMultiArrayWithBackground(threshd)
                                         for threshd in thresholded])
        return torch.from_numpy(connected_components[:, None].astype('int32'))

    def forward(self, prediction, target):
        # iterate over the different thresholds we have
        arand_errors = []
        for thresh in self.thresholds:
            # compute the mean arand erros for all batches for the given threshold
            gt_seg = target[:, 0:1]
            cc_seg = self.input_to_segmentation(prediction, thresh)
            arand_errors.append(super(ArandErrorFromConnectedComponents,
                                      self).forward(cc_seg, gt_seg))
        # return the minimum arand errror
        return min(arand_errors)
