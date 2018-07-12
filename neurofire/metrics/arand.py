import numpy as np
import torch

import inferno.utils.python_utils as pyu
from inferno.extensions.metrics.arand import ArandError

try:
    import affogato
    HAVE_AFFOGATO = True
except ImportError as e:
    HAVE_AFFOGATO = True
    # print("Couldn't find 'affinities' module, fast affinity calculation is not available")


class ArandErrorFromConnectedComponentsOnAffinities(ArandError):
    NP_DTYPE = 'float32'

    def __init__(self, thresholds=0.5, invert_affinities=False,
                 normalize_affinities=False):
        assert HAVE_AFFOGATO, "Couldn't find 'affogato' module, affinity calculation is not available"
        super(ArandErrorFromConnectedComponentsOnAffinities, self).__init__()
        self.thresholds = pyu.to_iterable(thresholds)
        self.invert_affinities = invert_affinities
        self.normalize_affinities = normalize_affinities

    def affinity_to_segmentation(self, affinity_batch, thresh):
        from affogato.segmentation import connected_components
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
            arand_errors.append(super(ArandErrorFromConnectedComponentsOnAffinities, self).forward(cc_seg, gt_seg))
        # return the minimum arand errror
        return min(arand_errors)
