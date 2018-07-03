import numpy as np
import torch

import inferno.utils.python_utils as pyu
from inferno.extensions.metrics.arand import ArandError

try:
    import affinities
    HAVE_AFFINITIES = True
except ImportError as e:
    HAVE_AFFINITIES = False
    # print("Couldn't find 'affinities' module, fast affinity calculation is not available")


class ArandErrorFromConnectedComponentsOnAffinities(ArandError):
    NP_DTYPE = 'float32'

    def __init__(self, thresholds=0.5, invert_affinities=False,
                 normalize_affinities=False):
        super(ArandErrorFromConnectedComponentsOnAffinities, self).__init__()
        self.thresholds = pyu.to_iterable(thresholds)
        self.invert_affinities = invert_affinities
        self.normalize_affinities = normalize_affinities

    # TODO there is another batch dimension in there
    def affinity_to_segmentation(self, affinity_batch):
        assert affinity_batch.dim() in [4, 5], \
            "`affinity_batch` must be a 4D batch of 2D images or a 5D batch of 3D volumes."
        affinity_batch = affinity_batch.cpu().numpy()
        # if specified, invert and / or normalize the affinities
        if self.invert_affinities:
            affinity_batch = 1. - affinity_batch
        if self.normalize_affinities:
            affinity_batch = affinity_batch / affinity_batch.max()
        # Compute the segmentation via connected components on the affinities
        # for each threshold
        connected_components_batches = []
        for batch in affinity_batch:
            connected_components = [affinities.connected_components(batch[0], thresh)
                                    for thresh in self.thresholds]
            # The following also has the shape (N, y, x) (wlog 3D)
            connected_components = np.array([affinities.connected_components(batch[0], thresh)
                                             for thresh in self.thresholds])
            # We reshape it to (N, 1, y, x) and convert to a torch tensor
            connected_component_tensor = torch.from_numpy(connected_components[:, None, ...])
            connected_components_batches.append(connected_component_tensor)
        return connected_components_batches

    def forward(self, prediction, target):
        # Threshold and convert to connected components
        connected_component_batches = self.affinity_to_segmentation(prediction)
        # Compute the arand once for ever threshold. Note that the threshold is global with
        # respect to the batches, which is intended.
        arand_errors = [super(ArandErrorFromConnectedComponentsOnAffinities, self).forward(batch, target)
                        for batch in connected_component_batches]
        # Select the best arand error
        return min(arand_errors)
