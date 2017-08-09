import torch
import numpy as np
import inferno.utils.python_utils as pyu
from inferno.extensions.metrics.arand import ArandError
from scipy.ndimage.measurements import label


class ArandErrorWithConnectedComponentsOnAffinities(ArandError):
    NP_DTYPE = 'float32'

    def __init__(self, thresholds=0.5, invert_probabilities=False,
                 normalize_probabilities=False):
        super(ArandErrorWithConnectedComponentsOnAffinities, self).__init__()
        self.thresholds = pyu.to_iterable(thresholds)
        self.invert_probabilities = invert_probabilities
        self.normalize_probabilities = normalize_probabilities

    def affinity_to_segmentation(self, affinity_batch):
        assert affinity_batch.dim() in [4, 5], \
            "`affinity_batch` must be a 4D batch of 2D images or a 5D batch of 3D volumes."
        affinity_batch = affinity_batch.cpu().numpy()
        # probability_batch.shape = (N, z, y, x) or (N, y, x)
        probability_batch = affinity_batch.mean(axis=1)
        if self.invert_probabilities:
            probability_batch = 1. - probability_batch
        if self.normalize_probabilities:
            probability_batch = probability_batch / probability_batch.max()
        # Threshold
        thresholded_batches = [(probability_batch > threshold).astype(self.NP_DTYPE)
                               for threshold in self.thresholds]
        # Run CC once for every threshold
        connected_components_batches = []
        # Run connected components on the thresholded batches
        for thresholded_batch in thresholded_batches:
            # The following also has the shape (N, y, x) (wlog 3D)
            connected_components = np.array([label(1. - volume_or_slice)[0]
                                             for volume_or_slice in thresholded_batch])
            # We reshape it to (N, 1, y, x) and convert to a torch tensor
            connected_component_tensor = torch.from_numpy(connected_components[:, None, ...])
            connected_components_batches.append(connected_component_tensor)
        return connected_components_batches

    def forward(self, prediction, target):
        # Threshold and convert to connected components
        connected_component_batches = self.affinity_to_segmentation(prediction)
        # Compute the arand once for ever threshold. Note that the threshold is global with
        # respect to the batches, which is intended.
        arand_errors = [super(ArandErrorWithConnectedComponentsOnAffinities, self).forward(batch,
                                                                                           target)
                        for batch in connected_component_batches]
        # Select the best arand error
        return min(arand_errors)
