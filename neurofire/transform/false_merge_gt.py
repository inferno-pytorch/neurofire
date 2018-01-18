import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from inferno.io.transform import Transform
import nifty.graph.rag as nrag


# TODO alternatively try affinity idea
class ArtificialFalseMerges(Transform):
    def __init__(self,
                 target_distances,
                 n_threads=8,
                 size_percentile=90,
                 **super_kwargs):

        assert isinstance(target_distances, (list, tuple))
        self.target_distances = target_distances
        self.n_threads = n_threads
        self.size_percentile = size_percentile
        super(ArtificialFalseMerges, self).__init__(**super_kwargs)

    def produce_false_merge(self, target):
        segmentation = target.squeeze()
        assert segmentation.ndim == 3
        # build the rag and the edge builder
        rag = nrag.gridRag(segmentation, numberOfThreads=self.n_threads)
        edge_builder = nrag.ragCoordinates(rag, numberOfThreads=self.n_threads)
        # find candidate objects with sufficient size in batch
        objects, counts = np.unique(segmentation, return_counts=True)
        size_threshold = np.percentile(counts, self.size_percentile)
        candidate_objects = objects[objects > size_threshold]
        # sample 2 adjacent candidate objects
        while True:
            candidate_obj = np.random.choice(candidate_objects, size=1)[0]
            merge_objects = []
            merge_objs = np.array([adj[0] for adj in rag.nodeAdjacency(candidate_obj)],
                                  dtype='uint32')
            if merge_objects.size == 0:
                continue
            # TODO allow merging more than one object ?!
            merge_obj = np.random.choice(merge_objects, size=1)[0]
            break
        # make volume with edge coordinates, false merge and mask
        # of resulting object
        edges = np.zeros(rag.numberOfEdges, dtype='uint32')
        edges[rag.findEdge(candidate_obj, merge_obj)] = 1

        # scipy distance trafo is inverted ...
        false_merge = (1. - edge_builder.edgesToVolume(edges, numberOfThreads=self.n_threads)).astype('bool')
        segmentation[segmentation == merge_obj] = candidate_obj
        mask = segmentation == candidate_obj
        return false_merge, mask

    def batch_function(self, tensors):
        # we expect data and target input
        assert len(tensors) == 2
        inputs, target = tensors
        false_merge_mask, object_mask = self.produce_false_merge(target)
        # construct target from false merge edge mask
        false_merge_distance = distance_transform_edt(false_merge_mask)
        false_merge_targets = [(false_merge_distance < fm_dist).astype('float32') for fm_dist in self.target_distances]
        # set to zero outside of object mask
        inverted_mask = np.logical_not(object_mask)
        for fmt in false_merge_targets:
            fmt[inverted_mask] = 0
        # build new inputs (= previous inputs + object masks)
        # and targets (= stacked fm targets)
        inputs = np.array([inputs, object_mask.astype(inputs.dtype)])
        targets = np.array(false_merge_targets)
        return inputs, targets
