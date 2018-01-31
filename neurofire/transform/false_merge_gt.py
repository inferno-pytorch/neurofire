import numpy as np
import vigra
from scipy.ndimage.morphology import distance_transform_edt
from inferno.io.transform import Transform
import nifty.graph.rag as nrag


# TODO alternatively try affinity idea
class ArtificialFalseMerges(Transform):
    def __init__(self,
                 target_distances,
                 n_threads=8,
                 size_percentile=90,
                 ignore_label=None,
                 crop_to_object=None,
                 **super_kwargs):

        assert isinstance(target_distances, (list, tuple))
        self.target_distances = target_distances
        self.n_threads = n_threads
        self.size_percentile = size_percentile
        self.ignore_label = ignore_label

        # we encode, by which number the output must be divisible in crop_to_output
        # if it is not None
        if crop_to_object is not None:
            assert isinstance(crop_to_object, (tuple, list, int))
            if isinstance(crop_to_object, int):
                self.crop_to_object = (crop_to_object,) * 3
            else:
                assert len(crop_to_object) == 3
                self.crop_to_object = crop_to_object
        else:
            self.crop_to_object = None

        super(ArtificialFalseMerges, self).__init__(**super_kwargs)

    def _produce_false_merge(self, target):
        segmentation = np.require(target.squeeze(), dtype='uint32')
        assert segmentation.ndim == 3
        segmentation = vigra.analysis.labelVolumeWithBackground(segmentation).astype('uint64')
        # find candidate objects with sufficient size in batch
        objects, counts = np.unique(segmentation, return_counts=True)
        size_threshold = np.percentile(counts, self.size_percentile)
        candidate_objects = objects[counts > size_threshold]
        # build the rag and the edge builder
        rag = nrag.gridRag(segmentation,
                           numberOfLabels=int(segmentation.max()+1),
                           numberOfThreads=self.n_threads)
        edge_builder = nrag.ragCoordinates(rag, numberOfThreads=self.n_threads)
        # sample 2 adjacent candidate objects
        while True:
            candidate_obj = int(np.random.choice(candidate_objects, size=1)[0])
            merge_objects = np.array([adj[0] for adj in rag.nodeAdjacency(candidate_obj)],
                                     dtype='uint32')
            if merge_objects.size == 0:
                continue
            # TODO allow merging more than one object ?!
            merge_obj = np.random.choice(merge_objects, size=1)[0]
            if self.ignore_label is not None:
                if self.ignore_label in (candidate_obj, merge_obj):
                    continue
            break
        # make volume with edge coordinates, false merge and mask
        # of resulting object
        edges = np.zeros(rag.numberOfEdges, dtype='uint32')
        edges[rag.findEdge(candidate_obj, merge_obj)] = 1
        # scipy distance trafo is inverted ...
        false_merge = (1. - edge_builder.edgesToVolume(edges, numberOfThreads=self.n_threads)).astype('bool')
        # make input mask
        mask = np.zeros_like(segmentation, dtype='float32')
        mask[segmentation == candidate_obj] = 1
        mask[segmentation == merge_obj] = 1
        print("merged", candidate_obj, merge_obj)
        return false_merge, mask

    # This is not working as well as I'd like
    def _crop_to_object(self, inputs, targets):
        shape = inputs.shape[1:]
        mask = inputs[1]
        masked = np.where(mask == 1.)

        # Find the min and max coordinates of the mask
        start_masked = [mm.min() for mm in masked]
        stop_masked = [mm.max() + 1 for mm in masked]

        extension = [sto - sta for sto, sta in zip(start_masked, stop_masked)]
        # check if the object mask fits into the expected network shapes
        overhangs = [ext % crop for ext, crop in zip(extension, self.crop_to_object)]

        # if not, extend it
        for ii, overhang in enumerate(overhangs):
            # no overhang, we can continue
            if overhang == 0:
                continue
            # check if we can remove the overhang at the start
            if start_masked[ii] - overhang >= 0:
                start_masked[ii] -= overhang
            # check if we can remove the overhang at the stop
            elif stop_masked[ii] + overhang < shape[ii]:
                stop_masked[ii] += overhang
            # if neither is possible, return the not-cropped object
            else:
                return inputs, targets, 4 * (slice(None),)

        # crop to mask
        bb = (slice(None),) + tuple(slice(sta, sto)
                                    for sta, sto in zip(start_masked, stop_masked))
        return inputs[bb], targets[bb], bb

    def batch_function(self, tensors, return_bounding_box=False):
        # we expect data and target input
        assert len(tensors) == 2, str(len(tensors))
        assert all(isinstance(tensor, np.ndarray) for tensor in tensors)
        inputs, target = tensors
        false_merge_mask, object_mask = self._produce_false_merge(target)
        # construct target from false merge edge mask
        false_merge_distance = distance_transform_edt(false_merge_mask)
        false_merge_targets = [(false_merge_distance < fm_dist).astype('float32') for fm_dist in self.target_distances]
        # set to zero outside of object mask
        inverted_mask = np.logical_not(object_mask)
        for fmt in false_merge_targets:
            fmt[inverted_mask] = 0

        # build new inputs (= previous inputs + object masks)
        # and targets (= stacked fm targets + object_mask)
        inputs = np.array([inputs, object_mask.astype(inputs.dtype)])
        targets = np.array([object_mask] + false_merge_targets)
        # crop the output to the selected object, if specified
        if self.crop_to_object is not None:
            inputs, targets, bb = self._crop_to_object(inputs, targets)

        if return_bounding_box:
            assert self.crop_to_object is not None
            return inputs, targets, bb[1:]
        else:
            return inputs, targets
