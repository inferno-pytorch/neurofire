import numpy as np
import inferno.utils.python_utils as pyu
from inferno.io.transform import Transform
from ..criteria.multi_scale_loss import Downsampler
from .segmentation import DtypeMapping

try:
    import affogato
    HAVE_AFFOGATO = True
except ImportError as e:
    HAVE_AFFOGATO = False
    # print("Couldn't find 'affinities' module, fast affinity calculation is not available")


# TODO add more options (membrane prediction)
# helper function that returns affinity transformation from config
def affinity_config_to_transform(**affinity_config):
    assert ('offsets' in affinity_config) != ('block_shapes' in affinity_config), \
        "Need either 'offsets' or 'block_shapes' parameter in config"

    if 'offsets' in affinity_config:
        return Segmentation2Affinities(**affinity_config)
    else:
        return Segmentation2MultiscaleAffinities(**affinity_config)


class Segmentation2Affinities(Transform, DtypeMapping):
    def __init__(self, offsets, dtype='float32',
                 retain_mask=False, ignore_label=None,
                 retain_segmentation=False, **super_kwargs):
        assert HAVE_AFFOGATO, "Couldn't find 'affogato' module, affinity calculation is not available"
        assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        super(Segmentation2Affinities, self).__init__(**super_kwargs)
        self.dim = len(offsets[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(off) == self.dim for off in offsets[1:])
        self.offsets = offsets
        self.dtype = dtype
        self.retain_mask = retain_mask
        self.ignore_label = ignore_label
        self.retain_segmentation = retain_segmentation
        # self.add_singleton_channel_dimension = add_singleton_channel_dimension

    def tensor_function(self, tensor):
        from affogato.affinities import compute_affinities
        # need to cast tensor to np array ?!
        if self.ignore_label is not None:
            output, mask = compute_affinities(tensor.squeeze(), self.offsets,
                                              ignore_label=self.ignore_label,
                                              have_ignore_label=True)
        else:
            output, mask = compute_affinities(tensor.squeeze(), self.offsets)

        # Cast to be sure
        if not output.dtype == self.dtype:
            output = output.astype(self.dtype)

        # We might want to carry the mask along.
        # If this is the case, we insert it after the targets.
        if self.retain_mask:
            output = np.concatenate((output, mask.astype(self.dtype, copy=False)),
                                    axis=0)

        # We might want to carry the segmentation along for validation.
        # If this is the case, we insert it before the targets.
        if self.retain_segmentation:
            output = np.concatenate((tensor.astype(self.dtype, copy=False), output),
                                    axis=0)

        return output


class Segmentation2MultiscaleAffinities(Transform, DtypeMapping):
    def __init__(self, block_shapes, dtype='float32', ignore_label=None,
                 retain_mask=False, retain_segmentation=False, **super_kwargs):
        super(Segmentation2MultiscaleAffinities, self).__init__(**super_kwargs)
        assert HAVE_AFFOGATO, "Couldn't find 'affogato' module, affinity calculation is not available"
        assert pyu.is_listlike(block_shapes)
        self.block_shapes = block_shapes
        self.dim = len(block_shapes[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(bs) == self.dim for bs in block_shapes[1:])

        self.dtype = dtype
        self.ignore_label = ignore_label
        self.retain_mask = retain_mask
        self.retain_segmentation = retain_segmentation
        if self.retain_segmentation:
            self.downsamplers = [Downsampler(bs) for bs in self.block_shapes]

    def tensor_function(self, tensor):
        from affogato.affinities import compute_multiscale_affinities, compute_affinities
        outputs = []
        for ii, bs in enumerate(self.block_shapes):
            # if the block shape is all ones, we can compute normal affinities
            # with nearest neighbor offsets. This should yield the same result,
            # but should be more efficient.
            original_scale = all(s == 1 for s in bs)
            if original_scale:
                offsets = [[0 if i != d else -1 for i in range(self.dim)]
                           for d in range(self.dim)]
                output, mask = compute_affinities(tensor.squeeze().astype('uint64'), offsets,
                                                  ignore_label=0 if self.ignore_label is None else self.ignore_label,
                                                  have_ignore_label=False if self.ignore_label is None else True)
            else:
                output, mask = compute_multiscale_affinities(tensor.squeeze().astype('uint64'), bs,
                                                             ignore_label=0 if self.ignore_label is None else self.ignore_label,
                                                             have_ignore_label=False if self.ignore_label is None else True)

            # Cast to be sure
            if not output.dtype == self.dtype:
                output = output.astype(self.dtype)

            # We might want to carry the mask along.
            # If this is the case, we insert it after the targets.
            if self.retain_mask:
                output = np.concatenate((output, mask.astype(self.dtype, copy=False)), axis=0)
            # We might want to carry the segmentation along for validation.
            # If this is the case, we insert it before the targets for the original scale.
            if self.retain_segmentation:
                output = np.concatenate((self.downsamplers[ii](tensor.astype(self.dtype, copy=False)),
                                         output), axis=0)
            outputs.append(output)

        return outputs
