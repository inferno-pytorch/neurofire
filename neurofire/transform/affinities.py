import numpy as np
from inferno.io.transform import Transform
import inferno.utils.python_utils as pyu
from .segmentation import DtypeMapping

try:
    import affinities
    HAVE_AFFINITIES = True
except ImportError as e:
    HAVE_AFFINITIES = False
    # print("Couldn't find 'affinities' module, fast affinity calculation is not available")


class Segmentation2Affinities(Transform, DtypeMapping):
    def __init__(self, offsets, dtype='float32',
                 retain_mask=False,
                 ignore_label=None,
                 **super_kwargs):
        assert HAVE_AFFINITIES, "Couldn't find 'affinities' module, affinity calculation is not available"
        assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        super(Segmentation2Affinities, self).__init__(**super_kwargs)
        self.dim = len(offsets[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(off) == self.dim for off in offsets[1:])
        self.offsets = offsets
        self.dtype = dtype
        self.retain_mask = retain_mask
        self.ignore_label = ignore_label
        # self.add_singleton_channel_dimension = add_singleton_channel_dimension

    def tensor_function(self, tensor):
        # need to cast tensor to np array ?!
        if self.ignore_label is not None:
            output, mask = affinities.compute_affinities(tensor, self.offsets,
                                                         ignore_label=self.ignore_label,
                                                         have_ignore_label=True)
        else:
            output, mask = affinities.compute_affinities(tensor, self.offsets)

        # Cast to be sure
        if not output.dtype == self.dtype:
            output = output.astype(self.dtype)

        # We might want to carry the mask along.
        # If this is the case, we insert it after the targets.
        if self.retain_mask:
            if mask.dtype != self.dtype:
                mask = mask.astype(self.dtype)
            output = np.concatenate((output, mask), axis=0)

        return output


class Segmentation2MultiscaleAffinities(Transform, DtypeMapping):
    def __init__(self, block_shapes, dtype='float32', ignore_label=None,
                 retain_mask=False, **super_kwargs):
        super(Segmentation2MultiscaleAffinities, self).__init__(**super_kwargs)
        assert HAVE_AFFINITIES, "Couldn't find 'affinities' module, affinity calculation is not available"
        assert pyu.is_listlike(block_shapes)
        self.block_shapes = block_shapes
        self.dim = len(block_shapes[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(bs) == self.dim for bs in block_shapes[1:])

        self.dtyp = dtype
        self.ignore_label = ignore_label
        self.retain_mask = retain_mask

    def tensor_function(self, tensor):
        outputs = []
        for bs in self.block_shapes:
            # need to cast tensor to np array ?!
            if self.ignore_label is None:
                output, mask = affinities.compute_multiscale_affinities(tensor, bs,
                                                                        ignore_label=self.ignore_label,
                                                                        have_ignore_label=True)
            else:
                output, mask = affinities.compute_multiscale_affinities(tensor, bs)

            # Cast to be sure
            if not output.dtype == self.dtype:
                output = output.astype(self.dtype)

            # We might want to carry the mask along.
            # If this is the case, we insert it after the targets.
            if self.retain_mask:
                if mask.dtype != self.dtype:
                    mask = mask.astype(self.dtype)
                output = np.concatenate((output, mask), axis=0)
            outputs.append(output)
        return outputs