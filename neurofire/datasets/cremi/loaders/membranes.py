from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast
from ....transforms.segmentation import Segmentation2Membranes, Segmentation2Affinities
from ....transforms.segmentation import NegativeExponentialDistanceTransform


class MembraneVolume(HDF5VolumeLoader):
    def __init__(self, path, path_in_h5_dataset=None,
                 data_slice=None, name=None, dtype='float32', nedt_gain=None,
                 **slicing_config):
        path_in_h5_dataset = path_in_h5_dataset if path_in_h5_dataset is not None else \
            '/volumes/labels/neuron_ids'
        # Init super
        super(MembraneVolume, self).__init__(path=path, path_in_h5_dataset=path_in_h5_dataset,
                                             data_slice=data_slice, name=name, **slicing_config)
        # Validate and record attributes
        assert isinstance(dtype, str)
        assert isinstance(nedt_gain, (type(None), int, float))
        self.dtype = dtype
        self.nedt_gain = nedt_gain or 1.
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(Segmentation2Membranes(dtype=self.dtype),
                             NegativeExponentialDistanceTransform(gain=self.nedt_gain),
                             Cast(self.dtype))
        return transforms


class AffinityVolume(MembraneVolume):
    def __init__(self, path, path_in_h5_dataset=None, data_slice=None, name=None,
                 dtype='float32', affinity_dim=3, **slicing_config):
        # Set attributes
        self.affinity_dim = affinity_dim
        # Initialize super
        super(AffinityVolume, self).__init__(path, path_in_h5_dataset=path_in_h5_dataset,
                                             data_slice=data_slice, name=name,
                                             dtype=dtype, nedt_gain=None,
                                             **slicing_config)
        # We don't need a nedt_gain anymore
        del self.nedt_gain

    def get_transforms(self):
        # The Segmentation2Affinities adds a channel dimension
        transforms = Compose(Segmentation2Affinities(self.affinity_dim,
                                                     add_singleton_channel_dimension=True),
                             Cast(self.dtype))
        return transforms
