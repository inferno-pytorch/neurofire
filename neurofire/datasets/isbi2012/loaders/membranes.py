from inferno.io.volumetric import TIFVolumeLoader, HDF5VolumeLoader
from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast
from ....transform.segmentation import NegativeExponentialDistanceTransform


class MembraneVolume(TIFVolumeLoader):
    def __init__(self, path, dtype='float32', nedt_gain=None, **slicing_config):
        # Init super
        super(MembraneVolume, self).__init__(path=path, **slicing_config)
        # Validate and record attributes
        assert isinstance(dtype, str)
        assert isinstance(nedt_gain, (type(None), int, float))
        self.dtype = dtype
        self.nedt_gain = nedt_gain or 1.
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        # no NEDT inversion for ISBI since labels give neuron rather than boundary probabilities
        transforms = Compose(
            NegativeExponentialDistanceTransform(gain=self.nedt_gain, invert=False),
            Cast(self.dtype)
        )
        return transforms


class MembraneVolumeHDF5(HDF5VolumeLoader):
    def __init__(self, path, path_in_h5_dataset, dtype='float32', nedt_gain=None,
                 **slicing_config):
        super(MembraneVolumeHDF5, self).__init__(path=path, path_in_h5_dataset=path_in_h5_dataset,
                                                 **slicing_config)
        assert isinstance(dtype, str)
        assert isinstance(nedt_gain, (type(None), int, float))
        self.dtype = dtype
        self.nedt_gain = nedt_gain or 1.
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(
            NegativeExponentialDistanceTransform(gain=self.nedt_gain, invert=False),
            Cast(self.dtype)
        )
        return transforms
