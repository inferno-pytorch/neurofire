from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric import TIFVolumeLoader, HDF5VolumeLoader
from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast, Normalize
from inferno.io.transform.image import AdditiveGaussianNoise


class RawVolume(TIFVolumeLoader):
    def __init__(self, path, dtype='float32', **slicing_config):
        # Init super
        super(RawVolume, self).__init__(path=path, **slicing_config)
        # Record attributes
        assert isinstance(dtype, str)
        self.dtype = dtype
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(
            Normalize(),
            # after normalize since raw data comes in uint8
            AdditiveGaussianNoise(sigma=.025),
            Cast(self.dtype))
        return transforms


class RawVolumeHDF5(HDF5VolumeLoader):
    def __init__(self, path, path_in_h5_dataset, dtype='float32', **slicing_config):
        # Init super
        super(RawVolumeHDF5, self).__init__(path=path, path_in_h5_dataset=path_in_h5_dataset, **slicing_config)
        assert isinstance(dtype, str)
        self.dtype = dtype
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(
            Normalize(),
            # after normalize since raw data comes in uint8
            AdditiveGaussianNoise(sigma=.025),
            Cast(self.dtype))
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        path = config.get('path')
        path_in_h5_dataset = config.get('path_in_h5_dataset', None)
        data_slice = config.get('data_slice', None)
        dtype = config.get('dtype', 'float32')
        slicing_config = config.get('slicing_config', None)
        print(path)
        return cls(path, path_in_h5_dataset=path_in_h5_dataset,
                   data_slice=data_slice, dtype=dtype,
                   **slicing_config)
