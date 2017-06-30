from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast, Normalize


class RawVolume(HDF5VolumeLoader):
    def __init__(self, path, path_in_h5_dataset=None,
                 data_slice=None, name=None, dtype='float32', **slicing_config):
        path_in_h5_dataset = path_in_h5_dataset if path_in_h5_dataset is not None else \
            '/volumes/raw'
        # Init super
        super(RawVolume, self).__init__(path=path, path_in_h5_dataset=path_in_h5_dataset,
                                        data_slice=data_slice, name=name, **slicing_config)
        # Record attributes
        assert isinstance(dtype, str)
        self.dtype = dtype
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(Normalize(),
                             Cast(self.dtype))
        return transforms
