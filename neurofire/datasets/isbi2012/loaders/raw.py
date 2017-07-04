from inferno.io.volumetric import tifVolumeLoader
from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast, Normalize, GaussNoise


class RawVolume(tifVolumeLoader):
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
                             GaussNoise(sigma=.025), # after normalize since raw data comes in uint8
                             Cast(self.dtype))
        return transforms
