from inferno.io.volumetric import TIFVolumeLoader
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
