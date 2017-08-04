from inferno.io.core import Zip
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict

from torch.utils.data.dataloader import DataLoader

from .membranes import MembraneVolume, MembraneVolumeHDF5, AffinityVolumeHDF5
from .raw import RawVolume, RawVolumeHDF5


class ISBI2012Dataset(Zip):
    def __init__(self, volume_config, slicing_config):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'raw' in volume_config
        assert 'membranes' in volume_config
        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))
        raw_volume_kwargs.update(slicing_config)
        membrane_volume_kwargs = dict(volume_config.get('membranes'))
        membrane_volume_kwargs.update(slicing_config)
        # Build volumes
        self.raw_volume = RawVolume(**raw_volume_kwargs)
        self.membrane_volume = MembraneVolume(**membrane_volume_kwargs)
        # Initialize zip
        super(ISBI2012Dataset, self).__init__(self.raw_volume, self.membrane_volume, sync=True)
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(RandomFlip3D(),
                             RandomRotate(),
                             ElasticTransform(alpha=2000., sigma=50.),  # Hard coded for now
                             AsTorchBatch(2))
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        return cls(volume_config=volume_config, slicing_config=slicing_config)


class ISBI2012DatasetHDF5(Zip):
    def __init__(self, volume_config, slicing_config):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'raw' in volume_config
        assert ('membranes' in volume_config) != ('affinities' in volume_config)
        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))
        raw_volume_kwargs.update(slicing_config)
        # Build raw volume
        self.raw_volume = RawVolumeHDF5(**raw_volume_kwargs)
        # Build membrane or affinity volume
        if 'membranes' in volume_config:
            membrane_volume_kwargs = dict(volume_config.get('membranes'))
            membrane_volume_kwargs.update(slicing_config)
            self.membrane_or_affinity_volume = MembraneVolumeHDF5(**membrane_volume_kwargs)
        elif 'affinities' in volume_config:
            affinity_volume_kwargs = dict(volume_config.get('affinities'))
            affinity_volume_kwargs.update(slicing_config)
            self.membrane_or_affinity_volume = AffinityVolumeHDF5(**affinity_volume_kwargs)
        # Initialize zip
        super(ISBI2012DatasetHDF5, self).__init__(self.raw_volume, self.membrane_or_affinity_volume, sync=True)
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(RandomFlip3D(),
                             RandomRotate(),
                             ElasticTransform(alpha=2000., sigma=50.),  # Hard coded for now
                             AsTorchBatch(2))
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        return cls(volume_config=volume_config, slicing_config=slicing_config)


def get_isbi_loader(config, use_hdf5=False):
    """
    Gets ISBI2012 Loader given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.
    use_hdf5: bool
        flag to specify whether data is hdf5 or tiff

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    if use_hdf5:
        dataset = ISBI2012DatasetHDF5.from_config(config)
    else:
        dataset = ISBI2012Dataset.from_config(config)
    loader = DataLoader(dataset, **config.get('loader_config'))
    return loader
