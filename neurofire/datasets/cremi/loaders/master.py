from inferno.io.core import Zip, Concatenate
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict

from torch.utils.data.dataloader import DataLoader

from .membranes import MembraneVolume
from .raw import RawVolume


class CREMIDataset(Zip):
    def __init__(self, name, volume_config, slicing_config):
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
        self.raw_volume = RawVolume(name=name, **raw_volume_kwargs)
        self.membrane_volume = MembraneVolume(name=name, **membrane_volume_kwargs)
        # Initialize zip
        super(CREMIDataset, self).__init__(self.raw_volume, self.membrane_volume, sync=True)
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(RandomFlip3D(),
                             RandomRotate(),
                             # Hard coded for now:
                             ElasticTransform(alpha=2000., sigma=50.))
        return transforms


class CREMIDatasets(Concatenate):
    def __init__(self, names, volume_config, slicing_config):
        # Make datasets and concatenate
        datasets = [CREMIDataset(name=name,
                                 volume_config=volume_config,
                                 slicing_config=slicing_config)
                    for name in names]
        # Concatenate
        super(CREMIDatasets, self).__init__(*datasets)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = AsTorchBatch(3)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        names = config.get('dataset_names')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        return cls(names=names, volume_config=volume_config,
                   slicing_config=slicing_config)


def get_cremi_loaders(config):
    """
    Gets CREMI Loaders given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    datasets = CREMIDatasets.from_config(config)
    loader = DataLoader(datasets, **config.get('loader_config'))
    return loader
