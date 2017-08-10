from inferno.io.core import Zip, Concatenate
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform, CenterCrop
from inferno.utils.io_utils import yaml2dict

from torch.utils.data.dataloader import DataLoader

from .membranes import MembraneVolume, AffinityVolume
from .raw import RawVolume


class CREMIDataset(Zip):
    def __init__(self, name, volume_config, slicing_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'raw' in volume_config
        assert ('membranes' in volume_config) != ('affinities' in volume_config)
        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))
        raw_volume_kwargs.update(slicing_config)
        # Build raw volume
        self.raw_volume = RawVolume(name=name, **raw_volume_kwargs)
        # Get kwargs for membrane or affinity volumes
        if 'membranes' in volume_config:
            membrane_volume_kwargs = dict(volume_config.get('membranes'))
            membrane_volume_kwargs.update(slicing_config)
            # Build membrane volume
            self.membrane_or_affinity_volume = MembraneVolume(name=name, **membrane_volume_kwargs)
        elif 'affinities' in volume_config:
            affinity_volume_kwargs = dict(volume_config.get('affinities'))
            affinity_volume_kwargs.update(slicing_config)
            # Build affinity volume
            self.membrane_or_affinity_volume = AffinityVolume(name=name, **affinity_volume_kwargs)
            pass
        else:
            raise NotImplementedError
        # Initialize zip
        super(CREMIDataset, self).__init__(self.raw_volume, self.membrane_or_affinity_volume,
                                           sync=True)
        # Set master config (for transforms)
        self.master_config = {} if master_config is None else master_config
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(RandomFlip3D(),
                             RandomRotate())
        if 'elastic_transform' in self.master_config:
            # Elastic transforms can be skipped by setting elastic_transform to false in the
            # yaml config file.
            if self.master_config.get('elastic_transform'):
                elastic_transform_config = self.master_config.get('elastic_transform')
                transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                                sigma=elastic_transform_config.get('sigma', 50.),
                                                order=elastic_transform_config.get('order', 1)))
        else:
            # Preserve legacy behaviour
            transforms.add(ElasticTransform(alpha=2000., sigma=50.))
        if self.master_config.get('crop_after_elastic_transform', False):
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(CenterCrop(**self.master_config.get('crop_after_elastic_transform', {})))
        return transforms


class CREMIDatasets(Concatenate):
    def __init__(self, names, volume_config, slicing_config, master_config=None):
        # Make datasets and concatenate
        datasets = [CREMIDataset(name=name,
                                 volume_config=volume_config,
                                 slicing_config=slicing_config,
                                 master_config=master_config)
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
        master_config = config.get('master_config')
        return cls(names=names, volume_config=volume_config,
                   slicing_config=slicing_config, master_config=master_config)


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
