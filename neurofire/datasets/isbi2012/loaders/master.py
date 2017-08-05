from inferno.io.core import Zip
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict

from torch.utils.data.dataloader import DataLoader

from .membranes import MembraneVolume, MembraneVolumeHDF5
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
        raw_volume_format = self.detect_hdf5_or_tif(**raw_volume_kwargs)
        if raw_volume_format == 'hdf5':
            self.raw_volume = RawVolumeHDF5(**raw_volume_kwargs)
        elif raw_volume_format == 'tif':
            self.raw_volume = RawVolume(**raw_volume_kwargs)
        else:
            raise NotImplementedError
        membrane_volume_format = self.detect_hdf5_or_tif(**membrane_volume_kwargs)
        if membrane_volume_format == 'hdf5':
            self.membrane_volume = MembraneVolumeHDF5(**membrane_volume_kwargs)
        elif membrane_volume_format == 'tif':
            self.membrane_volume = MembraneVolume(**membrane_volume_kwargs)
        else:
            raise NotImplementedError
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

    @staticmethod
    def detect_hdf5_or_tif(**volume_kwargs):
        assert 'path' in volume_kwargs, "Path to volume not provided."
        if volume_kwargs['path'].endswith('.h5'):
            return 'hdf5'
        elif volume_kwargs['path'].endswith('.tif') or volume_kwargs['path'].endswith('.tiff'):
            return 'tif'
        else:
            raise NotImplementedError("Unrecognized file format.")

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        return cls(volume_config=volume_config, slicing_config=slicing_config)


def get_isbi_loader(config):
    """
    Gets ISBI2012 Loader given a the path to a configuration file. Supported file formats are
    HDF5 (.h5) and TIF (.tif or .tiff).

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
    dataset = ISBI2012Dataset.from_config(config)
    loader = DataLoader(dataset, **config.get('loader_config'))
    return loader
