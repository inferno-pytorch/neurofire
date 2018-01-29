from torch.utils.data.dataloader import DataLoader
from inferno.io.core import ZipReject, Concatenate
from inferno.io.transform import Compose
from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict
from inferno.io.transform.generic import AsTorchBatch

from neurofire.transform.volume import RejectNonZeroThreshold
from neurofire.transform.false_merge_gt import ArtificialFalseMerges
from ..cremi.loaders import RawVolume, SegmentationVolume


class CREMIMerges(ZipReject):
    def __init__(self, name, volume_config, slicing_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'raw' in volume_config
        assert 'segmentation' in volume_config

        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))

        # Build raw volume
        raw_volume_kwargs.update(slicing_config)
        self.raw_volume = RawVolume(name=name, **raw_volume_kwargs)

        # Get kwargs for segmentation volume
        segmentation_volume_kwargs = dict(volume_config.get('segmentation'))
        segmentation_volume_kwargs.update(slicing_config)
        self.affinity_offsets = segmentation_volume_kwargs.pop('affinity_offsets', None)
        # Build segmentation volume
        self.segmentation_volume = SegmentationVolume(name=name, **segmentation_volume_kwargs)

        rejection_threshold = volume_config.get('rejection_threshold', 0.5)
        # Initialize zipreject
        super(CREMIMerges, self).__init__(self.raw_volume, self.segmentation_volume,
                                          sync=True, rejection_dataset_indices=1,
                                          rejection_criterion=RejectNonZeroThreshold(rejection_threshold))
        # Set master config (for transforms)
        self.master_config = {} if master_config is None else master_config
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(RandomFlip3D(),
                             RandomRotate())

        # Elastic transforms can be skipped by setting elastic_transform to false in the
        # yaml config file.
        if self.master_config.get('elastic_transform'):
            elastic_transform_config = self.master_config.get('elastic_transform')
            transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                            sigma=elastic_transform_config.get('sigma', 50.),
                                            order=elastic_transform_config.get('order', 0)))
        transforms.add(ArtificialFalseMerges(**self.master_config.get('false_merge_config')))
        return transforms


class CREMIMergeDatasets(Concatenate):
    def __init__(self, names, volume_config, slicing_config, master_config=None):
        # Make datasets and concatenate
        datasets = [CREMIMerges(name=name,
                                volume_config=volume_config,
                                slicing_config=slicing_config,
                                master_config=master_config)
                    for name in names]
        # Concatenate
        super(CREMIMergeDatasets, self).__init__(*datasets)
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


def get_cremi_merge_loaders(config):
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
    datasets = CREMIMergeDatasets.from_config(config)
    loader = DataLoader(datasets, **config.get('loader_config'))
    return loader
