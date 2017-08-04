import os
import unittest


class TestMaster(unittest.TestCase):
    DATA_CONFIG = os.path.join(os.path.dirname(__file__), 'data_config_test.yml')
    DATA_CONFIG_AFFINITIES = os.path.join(os.path.dirname(__file__),
                                          'data_config_test_affinities.yml')
    PLOT_DIRECTORY = os.path.join(os.path.dirname(__file__), 'plots')

    def test_master(self):
        from neurofire.datasets.isbi2012.loaders.master import ISBI2012Dataset
        from inferno.utils.io_utils import print_tensor

        dataset = ISBI2012Dataset.from_config(self.DATA_CONFIG)
        # Get from dataset
        batch = dataset[0]
        # Validate
        self.assertEqual(len(batch), 2)
        for _batch in batch:
            self.assertEqual(list(_batch.size()), [1, 576, 576])    # batch axis added by loader
        # Print to file
        if os.path.exists(self.PLOT_DIRECTORY):
            assert os.path.isdir(self.PLOT_DIRECTORY)
        else:
            os.mkdir(self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[0].numpy()[None, ...],
                     prefix='RAW',
                     directory=self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[1].numpy()[None, ...],
                     prefix='MEM',
                     directory=self.PLOT_DIRECTORY)
        print("Plots printed to {}.".format(self.PLOT_DIRECTORY))

    def test_master_affinity(self):
        from neurofire.datasets.isbi2012.loaders.master import ISBI2012DatasetHDF5
        from inferno.utils.io_utils import print_tensor

        dataset = ISBI2012DatasetHDF5.from_config(self.DATA_CONFIG_AFFINITIES)
        # Get from dataset
        batch = dataset[0]
        # Validate
        self.assertEqual(len(batch), 2)

        self.assertEqual(list(batch[0].size()), [1, 576, 576])
        self.assertEqual(list(batch[1].size()), [2, 576, 576])
        # Print to file
        if os.path.exists(self.PLOT_DIRECTORY):
            assert os.path.isdir(self.PLOT_DIRECTORY)
        else:
            os.mkdir(self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[0].numpy()[None, ...],
                     prefix='AFFINP',
                     directory=self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[1].numpy()[None, ...],
                     prefix='AFFTAR',
                     directory=self.PLOT_DIRECTORY)
        print("Plots printed to {}.".format(self.PLOT_DIRECTORY))


if __name__ == '__main__':
    TestMaster().test_master_affinity()
