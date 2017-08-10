import os
import unittest


class TestMaster(unittest.TestCase):
    DATA_CONFIG = os.path.join(os.path.dirname(__file__), 'data_config_test_{}.yml')
    PLOT_DIRECTORY = os.path.join(os.path.dirname(__file__), 'plots')

    def _test_master_membranes_or_affinities(self, data_config):
        from neurofire.datasets.cremi.loaders.master import CREMIDatasets
        from inferno.utils.io_utils import print_tensor

        dataset = CREMIDatasets.from_config(data_config)
        # Get from dataset
        batch = dataset[0]
        # Validate
        self.assertEqual(len(batch), 2)
        for _batch in batch:
            self.assertEqual(list(_batch.size()), [1, 3, 512, 512])
        # Print to file
        if os.path.exists(self.PLOT_DIRECTORY):
            assert os.path.isdir(self.PLOT_DIRECTORY)
        else:
            os.mkdir(self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[0].numpy()[None, ...],
                     prefix='RAW',
                     directory=self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[1].numpy()[None, ...],
                     prefix='TAR',
                     directory=self.PLOT_DIRECTORY)
        print("Plots printed to {}.".format(self.PLOT_DIRECTORY))

    def test_master_membranes(self):
        data_config = self.DATA_CONFIG.format('membranes')
        from neurofire.datasets.cremi.loaders.master import CREMIDatasets
        from inferno.utils.io_utils import print_tensor

        dataset = CREMIDatasets.from_config(data_config)
        # Get from dataset
        batch = dataset[0]
        # Validate
        self.assertEqual(len(batch), 2)
        for _batch in batch:
            self.assertEqual(list(_batch.size()), [1, 3, 512, 512])
        # Print to file
        if os.path.exists(self.PLOT_DIRECTORY):
            assert os.path.isdir(self.PLOT_DIRECTORY)
        else:
            os.mkdir(self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[0].numpy()[None, ...],
                     prefix='MEMRAW',
                     directory=self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[1].numpy()[None, ...],
                     prefix='MEMTAR',
                     directory=self.PLOT_DIRECTORY)
        print("Plots printed to {}.".format(self.PLOT_DIRECTORY))

    def test_master_affinities(self):
        data_config = self.DATA_CONFIG.format('affinities')
        from neurofire.datasets.cremi.loaders.master import CREMIDatasets
        from inferno.utils.io_utils import print_tensor

        dataset = CREMIDatasets.from_config(data_config)
        # Get from dataset
        batch = dataset[0]
        # Validate
        self.assertEqual(len(batch), 2)
        self.assertEqual(list(batch[0].size()), [1, 5, 512, 512])
        self.assertEqual(list(batch[1].size())[1:], [5, 512, 512])
        self.assertIn(batch[1].size(0), [2, 3])
        # Print to file
        if os.path.exists(self.PLOT_DIRECTORY):
            assert os.path.isdir(self.PLOT_DIRECTORY)
        else:
            os.mkdir(self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[0].numpy()[None, ...],
                     prefix='AFFRAW',
                     directory=self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[1].numpy()[None, ...],
                     prefix='AFFTAR',
                     directory=self.PLOT_DIRECTORY)
        print("Plots printed to {}.".format(self.PLOT_DIRECTORY))

    def test_master_affinities_multi_order(self):
        data_config = self.DATA_CONFIG.format('affinities_multi_order')
        from neurofire.datasets.cremi.loaders.master import CREMIDatasets
        from inferno.utils.io_utils import print_tensor

        dataset = CREMIDatasets.from_config(data_config)
        # Get from dataset
        batch = dataset[0]
        # Validate
        self.assertEqual(len(batch), 2)
        self.assertEqual(list(batch[0].size()), [1, 5, 512, 512])
        self.assertEqual(list(batch[1].size())[1:], [5, 512, 512])
        self.assertIn(batch[1].size(0), [2 * 4, 3 * 4])
        # Print to file
        if os.path.exists(self.PLOT_DIRECTORY):
            assert os.path.isdir(self.PLOT_DIRECTORY)
        else:
            os.mkdir(self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[0].numpy()[None, ...],
                     prefix='MOAFFRAW',
                     directory=self.PLOT_DIRECTORY)
        print_tensor(tensor=batch[1].numpy()[None, ...],
                     prefix='MOAFFTAR',
                     directory=self.PLOT_DIRECTORY)
        print("Plots printed to {}.".format(self.PLOT_DIRECTORY))


if __name__ == '__main__':
    TestMaster().test_master_affinities_multi_order()
    # TestMaster().test_master_membranes()
    # unittest.main()
