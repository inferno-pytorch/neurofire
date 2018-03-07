import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import MultiscaleModelTester


class TestHed(unittest.TestCase):
    def test_dense_hed_2d(self):
        from neurofire.models import DenseHED
        shape = (1, 1, 128, 128)
        tester = MultiscaleModelTester(shape, 6 * [shape])
        if cuda.is_available():
            tester.cuda()
        tester(DenseHED(1, 1, 16, block_type_key='dense'))
        tester(DenseHED(1, 1, 16, block_type_key='default'))

    def test_dense_hed_3d(self):
        from neurofire.models import DenseHED
        shape = (1, 1, 64, 64, 64)
        tester = MultiscaleModelTester(shape, 6 * [shape])
        if cuda.is_available():
            tester.cuda()
        tester(DenseHED(1, 1, 16, block_type_key='dense3d',
                        output_type_key='default3d',
                        sampling_type_key='default3d'))
        tester(DenseHED(1, 1, 16, block_type_key='default3d',
                        output_type_key='default3d',
                        sampling_type_key='default3d'))


if __name__ == '__main__':
    unittest.main()
