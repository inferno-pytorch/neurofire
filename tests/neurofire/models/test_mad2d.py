import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import MultiscaleModelTester


class TestMad(unittest.TestCase):
    # FIXME this will always fail, because the tester expects 
    # that all outputs have the original size again
    # this is obviously not the case when we return downscaled 
    # outputs. Could be fixed by giving scaling factors as additional
    # arguments to the tester
    @unittest.expectedFailure
    def test_mad_2d(self):
        from neurofire.models import MAD2D
        shape = (1, 1, 256, 256)
        tester = MultiscaleModelTester(shape, 6 * [shape])
        if cuda.is_available():
            tester.cuda()
        tester(MAD2D(1, 1, 64, 2))


if __name__ == '__main__':
    unittest.main()
