import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import MultiscaleModelTester


class TestWnetMultiscale(unittest.TestCase):
    def test_wnet_multiscale_2d(self):
        from neurofire.models import WNet2DMultiscale
        input_shape = (1, 1, 512, 512)
        output_shape = ((1, 1, 512, 512),
                        (1, 1, 256, 256),
                        (1, 1, 128, 128),
                        (1, 1, 64, 64))
        tester = MultiscaleModelTester(input_shape, output_shape)
        if cuda.is_available():
            tester.cuda()
        tester(WNet2DMultiscale(1, 1,
                                initial_num_fmaps=12,
                                fmap_growth=3))


if __name__ == '__main__':
    unittest.main()
