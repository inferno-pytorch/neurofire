import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import MultiscaleModelTester


class TestUnetMultiscale(unittest.TestCase):
    def test_unet_multiscale_2d(self):
        from neurofire.models import UNet2DMultiscale
        input_shape = (1, 1, 512, 512)
        output_shape = ((1, 1, 512, 512),
                        (1, 1, 256, 256),
                        (1, 1, 128, 128),
                        (1, 1, 64, 64))
        tester = MultiscaleModelTester(input_shape, output_shape)
        if cuda.is_available():
            tester.cuda()
        tester(UNet2DMultiscale(1, 1,
                                initial_num_fmaps=12,
                                fmap_growth=3))

    def test_unet_multiscale_3d(self):
        from neurofire.models import UNet3DMultiscale

        input_shape = (1, 1, 32, 256, 256)
        output_shape = ((1, 1, 32, 256, 256),
                        (1, 1, 16, 128, 128),
                        (1, 1, 8, 64, 64),
                        (1, 1, 4, 32, 32))

        tester = MultiscaleModelTester(input_shape, output_shape)
        if cuda.is_available():
            tester.cuda()

        # test default unet 3d
        tester(UNet3DMultiscale(1, 1,
                                initial_num_fmaps=12,
                                fmap_growth=3,
                                scale_factor=2))

        # test with residual block
        tester(UNet3DMultiscale(1, 1,
                                initial_num_fmaps=12,
                                fmap_growth=3,
                                scale_factor=2,
                                add_residual_connections=True))

        # test unet 3d with anisotropic sampling
        output_shape = ((1, 1, 32, 256, 256),
                        (1, 1, 32, 128, 128),
                        (1, 1, 32, 64, 64),
                        (1, 1, 32, 32, 32))

        tester = MultiscaleModelTester(input_shape, output_shape)
        if cuda.is_available():
            tester.cuda()

        tester(UNet3DMultiscale(1, 1,
                                initial_num_fmaps=12,
                                fmap_growth=3,
                                scale_factor=[(1, 2, 2),
                                              (1, 2, 2),
                                              (1, 2, 2)]))


if __name__ == '__main__':
    unittest.main()
