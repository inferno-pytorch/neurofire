import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import ModelTester


class ModelTest(unittest.TestCase):
    def test_dense_unet_2d(self):
        from neurofire.models.dense_unet import DUNet2D
        tester = ModelTester((1, 1, 512, 512), (1, 1, 512, 512))
        if cuda.is_available():
            tester.cuda()
        tester(DUNet2D(1, 1))

    def test_dense_unet_2p5d(self):
        from neurofire.models.dense_unet import DUNet2p5D
        tester = ModelTester((1, 1, 3, 512, 512), (1, 1, 3, 512, 512))
        if cuda.is_available():
            tester.cuda()
        tester(DUNet2p5D(1, 1, 3))

    def test_dense_unet_3d(self):
        from neurofire.models.dense_unet import DUNet3D
        tester = ModelTester((1, 1, 32, 256, 256), (1, 1, 32, 256, 256))
        if cuda.is_available():
            tester.cuda()
        tester(DUNet3D(1, 1, sampling_scale=2))

    def test_dense_unet_g_3d(self):
        from neurofire.models.dense_unet import DUNet3D
        tester = ModelTester((1, 1, 32, 256, 256), (1, 1, 32, 256, 256))
        if cuda.is_available():
            tester.cuda()
        tester(DUNet3D(1, 1, sampling_scale=2,
                       encoder_type_key='G', decoder_type_key='G', base_type_key='G'))


if __name__ == '__main__':
    unittest.main()
