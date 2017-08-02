import unittest


class TestUNet2D(unittest.TestCase):
    CUDA = True

    # apply unet to a random tensor
    def test_unet2D(self):
        import torch
        from torch.autograd import Variable
        from neurofire.models import UNet2D

        input_shape = [1, 1, 256, 256]
        model = UNet2D(1, 1, n_scale=4)
        input = Variable(torch.rand(*input_shape))
        output = model(input)
        self.assertEqual(list(output.size()), input_shape)

    def test_unet2D_with_tester(self):
        from neurofire.models import UNet2D
        from inferno.utils.model_utils import ModelTester
        # I've written a small tester that does away with the boilerplate.
        tester = ModelTester(input_shape=(1, 1, 256, 256),
                             expected_output_shape=(1, 1, 256, 256))
        if self.CUDA:
            # To test with cuda, you could use:
            tester.cuda()(UNet2D(1, 1, n_scale=4))
        else:
            tester(UNet2D(1, 1, n_scale=4))
        # You're fine if this made it here. :)

    def test_layer_with_tester(self):
        from inferno.extensions.layers.convolutional import ConvELU2D
        from inferno.utils.model_utils import ModelTester

        tester = ModelTester(input_shape=(1, 1, 256, 256),
                             expected_output_shape=(1, 1, 256, 256))
        if self.CUDA:
            tester.cuda()(ConvELU2D(1, 1, 3))
        else:
            tester(ConvELU2D(1, 1, 3))

if __name__ == '__main__':
    unittest.main()
