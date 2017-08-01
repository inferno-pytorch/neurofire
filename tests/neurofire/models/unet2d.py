import unittest


class TestUNet2D(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
