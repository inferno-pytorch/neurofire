import unittest


class TestCantor(unittest.TestCase):
    def test_cantor_initiator(self):
        import torch
        from torch.autograd import Variable
        import neurofire.models.cantor as cantor
        # Build model
        initiator = cantor.CantorInitiator(3, base_width=30)
        # Build dummy input
        input_shape = [1, 3, 128, 128]
        input = Variable(torch.rand(*input_shape))
        # Get output
        # noinspection PyCallingNonCallable
        output = initiator(input)
        # Validate
        self.assertEqual(len(output), 7)
        self.assertEqual(list(output[0].size()), [1, 120, 16, 16])
        self.assertEqual(list(output[1].size()), [1, 120, 32, 32])
        self.assertEqual(list(output[2].size()), [1, 90, 32, 32])
        self.assertEqual(list(output[3].size()), [1, 90, 64, 64])
        self.assertEqual(list(output[4].size()), [1, 60, 64, 64])
        self.assertEqual(list(output[5].size()), [1, 60, 128, 128])
        self.assertEqual(list(output[6].size()), [1, 30, 128, 128])

    # noinspection PyCallingNonCallable
    def test_cantor_terminator(self):
        import torch
        from torch.autograd import Variable
        import neurofire.models.cantor as cantor
        from inferno.extensions.containers.sequential import Sequential2
        # Build model
        initiator = cantor.CantorInitiator(3, base_width=30)
        terminator = cantor.CantorTerminator(1, base_width=30, activation='Sigmoid')

        # Build dummy input
        input_shape = [1, 3, 128, 128]
        input = Variable(torch.rand(*input_shape))
        initiator_output = initiator(input)
        output = terminator(*initiator_output)
        # Validate
        self.assertEqual(list(output.size()), [1, 1, 128, 128])


if __name__ == '__main__':
    TestCantor().test_cantor_terminator()
