import unittest
import numpy as np
from neurofire.criteria.malis.malis import MalisLoss, ConstrainedMalisLoss, Malis
import torch
from torch.autograd import Variable


class TestMalis(unittest.TestCase):
    BATCH_SIZE = 10

    def generate_test_data(self, generate_2d=True, reduce_num_affinity_channels=False):
        shape = (100, 100) if generate_2d else (100, 100, 100)
        dim = 2 if generate_2d else 3

        affinities = np.random.random((dim,) + shape).astype('float32')
        groundtruth = np.zeros(shape, dtype='int64')

        current_label = 0
        for x in range(groundtruth.shape[0]):
            for y in range(groundtruth.shape[1]):
                groundtruth[x, y] = current_label
                # change label with probability .3
                if np.random.random() > .7:
                    current_label += 1

        affinities = np.array([affinities.copy() for _ in range(self.BATCH_SIZE)])
        # groundtruth needs an extra channel axis
        groundtruth = np.array([groundtruth.copy() for _ in range(self.BATCH_SIZE)])[:, None, ...]
        if reduce_num_affinity_channels:
            affinities = affinities[:, 1:, ...]
        return affinities, groundtruth

    def test_malis_loss(self):
        affinities, ground_truth = self.generate_test_data()
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        criterion = MalisLoss()
        # Evaluate criterion
        loss = criterion(affinities, ground_truth).sum()
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)

    def test_malis_auto_loss_3d(self):
        # This is slow
        affinities, ground_truth = self.generate_test_data(generate_2d=False)
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        criterion = MalisLoss(malis_dim='auto')
        # Evaluate criterion
        loss = criterion(affinities, ground_truth).sum()
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)
        self.assertEqual(affinities.grad.size(1), 3)
        self.assertEqual(affinities.grad.dim(), 5)

    def test_malis_2d_loss_3d(self):
        affinities, ground_truth = self.generate_test_data(generate_2d=False,
                                                           reduce_num_affinity_channels=True)
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        criterion = MalisLoss(malis_dim=2)
        # Evaluate criterion
        loss = criterion(affinities, ground_truth).sum()
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)
        self.assertEqual(affinities.grad.size(1), 2)
        self.assertEqual(affinities.grad.dim(), 5)

    def test_constrained_malis_auto_loss_2d(self):
        affinities, ground_truth = self.generate_test_data()
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        criterion = ConstrainedMalisLoss()
        # Evaluate criterion
        loss = criterion(affinities, ground_truth).sum()
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)

    def test_constrained_malis_auto_loss_3d(self):
        affinities, ground_truth = self.generate_test_data(generate_2d=False)
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        criterion = ConstrainedMalisLoss()
        # Evaluate criterion
        loss = criterion(affinities, ground_truth).sum()
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)
        self.assertEqual(affinities.grad.size(1), 3)
        self.assertEqual(affinities.grad.dim(), 5)

    def test_constrained_malis_2d_loss_3d(self):
        affinities, ground_truth = self.generate_test_data(generate_2d=False,
                                                           reduce_num_affinity_channels=True)
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        criterion = ConstrainedMalisLoss(malis_dim=2)
        # Evaluate criterion
        loss = criterion(affinities, ground_truth).sum()
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)
        self.assertEqual(affinities.grad.size(1), 2)
        self.assertEqual(affinities.grad.dim(), 5)

    def test_malis(self):
        affinities, ground_truth = self.generate_test_data()
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        malis = Malis(constrained=True)
        # Forward and back
        loss = malis(affinities, ground_truth)
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)

    def test_malis_2d3d(self):
        affinities, ground_truth = self.generate_test_data(generate_2d=False,
                                                           reduce_num_affinity_channels=True)
        # Convert to variables
        affinities = Variable(torch.from_numpy(affinities), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth))
        # Build criterion
        malis = Malis(constrained=True, malis_dim=2)
        # Forward and back
        loss = malis(affinities, ground_truth)
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)
        self.assertEqual(affinities.grad.size(1), 2)
        self.assertEqual(affinities.grad.dim(), 5)

    def test_malis_gpu(self):
        if not torch.cuda.is_available():
            return
        affinities, ground_truth = self.generate_test_data()
        # Convert to variables and transfer to GPU
        affinities = Variable(torch.from_numpy(affinities).cuda(), requires_grad=True)
        ground_truth = Variable(torch.from_numpy(ground_truth).cuda())
        # Build criterion
        malis = Malis(constrained=True)
        # Forward and back
        loss = malis(affinities, ground_truth)
        loss.backward()
        # Validate
        self.assertIsNotNone(affinities.grad)
        self.assertFalse(loss.data.is_cuda)
        self.assertTrue(affinities.grad.data.is_cuda)


if __name__ == '__main__':
    TestMalis().test_malis_auto_loss_3d()
    # unittest.main()
